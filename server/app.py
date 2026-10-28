"""Мини-сервер оператора: FastAPI + WebSocket.

Конвейер работает в фоновом потоке этого же процесса и передаёт события в Hub,
Hub рассылает их по WebSocket на экран оператора. Решения оператора пишутся в JSON
события и в журнал events/decisions.jsonl.
"""

import argparse
import asyncio
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from qat.detectors.registry import (
    DETECTORS,
    make_detector,
    make_postproc,
    make_wake,
    parse_detector_list,
)
from qat.logging_setup import setup_logging
from qat.runtime.audio_io import AudioSource, MicrophoneSource, WavReplaySource
from qat.runtime.events import EVENTS_DIR, MERGE_WINDOW_SEC, EventWriter
from qat.runtime.pipeline import run_events_multi
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import Wake

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# ASSUMPTION: в памяти и в истории для новых подключений — последние 50 событий.
HISTORY_SIZE = 50
# Телеметрия FastAPI (OpenTelemetry) выключена целиком: проект офлайн, CLAUDE.md раздел 10.
NO_TELEMETRY = {
    "tracing": False,
    "metrics": False,
    "logs": False,
    "operation_spans": False,
    "auto_configure": False,
}


class Decision(BaseModel):
    # ASSUMPTION: значения решения оператора — "confirmed" / "rejected".
    decision: Literal["confirmed", "rejected"]


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


class Hub:
    """Состояние событий и рассылка по WebSocket. Методы вызываются в цикле событий сервера."""

    def __init__(self, events_dir: Path):
        self.events_dir = events_dir
        self.events: OrderedDict[str, dict] = OrderedDict()
        self.decisions: dict[str, str] = {}
        self.clients: set[WebSocket] = set()

    async def publish(self, event: dict) -> None:
        event_id = event["event_id"]
        decided = self.decisions.get(event_id)
        if decided is not None and event.get("operator_decision") != decided:
            # Второй этап события пришёл после решения: EventWriter записал JSON без решения.
            event = {**event, "operator_decision": decided}
            _write_json_atomic(self.events_dir / f"{event_id}.json", event)
        self.events[event_id] = event
        self.events.move_to_end(event_id)
        while len(self.events) > HISTORY_SIZE:
            self.events.popitem(last=False)
        await self._broadcast({"type": "event", "event": event})

    async def decide(self, event_id: str, decision: str) -> dict:
        event = self.events.get(event_id)
        if event is None:
            raise HTTPException(404, "событие не найдено")
        if event_id in self.decisions:
            # ASSUMPTION: одно решение на событие, повторное — 409.
            raise HTTPException(409, "решение по событию уже принято")
        self.decisions[event_id] = decision
        event = {**event, "operator_decision": decision}
        self.events[event_id] = event
        _write_json_atomic(self.events_dir / f"{event_id}.json", event)
        record = {
            "event_id": event_id,
            "decision": decision,
            "decided_at_utc": datetime.now(UTC).isoformat(),
        }
        with (self.events_dir / "decisions.jsonl").open("a") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
        await self._broadcast({"type": "event", "event": event})
        return event

    async def _broadcast(self, message: dict) -> None:
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                self.clients.discard(ws)


def _realtime(source: AudioSource) -> Iterator:
    """Отдаёт блоки в темпе реального времени (для демо на WAV)."""
    period = source.block_size / source.sample_rate
    start = time.monotonic()
    for i, block in enumerate(source):
        delay = start + i * period - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        yield block


def _run_pipeline(
    make_source: Callable[[], AudioSource],
    realtime: bool,
    threshold: float | None,
    events_dir: Path,
    hub: Hub,
    loop: asyncio.AbstractEventLoop,
    stop: threading.Event,
    detector_names: str = "energy",
) -> None:
    try:
        detectors = [make_detector(n) for n in parse_detector_list(detector_names)]
        buffer = RingBuffer()
        postprocs = [make_postproc(d, buffer.sample_rate, threshold) for d in detectors]
        wakes = [make_wake(d) for d in detectors]
        writer = EventWriter(detectors, buffer, events_dir, merge_window_sec=MERGE_WINDOW_SEC)
        source = make_source()
        # Микрофон — контекстный менеджер (открывает поток), replay — нет.
        ctx = source if hasattr(source, "__enter__") else nullcontext(source)
        logger.info("конвейер запущен, детекторы: %s",
                    ", ".join(f"{d.name} {d.version} (порог {p.threshold})"
                              for d, p in zip(detectors, postprocs, strict=True)))
        with ctx as src:
            blocks = _realtime(src) if realtime else src

            def until_stopped():
                for block in blocks:
                    if stop.is_set():
                        return
                    yield block

            events = run_events_multi(
                until_stopped(), detectors, Wake(), buffer, postprocs, writer, wakes
            )
            for event in events:
                # Ждём доставки, чтобы этапы события не перепутались местами.
                asyncio.run_coroutine_threadsafe(hub.publish(event.to_json()), loop).result()
        logger.info("конвейер остановлен")
    except Exception:
        logger.exception("конвейер упал")


def create_app(
    make_source: Callable[[], AudioSource] | None = None,
    realtime: bool = False,
    threshold: float | None = None,
    events_dir: str | Path = EVENTS_DIR,
    detector: str = "energy",
) -> FastAPI:
    """detector — имя или список через запятую; threshold=None — пороги из реестра."""
    events_dir = Path(events_dir)
    events_dir.mkdir(parents=True, exist_ok=True)
    hub = Hub(events_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()
        thread = None
        if make_source is not None:
            args = (make_source, realtime, threshold, events_dir, hub,
                    asyncio.get_running_loop(), stop, detector)
            thread = threading.Thread(target=_run_pipeline, args=args, daemon=True)
            thread.start()
        yield
        stop.set()

    app = FastAPI(lifespan=lifespan, telemetry=NO_TELEMETRY)
    app.state.hub = hub

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        await ws.send_json({"type": "history", "events": list(hub.events.values())})
        hub.clients.add(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(ws)

    @app.get("/clips/{event_id}.wav")
    async def clip(event_id: str):
        try:
            canonical = str(uuid.UUID(event_id))
        except ValueError:
            raise HTTPException(404, "клип не найден") from None
        path = events_dir / f"{canonical}.wav"
        if canonical != event_id or not path.is_file():
            raise HTTPException(404, "клип не найден")
        return FileResponse(path, media_type="audio/wav")

    @app.post("/events/{event_id}/decision")
    async def decide(event_id: str, body: Decision):
        return await hub.decide(event_id, body.decision)

    return app


def _main() -> None:
    parser = argparse.ArgumentParser(description="Сервер и экран оператора")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true", help="микрофон")
    src.add_argument("--wav", help="replay WAV-файла (16 кГц)")
    parser.add_argument("--detector", default="energy",
                        help=f"имя или список через запятую: {', '.join(sorted(DETECTORS))}")
    parser.add_argument("--realtime", action="store_true", help="replay в темпе реального времени")
    parser.add_argument("--threshold", type=float, default=None,
                        help="один порог для всех детекторов (по умолчанию — из реестра)")
    parser.add_argument("--events-dir", default=str(EVENTS_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    setup_logging()

    make_source = MicrophoneSource if args.mic else (lambda: WavReplaySource(args.wav))
    app = create_app(make_source, args.realtime, args.threshold, args.events_dir, args.detector)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    _main()
