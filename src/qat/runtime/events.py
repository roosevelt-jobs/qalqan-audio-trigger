"""События тревоги: JSON + аудиоклип из буфера + SHA-256 клипа.

Событие выдаётся в два этапа (согласовано для S6, чтобы тревога не ждала «+5 с»):
1. on_trigger — сразу при срабатывании: JSON с полями клипа = null.
2. poll / flush — когда хвост клипа записан: тот же event_id, клип, хеш и фактические смещения.
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf

from qat.detectors.base import Detector
from qat.runtime.postproc import Trigger
from qat.runtime.ring_buffer import RingBuffer

EVENTS_DIR = Path(__file__).resolve().parents[3] / "events"


@dataclass(frozen=True)
class AlarmEvent:
    event_id: str
    timestamp_utc: str
    detector: str
    detector_version: str
    score: float
    threshold: float
    clip_path: str | None = None
    clip_sha256: str | None = None
    clip_start_offset_sec: float | None = None
    clip_end_offset_sec: float | None = None
    operator_decision: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @property
    def clip_ready(self) -> bool:
        return self.clip_path is not None


class EventWriter:
    # ASSUMPTION: timestamp_utc — время срабатывания по часам компьютера
    # (при replay — время обработки, а не время внутри записи).

    def __init__(
        self,
        detector: Detector,
        buffer: RingBuffer,
        events_dir: str | Path = EVENTS_DIR,
        clip_before_sec: float = 5.0,
        clip_after_sec: float = 5.0,
    ):
        self.detector = detector
        self.buffer = buffer
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.before = round(clip_before_sec * buffer.sample_rate)
        self.after = round(clip_after_sec * buffer.sample_rate)
        self._pending: list[tuple[AlarmEvent, int]] = []

    def on_trigger(self, trigger: Trigger) -> AlarmEvent:
        """Этап 1: событие без клипа, сразу."""
        event = AlarmEvent(
            event_id=str(uuid.uuid4()),
            timestamp_utc=datetime.now(UTC).isoformat(),
            detector=self.detector.name,
            detector_version=self.detector.version,
            score=trigger.score,
            threshold=trigger.threshold,
        )
        self._write_json(event)
        self._pending.append((event, trigger.sample))
        return event

    def poll(self) -> list[AlarmEvent]:
        """Этап 2 для событий, у которых хвост клипа уже записан в буфер."""
        ready = [p for p in self._pending if p[1] + self.after <= self.buffer.total_written]
        return [self._finalize(event, center, center + self.after) for event, center in ready]

    def flush(self) -> list[AlarmEvent]:
        """Конец потока: закрыть все ожидающие события с тем аудио, что есть."""
        end = self.buffer.total_written
        pending = list(self._pending)
        return [self._finalize(e, center, min(center + self.after, end)) for e, center in pending]

    def _finalize(self, event: AlarmEvent, center: int, end: int) -> AlarmEvent:
        self._pending = [p for p in self._pending if p[0].event_id != event.event_id]
        oldest, _ = self.buffer.available_range()
        # Клип не дополняется тишиной: начало обрезается до доступного, смещения — фактические.
        start = max(center - self.before, oldest)
        audio = self.buffer.extract(start, end)

        clip = self.events_dir / f"{event.event_id}.wav"
        sf.write(str(clip), audio, self.buffer.sample_rate, subtype="FLOAT")
        sr = self.buffer.sample_rate
        done = replace(
            event,
            clip_path=f"{self.events_dir.name}/{clip.name}",
            clip_sha256=hashlib.sha256(clip.read_bytes()).hexdigest(),
            clip_start_offset_sec=(start - center) / sr,
            clip_end_offset_sec=(end - center) / sr,
        )
        self._write_json(done)
        return done

    def _write_json(self, event: AlarmEvent) -> None:
        path = self.events_dir / f"{event.event_id}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(event.to_json(), ensure_ascii=False, indent=2))
        tmp.replace(path)  # атомарно: читатель не увидит недописанный JSON
