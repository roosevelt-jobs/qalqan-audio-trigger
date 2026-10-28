"""Конвейер реального времени: источник -> буфер -> пробуждение -> детектор -> поток оценок
-> постобработка -> события тревоги.

Концы окон лежат на фиксированной сетке, кратной hop детектора (в сэмплах от старта потока),
независимо от размера блока — так позиции окон воспроизводимы.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, NamedTuple

from qat.detectors.base import Detector
from qat.logging_setup import setup_logging
from qat.runtime.audio_io import AudioSource, WavReplaySource
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import Wake

if TYPE_CHECKING:  # postproc и events сами импортируют этот модуль
    from qat.runtime.events import AlarmEvent, EventWriter
    from qat.runtime.postproc import PostProcessor


class Score(NamedTuple):
    sample: int  # номер сэмпла конца окна от старта потока
    score: float


def _scores_per_block_multi(
    source: AudioSource,
    detectors: Sequence[Detector],
    wake: Wake,
    buffer: RingBuffer,
    wakes: Sequence[Wake | None] | None = None,
) -> Iterator[list[list[Score]]]:
    """Для каждого блока источника и каждого детектора: оценки окон, закончившихся в блоке.

    Источник и буфер общие; у каждого детектора своя сетка окон. Пробуждение общее (wake),
    но детектору можно дать своё: wakes[i] не None (S14a: B — только по громкости).
    """
    grids = []
    for det in detectors:
        window = round(det.window_sec * buffer.sample_rate)
        hop = round(det.hop_sec * buffer.sample_rate)
        if window > buffer.capacity:
            raise ValueError(f"окно {window} сэмплов больше буфера {buffer.capacity}")
        grids.append([window, hop, math.ceil(window / hop) * hop])  # окно, шаг, конец след. окна
    own = list(wakes) if wakes is not None else [None] * len(detectors)
    if len(own) != len(detectors):
        raise ValueError("wakes: по одному на детектор (None — общее пробуждение)")
    was_active = [False] * len(detectors)

    for block in source:
        buffer.write(block)
        # ASSUMPTION: окно оценивается, если wake активен на блоке, где пересекается граница шага.
        shared = wake.process(block)
        actives = [w.process(block) if w is not None else shared for w in own]

        per_detector = []
        for i, (det, grid) in enumerate(zip(detectors, grids, strict=True)):
            active = actives[i]
            if was_active[i] and not active:
                # ASSUMPTION: после паузы детектор начинает без накопленного контекста.
                det.reset()
            was_active[i] = active
            window, hop, next_end = grid
            scores = []
            while buffer.total_written >= next_end:
                if active:
                    audio = buffer.extract(next_end - window, next_end)
                    scores.append(Score(next_end, det.process(audio)))
                next_end += hop
            grid[2] = next_end
            per_detector.append(scores)
        yield per_detector


def _scores_per_block(
    source: AudioSource, detector: Detector, wake: Wake, buffer: RingBuffer
) -> Iterator[list[Score]]:
    for per_detector in _scores_per_block_multi(source, [detector], wake, buffer):
        yield per_detector[0]


def run_scores(
    source: AudioSource, detector: Detector, wake: Wake, buffer: RingBuffer
) -> Iterator[Score]:
    for scores in _scores_per_block(source, detector, wake, buffer):
        yield from scores


def run_events(
    source: AudioSource,
    detector: Detector,
    wake: Wake,
    buffer: RingBuffer,
    postproc: PostProcessor,
    writer: EventWriter,
) -> Iterator[AlarmEvent]:
    """События в два этапа: сразу при срабатывании (без клипа), затем с готовым клипом."""
    yield from run_events_multi(source, [detector], wake, buffer, [postproc], writer)


def run_events_multi(
    source: AudioSource,
    detectors: Sequence[Detector],
    wake: Wake,
    buffer: RingBuffer,
    postprocs: Sequence[PostProcessor],
    writer: EventWriter,
    wakes: Sequence[Wake | None] | None = None,
) -> Iterator[AlarmEvent]:
    """Несколько детекторов параллельно, тревога от любого («любой из двух»).

    Срабатывания разных детекторов близко по времени writer сливает в одну тревогу:
    то же событие выдаётся ещё раз с обновлённым составом детекторов.
    wakes — своё пробуждение для детектора (None — общее wake).
    """
    for per_detector in _scores_per_block_multi(source, detectors, wake, buffer, wakes):
        for det, postproc, scores in zip(detectors, postprocs, per_detector, strict=True):
            for score in scores:
                trigger = postproc.process(score)
                if trigger is not None and (event := writer.on_trigger(trigger, det)):
                    yield event
        yield from writer.poll()
    yield from writer.flush()


def _main() -> None:
    """Прогон WAV через конвейер: события тревоги и клипы в events/."""
    from qat.detectors.registry import (
        DETECTORS,
        make_detector,
        make_postproc,
        make_wake,
        parse_detector_list,
    )
    from qat.runtime.events import EVENTS_DIR, MERGE_WINDOW_SEC, EventWriter

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--detector", default="energy",
                        help=f"имя или список через запятую: {', '.join(sorted(DETECTORS))}")
    parser.add_argument("--threshold", type=float, default=None,
                        help="один порог для всех детекторов (по умолчанию — из реестра)")
    parser.add_argument("--events-dir", default=str(EVENTS_DIR))
    args = parser.parse_args()
    setup_logging()

    source = WavReplaySource(args.wav)
    detectors = [make_detector(n) for n in parse_detector_list(args.detector)]
    buffer = RingBuffer()
    postprocs = [make_postproc(d, buffer.sample_rate, args.threshold) for d in detectors]
    writer = EventWriter(detectors, buffer, args.events_dir, merge_window_sec=MERGE_WINDOW_SEC)
    wakes = [make_wake(d) for d in detectors]
    for event in run_events_multi(source, detectors, Wake(), buffer, postprocs, writer, wakes):
        print(json.dumps(event.to_json(), ensure_ascii=False))


if __name__ == "__main__":
    _main()
