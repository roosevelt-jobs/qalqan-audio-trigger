"""Конвейер реального времени: источник -> буфер -> пробуждение -> детектор -> поток оценок
-> постобработка -> события тревоги.

Концы окон лежат на фиксированной сетке, кратной hop детектора (в сэмплах от старта потока),
независимо от размера блока — так позиции окон воспроизводимы.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
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


def _scores_per_block(
    source: AudioSource, detector: Detector, wake: Wake, buffer: RingBuffer
) -> Iterator[list[Score]]:
    """Для каждого блока источника: оценки окон, закончившихся в этом блоке."""
    window = round(detector.window_sec * buffer.sample_rate)
    hop = round(detector.hop_sec * buffer.sample_rate)
    if window > buffer.capacity:
        raise ValueError(f"окно {window} сэмплов больше буфера {buffer.capacity}")
    next_end = math.ceil(window / hop) * hop
    was_active = False

    for block in source:
        buffer.write(block)
        # ASSUMPTION: окно оценивается, если wake активен на блоке, где пересекается граница шага.
        active = wake.process(block)
        if was_active and not active:
            # ASSUMPTION: после паузы детектор начинает без накопленного контекста.
            detector.reset()
        was_active = active

        scores = []
        while buffer.total_written >= next_end:
            if active:
                audio = buffer.extract(next_end - window, next_end)
                scores.append(Score(next_end, detector.process(audio)))
            next_end += hop
        yield scores


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
    for scores in _scores_per_block(source, detector, wake, buffer):
        for score in scores:
            trigger = postproc.process(score)
            if trigger is not None:
                yield writer.on_trigger(trigger)
        yield from writer.poll()
    yield from writer.flush()


def _main() -> None:
    """Прогон WAV через конвейер: события тревоги и клипы в events/."""
    from qat.detectors.registry import DETECTORS, make_detector
    from qat.runtime.events import EVENTS_DIR, EventWriter
    from qat.runtime.postproc import PostProcessor

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--detector", choices=sorted(DETECTORS), default="energy")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--events-dir", default=str(EVENTS_DIR))
    args = parser.parse_args()
    setup_logging()

    source = WavReplaySource(args.wav)
    detector = make_detector(args.detector)
    buffer = RingBuffer()
    hop = round(detector.hop_sec * buffer.sample_rate)
    postproc = PostProcessor(hop, threshold=args.threshold)
    writer = EventWriter(detector, buffer, args.events_dir)
    for event in run_events(source, detector, Wake(), buffer, postproc, writer):
        print(json.dumps(event.to_json(), ensure_ascii=False))


if __name__ == "__main__":
    _main()
