"""Конвейер реального времени: источник -> буфер -> пробуждение -> детектор -> поток оценок.

Концы окон лежат на фиксированной сетке, кратной hop детектора (в сэмплах от старта потока),
независимо от размера блока — так позиции окон воспроизводимы.
"""

import argparse
import math
from collections.abc import Iterator
from typing import NamedTuple

from qat.detectors.base import Detector
from qat.logging_setup import setup_logging
from qat.runtime.audio_io import AudioSource, WavReplaySource
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import Wake


class Score(NamedTuple):
    sample: int  # номер сэмпла конца окна от старта потока
    score: float


def run_scores(
    source: AudioSource, detector: Detector, wake: Wake, buffer: RingBuffer
) -> Iterator[Score]:
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

        while buffer.total_written >= next_end:
            if active:
                audio = buffer.extract(next_end - window, next_end)
                yield Score(next_end, detector.process(audio))
            next_end += hop


def _main() -> None:
    """Прогон WAV через конвейер с заглушкой energy: максимум оценки по секундам."""
    from qat.detectors.energy import EnergyDetector

    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--wav", required=True)
    args = parser.parse_args()
    setup_logging()

    source = WavReplaySource(args.wav)
    buffer = RingBuffer()
    per_second: dict[int, float] = {}
    for s in run_scores(source, EnergyDetector(), Wake(), buffer):
        sec = s.sample // source.sample_rate
        per_second[sec] = max(per_second.get(sec, 0.0), s.score)

    for sec in range(math.ceil(buffer.total_written / source.sample_rate)):
        value = f"{per_second[sec]:.2f}" if sec in per_second else "-"
        print(f"{sec:3d} с  {value}")


if __name__ == "__main__":
    _main()
