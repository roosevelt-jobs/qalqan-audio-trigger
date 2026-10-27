"""Постобработка потока оценок: сглаживание -> порог -> N подряд -> дебаунс.

Один Trigger на эпизод превышения порога; между срабатываниями не меньше debounce_sec.
Разрыв в сетке окон (wake был неактивен) сбрасывает сглаживание и счётчик.
"""

from collections import deque
from typing import NamedTuple

from qat.runtime.audio_io import SAMPLE_RATE
from qat.runtime.pipeline import Score


class Trigger(NamedTuple):
    sample: int  # конец окна, на котором сработало (от старта потока)
    score: float  # сглаженная оценка в момент срабатывания
    threshold: float


class PostProcessor:
    # ASSUMPTION: порог 0.8, сглаживание по 3 окнам, 2 подряд, дебаунс 5 с — стартовые значения;
    # рабочий порог выбирается по DET-кривой (S8).

    def __init__(
        self,
        hop_samples: int,
        threshold: float = 0.8,
        smooth_windows: int = 3,
        consecutive: int = 2,
        debounce_sec: float = 5.0,
        sample_rate: int = SAMPLE_RATE,
    ):
        if hop_samples <= 0:
            raise ValueError(f"hop_samples должен быть > 0, получено {hop_samples}")
        if smooth_windows < 1:
            raise ValueError(f"smooth_windows должен быть >= 1, получено {smooth_windows}")
        if consecutive < 1:
            raise ValueError(f"consecutive должен быть >= 1, получено {consecutive}")
        self.hop_samples = hop_samples
        self.threshold = threshold
        self.smooth_windows = smooth_windows
        self.consecutive = consecutive
        self.debounce_samples = round(debounce_sec * sample_rate)
        self.reset()

    def reset(self) -> None:
        self._history: deque[float] = deque(maxlen=self.smooth_windows)
        self._prev_sample: int | None = None
        self._count = 0
        self._episode_fired = False
        self._last_trigger: int | None = None

    def process(self, score: Score) -> Trigger | None:
        if self._prev_sample is not None and score.sample - self._prev_sample != self.hop_samples:
            self._start_new_run()
        self._prev_sample = score.sample

        self._history.append(score.score)
        # ASSUMPTION: в начале серии среднее берётся по тем окнам, что уже есть.
        smoothed = sum(self._history) / len(self._history)

        if smoothed < self.threshold:
            self._count = 0
            self._episode_fired = False
            return None

        self._count += 1
        if self._count < self.consecutive or self._episode_fired:
            return None
        self._episode_fired = True
        if self._last_trigger is not None and (
            score.sample - self._last_trigger < self.debounce_samples
        ):
            return None
        self._last_trigger = score.sample
        return Trigger(score.sample, smoothed, self.threshold)

    def _start_new_run(self) -> None:
        self._history.clear()
        self._count = 0
        self._episode_fired = False
