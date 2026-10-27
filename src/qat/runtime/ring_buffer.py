"""Кольцевой буфер последних N секунд аудио для вырезания клипов событий.

Время адресуется номером сэмпла от старта потока: сэмпл 0 — первый записанный.
"""

import numpy as np

from qat.runtime.audio_io import SAMPLE_RATE


class RingBuffer:
    # ASSUMPTION: пишет и читает один поток (цикл конвейера), поэтому без блокировок.

    def __init__(self, capacity_sec: float = 30.0, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.capacity = round(capacity_sec * sample_rate)
        if self.capacity <= 0:
            raise ValueError(f"ёмкость должна быть > 0, получено {capacity_sec} с")
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self.total_written = 0

    def write(self, block: np.ndarray) -> None:
        if block.dtype != np.float32 or block.ndim != 1:
            raise ValueError(
                f"нужен одномерный float32, получено {block.dtype} shape {block.shape}"
            )
        n = len(block)
        data = block[-self.capacity :] if n > self.capacity else block
        pos = (self.total_written + n - len(data)) % self.capacity
        first = min(len(data), self.capacity - pos)
        self._buf[pos : pos + first] = data[:first]
        self._buf[: len(data) - first] = data[first:]
        self.total_written += n

    def available_range(self) -> tuple[int, int]:
        """[start, end) номеров сэмплов, которые ещё лежат в буфере."""
        return max(0, self.total_written - self.capacity), self.total_written

    def extract(self, start: int, end: int) -> np.ndarray:
        """Копия сэмплов [start, end). ValueError, если их нет в буфере."""
        oldest, newest = self.available_range()
        if start > end:
            raise ValueError(f"start {start} > end {end}")
        if start < 0:
            raise ValueError(f"сэмпл {start} до старта потока (сэмпл 0)")
        if start < oldest:
            raise ValueError(f"сэмплы с {start} уже вытеснены, доступно с {oldest}")
        if end > newest:
            raise ValueError(f"сэмплы до {end} ещё не записаны, записано {newest}")
        n = end - start
        pos = start % self.capacity
        first = min(n, self.capacity - pos)
        return np.concatenate([self._buf[pos : pos + first], self._buf[: n - first]])

    def clip(
        self, center: int, start_offset_sec: float = -5.0, end_offset_sec: float = 5.0
    ) -> np.ndarray:
        """Клип вокруг сэмпла center со смещениями в секундах (как в JSON события)."""
        start = center + round(start_offset_sec * self.sample_rate)
        end = center + round(end_offset_sec * self.sample_rate)
        return self.extract(start, end)
