"""ЗАГЛУШКА: «детектор» по громкости окна. Не детектирует тревогу, нужен для отладки конвейера.

Оценка: RMS окна в dBFS, линейно -60 dBFS -> 0, 0 dBFS -> 1, с обрезкой в [0, 1].
"""

import numpy as np

from qat.runtime.wake import block_dbfs

FLOOR_DBFS = -60.0


class EnergyDetector:
    name = "energy"
    version = "0.1.0"
    # ASSUMPTION: окно 0.5 с и шаг 0.1 с для заглушки.
    window_sec = 0.5
    hop_sec = 0.1

    def process(self, audio_window: np.ndarray) -> float:
        score = (block_dbfs(audio_window) - FLOOR_DBFS) / -FLOOR_DBFS
        return float(np.clip(score, 0.0, 1.0))

    def reset(self) -> None:
        pass
