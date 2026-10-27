"""Интерфейс детектора (контракт CLAUDE.md, раздел 3). Менять только с разрешения."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Detector(Protocol):
    name: str            # "kws", "sed_panns", "asr_keywords"
    version: str
    window_sec: float    # длина окна анализа
    hop_sec: float       # шаг окна
    def process(self, audio_window: np.ndarray) -> float: ...   # оценка 0..1 для одного окна
    def reset(self) -> None: ...
