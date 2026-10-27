"""Источники звука: микрофон и replay WAV с одинаковым форматом блоков.

Формат блока (контракт CLAUDE.md, раздел 3): numpy float32, моно,
shape (block_size,), sample_rate 16000, значения в [-1, 1].
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
BLOCK_SIZE = 480  # 30 мс


class AudioSource(Protocol):
    sample_rate: int
    block_size: int

    def __iter__(self) -> Iterator[np.ndarray]: ...


def _to_mono(block: np.ndarray) -> np.ndarray:
    """(frames, channels) -> (frames,) усреднением каналов, float32 в [-1, 1]."""
    mono = block.mean(axis=1, dtype=np.float32)
    # ASSUMPTION: float-WAV может содержать значения за пределами [-1, 1]; обрезаем по контракту.
    return np.clip(mono, -1.0, 1.0)


class WavReplaySource:
    """Читает WAV блоками по block_size. Последний неполный блок дополняется нулями."""

    def __init__(self, path: str | Path, block_size: int = BLOCK_SIZE):
        self.path = Path(path)
        self.sample_rate = SAMPLE_RATE
        self.block_size = block_size
        info = sf.info(str(self.path))
        if info.samplerate != SAMPLE_RATE:
            raise ValueError(
                f"{self.path}: частота {info.samplerate} Гц, нужна {SAMPLE_RATE} Гц. "
                f"Конвертировать: afconvert -f WAVE -d LEF32@{SAMPLE_RATE} "
                f'"{self.path}" "{self.path.with_stem(self.path.stem + "_16k")}"'
            )

    def __iter__(self) -> Iterator[np.ndarray]:
        blocks = sf.blocks(
            str(self.path),
            blocksize=self.block_size,
            dtype="float32",
            always_2d=True,
            fill_value=0.0,
        )
        for block in blocks:
            yield _to_mono(block)
