"""Источники звука: микрофон и replay WAV с одинаковым форматом блоков.

Формат блока (контракт CLAUDE.md, раздел 3): numpy float32, моно,
shape (block_size,), sample_rate 16000, значения в [-1, 1].
"""

import argparse
import logging
import queue
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd
import soundfile as sf

from qat.logging_setup import setup_logging

logger = logging.getLogger(__name__)

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


class MicrophoneSource:
    """Захват с микрофона. Использовать как контекстный менеджер:

        with MicrophoneSource() as mic:
            for block in mic: ...
    """

    def __init__(self, block_size: int = BLOCK_SIZE, device: int | str | None = None):
        self.sample_rate = SAMPLE_RATE
        self.block_size = block_size
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=block_size,
            device=device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, time, status) -> None:
        if status:
            logger.warning("микрофон: %s", status)
        self._queue.put(_to_mono(indata.copy()))

    def __enter__(self) -> "MicrophoneSource":
        self._stream.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stream.stop()
        self._stream.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        while self._stream.active:
            try:
                yield self._queue.get(timeout=0.5)
            except queue.Empty:
                continue


def _main() -> None:
    """Ручная проверка микрофона: раз в секунду печатает число блоков и RMS."""
    parser = argparse.ArgumentParser(description=_main.__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    setup_logging()

    per_second = SAMPLE_RATE // BLOCK_SIZE
    total = int(args.seconds * per_second)
    logger.info("устройство ввода: %s", sd.query_devices(kind="input")["name"])
    with MicrophoneSource() as mic:
        second: list[np.ndarray] = []
        for i, block in enumerate(mic, start=1):
            second.append(block)
            if len(second) == per_second:
                rms = float(np.sqrt(np.mean(np.concatenate(second) ** 2)))
                logger.info("блоков %d, RMS %.4f", i, rms)
                second.clear()
            if i >= total:
                break


if __name__ == "__main__":
    _main()
