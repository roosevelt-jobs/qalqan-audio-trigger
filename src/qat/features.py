"""Признаки модели A: log-mel, 40 полос, окно 25 мс, шаг 10 мс (контракт CLAUDE.md, раздел 3).

ОДИН код признаков для обучения (qat.train) и для работы детектора (qat.detectors.kws).
Дублировать вычисление признаков в другом месте запрещено.
"""

import librosa
import numpy as np

from qat.runtime.audio_io import SAMPLE_RATE

N_MELS = 40
WIN_LENGTH = 400  # 25 мс при 16 кГц
HOP_LENGTH = 160  # 10 мс
N_FFT = 512
FMIN, FMAX = 20.0, 8000.0
LOG_FLOOR = 1e-6

WINDOW_SEC = 1.5  # окно модели A
WINDOW_SAMPLES = round(WINDOW_SEC * SAMPLE_RATE)
N_FRAMES = 1 + WINDOW_SAMPLES // HOP_LENGTH  # 151 кадр (center=True)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """(samples,) float32 16 кГц -> (кадры, 40) float32, натуральный логарифм мощности."""
    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32), sr=SAMPLE_RATE, n_fft=N_FFT,
        win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        center=True, power=2.0,
    )
    return np.log(mel + LOG_FLOOR).T.astype(np.float32)


def fit_window(audio: np.ndarray, start: int) -> np.ndarray:
    """Окно ровно WINDOW_SAMPLES с позиции start (start < 0 — тишина перед звуком)."""
    out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    src_from, dst_from = max(0, start), max(0, -start)
    n = max(0, min(len(audio) - src_from, WINDOW_SAMPLES - dst_from))
    out[dst_from : dst_from + n] = audio[src_from : src_from + n]
    return out
