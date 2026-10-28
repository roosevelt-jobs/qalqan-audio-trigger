"""Аугментация клипов для обучения модели A. Детерминирована по seed.

Цепочка (каждый шаг — с вероятностью): темп/высота → «крик» → эхо помещения → полоса
нагрудного микрофона → фоновый шум с заданным SNR → громкость.
Параметры каждого применения возвращаются, чтобы записать их в манифест.
"""

from collections.abc import Sequence

import librosa
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

from qat.runtime.audio_io import SAMPLE_RATE


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Шум, подогнанный по длине и громкости так, чтобы SNR был ровно snr_db."""
    if len(noise) < len(signal):
        noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
    noise = noise[: len(signal)]
    gain = rms(signal) / (rms(noise) * 10 ** (snr_db / 20))
    return signal + gain * noise


def synthetic_rir(rt60: float, rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Импульсный отклик комнаты: экспоненциально затухающий шум (затухание 60 дБ за rt60)."""
    n = max(1, int(rt60 * sr))
    t = np.arange(n) / sr
    rir = rng.standard_normal(n) * np.exp(-6.908 * t / rt60)  # ln(1000) = 6.908
    rir[0] = 1.0
    return (rir / np.abs(rir).max()).astype(np.float32)


def shout(x: np.ndarray, semitones: float, drive_db: float) -> np.ndarray:
    """Грубая имитация крика: выше тон, сильнее, с мягким перегрузом (tanh)."""
    y = librosa.effects.pitch_shift(x, sr=SAMPLE_RATE, n_steps=semitones)
    y = np.tanh(y * 10 ** (drive_db / 20))
    return y / (np.abs(y).max() + 1e-9) * 0.9


def bandpass(x: np.ndarray, low: float, high: float) -> np.ndarray:
    sos = butter(4, [low, high], btype="band", fs=SAMPLE_RATE, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def augment(
    x: np.ndarray, noises: Sequence[np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, dict]:
    """Одна случайная аугментация клипа. Возвращает (аудио float32 в [-1, 1], параметры)."""
    p: dict = {}
    y = x.astype(np.float32)
    if rng.random() < 0.5:
        p["stretch"] = float(rng.uniform(0.85, 1.15))
        y = librosa.effects.time_stretch(y, rate=p["stretch"])
    if rng.random() < 0.4:
        p["pitch"] = float(rng.uniform(-3, 3))
        y = librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=p["pitch"])
    if rng.random() < 0.3:
        p["shout"] = {"semitones": float(rng.uniform(2, 5)), "drive_db": float(rng.uniform(6, 18))}
        y = shout(y, **p["shout"])
    if rng.random() < 0.5:
        p["rt60"] = float(rng.uniform(0.15, 0.8))
        y = fftconvolve(y, synthetic_rir(p["rt60"], rng))[: len(y)].astype(np.float32)
    if rng.random() < 0.5:
        p["band"] = [float(rng.uniform(80, 300)), float(rng.uniform(3500, 7500))]
        y = bandpass(y, *p["band"])
    if noises and rng.random() < 0.8:
        i = int(rng.integers(len(noises)))
        p["noise"] = {"index": i, "snr_db": float(rng.uniform(0, 20))}
        y = mix_at_snr(y, noises[i], p["noise"]["snr_db"])
    p["gain_db"] = float(rng.uniform(-20, 0))
    peak = np.abs(y).max() + 1e-9
    y = y / peak * 0.9 * 10 ** (p["gain_db"] / 20)
    return np.clip(y, -1.0, 1.0).astype(np.float32), p
