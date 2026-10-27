from pathlib import Path

import numpy as np
import pytest

from qat.runtime.audio_io import BLOCK_SIZE, SAMPLE_RATE, WavReplaySource
from qat.runtime.wake import VAD_CHUNK, SileroVad, Wake

# Фраза голосом Milena (macOS say), 2 с тишины по краям; речь примерно 2.0–5.9 с.
SPEECH_WAV = Path(__file__).parent / "data" / "speech_ru.wav"


class FakeVad:
    """Заглушка VAD: запоминает куски и возвращает заданную вероятность."""

    def __init__(self, prob=0.0):
        self.prob = prob
        self.chunks = []

    def __call__(self, chunk):
        assert chunk.shape == (VAD_CHUNK,)
        self.chunks.append(chunk.copy())
        return self.prob

    def reset(self):
        pass


def blocks_of(signal):
    return [signal[i : i + BLOCK_SIZE] for i in range(0, len(signal) - BLOCK_SIZE + 1, BLOCK_SIZE)]


def noise(seconds, dbfs, seed=0):
    rng = np.random.default_rng(seed)
    n = round(seconds * SAMPLE_RATE)
    return (rng.standard_normal(n) * 10 ** (dbfs / 20)).astype(np.float32)


@pytest.fixture(scope="module")
def silero():
    return SileroVad()


def test_silence_is_inactive_without_vad_calls():
    vad = FakeVad(prob=1.0)
    wake = Wake(vad=vad)

    results = [wake.process(b) for b in blocks_of(np.zeros(SAMPLE_RATE * 2, dtype=np.float32))]

    assert not any(results)
    assert vad.chunks == []


def test_loud_sound_wakes_without_vad():
    vad = FakeVad(prob=0.0)
    wake = Wake(vad=vad, hangover_sec=0.0)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # около -9 dBFS

    results = [wake.process(b) for b in blocks_of(tone)]

    assert all(results)
    assert vad.chunks == []


def test_rechunking_480_to_512_keeps_all_samples():
    vad = FakeVad(prob=0.0)
    wake = Wake(vad=vad, loud_threshold_dbfs=0.0)
    signal = noise(1.0, -30)  # между порогами: решает VAD

    for b in blocks_of(signal):
        wake.process(b)

    fed = np.concatenate(vad.chunks)
    n_full = (len(signal) // BLOCK_SIZE * BLOCK_SIZE) // VAD_CHUNK * VAD_CHUNK
    np.testing.assert_array_equal(fed, signal[:n_full])


def test_hangover_keeps_active_after_sound():
    vad = FakeVad()
    wake = Wake(vad=vad, hangover_sec=0.3)
    loud = np.full(BLOCK_SIZE, 0.5, dtype=np.float32)
    silent = np.zeros(BLOCK_SIZE, dtype=np.float32)

    assert wake.process(loud)
    after = [wake.process(silent) for _ in range(20)]

    # 0.3 с = 4800 сэмплов = 10 блоков по 480
    assert after == [True] * 10 + [False] * 10


def test_noise_between_thresholds_is_rejected_by_vad(silero):
    wake = Wake(vad=silero)

    results = [wake.process(b) for b in blocks_of(noise(5.0, -35))]

    assert not any(results)


def test_speech_wakes_through_vad(silero):
    # Громкий порог отключён, шум -45 dBFS выше порога тишины: везде решает VAD.
    wake = Wake(vad=silero, loud_threshold_dbfs=0.0)
    speech = np.concatenate(list(WavReplaySource(SPEECH_WAV)))
    signal = np.clip(speech + noise(len(speech) / SAMPLE_RATE, -45), -1, 1).astype(np.float32)

    active = np.array([wake.process(b) for b in blocks_of(signal)])
    t = np.arange(len(active)) * BLOCK_SIZE / SAMPLE_RATE

    assert not active[t < 1.9].any()
    assert active[(t >= 2.3) & (t < 5.5)].mean() >= 0.95
    assert not active[t >= 7.0].any()  # конец речи ~5.9 с + hangover 1 с
