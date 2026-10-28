import numpy as np
import pytest

from qat.data.augment import augment, mix_at_snr, rms, shout, synthetic_rir

SR = 16000


def tone(seconds=1.0, f=440.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_mix_at_snr_exact():
    rng = np.random.default_rng(0)
    s, n = tone(), rng.standard_normal(SR // 3).astype(np.float32)  # шум короче — повторится

    for snr in (0.0, 10.0, 20.0):
        mixed = mix_at_snr(s, n, snr)
        noise_part = mixed - s
        assert len(mixed) == len(s)
        assert 20 * np.log10(rms(s) / rms(noise_part)) == pytest.approx(snr, abs=1e-6)


def test_rir_decays_60db_over_rt60():
    rir = synthetic_rir(0.4, np.random.default_rng(0))
    assert len(rir) == int(0.4 * SR)
    head, tail = np.abs(rir[: SR // 50]).max(), np.abs(rir[-SR // 50 :]).max()
    assert 20 * np.log10(head / tail) > 40  # к концу затухло сильно


def test_shout_raises_pitch_and_stays_in_range():
    y = shout(tone(f=200.0), semitones=4.0, drive_db=12.0)
    spec = np.abs(np.fft.rfft(y))
    f0 = np.argmax(spec) * SR / len(y)
    assert 240 < f0 < 270  # 200 Гц * 2^(4/12) ≈ 252 Гц
    assert np.abs(y).max() <= 0.9 + 1e-6


def test_augment_deterministic_and_valid():
    x = tone(1.5)
    noises = [np.random.default_rng(1).standard_normal(SR).astype(np.float32)]

    a, pa = augment(x, noises, np.random.default_rng(42))
    b, pb = augment(x, noises, np.random.default_rng(42))
    c, pc = augment(x, noises, np.random.default_rng(43))

    np.testing.assert_array_equal(a, b)
    assert pa == pb and pa != pc
    assert a.dtype == np.float32 and np.abs(a).max() <= 1.0
    assert "gain_db" in pa
