import numpy as np
import pytest
import soundfile as sf

from qat.runtime.audio_io import BLOCK_SIZE, SAMPLE_RATE, WavReplaySource


def _write_wav(path, data, samplerate=SAMPLE_RATE):
    sf.write(str(path), data, samplerate, subtype="FLOAT")
    return path


def test_replay_mono_block_format(tmp_path):
    n = BLOCK_SIZE * 3 + 100
    data = (0.5 * np.sin(np.arange(n) / 10)).astype(np.float32)
    source = WavReplaySource(_write_wav(tmp_path / "mono.wav", data))

    blocks = list(source)

    assert source.sample_rate == SAMPLE_RATE
    assert len(blocks) == 4
    for block in blocks:
        assert block.dtype == np.float32
        assert block.shape == (BLOCK_SIZE,)
        assert np.all(np.abs(block) <= 1.0)
    np.testing.assert_allclose(np.concatenate(blocks)[:n], data, atol=1e-6)
    assert np.all(blocks[-1][100:] == 0.0)


def test_replay_stereo_is_mean_of_channels(tmp_path):
    n = BLOCK_SIZE * 2
    left = np.full(n, 0.6, dtype=np.float32)
    right = np.full(n, -0.2, dtype=np.float32)
    path = _write_wav(tmp_path / "stereo.wav", np.stack([left, right], axis=1))

    blocks = list(WavReplaySource(path))

    assert len(blocks) == 2
    for block in blocks:
        assert block.shape == (BLOCK_SIZE,)
        np.testing.assert_allclose(block, 0.2, atol=1e-6)


def test_replay_rejects_wrong_sample_rate(tmp_path):
    path = _write_wav(tmp_path / "rec48k.wav", np.zeros(4800, dtype=np.float32), 48000)

    with pytest.raises(ValueError, match="48000 Гц") as exc:
        WavReplaySource(path)

    assert "afconvert -f WAVE -d LEF32@16000" in str(exc.value)
