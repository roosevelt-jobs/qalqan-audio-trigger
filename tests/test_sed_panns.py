from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qat.detectors.base import Detector
from qat.detectors.panns_cnn14 import Cnn14_16k
from qat.detectors.registry import make_detector
from qat.detectors.sed_panns import (
    SCREAM_SHOUT_CLASSES,
    WEIGHTS_PATH,
    SedPannsDetector,
    load_labels,
    load_model,
)

SR = 16000
SPEECH_WAV = Path(__file__).parent / "data" / "speech_ru.wav"
needs_weights = pytest.mark.skipif(
    not WEIGHTS_PATH.is_file(), reason="нет весов PANNs, см. models/panns/README.md"
)


@pytest.fixture(scope="module")
def random_detector():
    """Детектор на модели со случайными весами: проверяет код без 359 МБ весов."""
    return SedPannsDetector(model=Cnn14_16k())


@pytest.fixture(scope="module")
def real_detector():
    return SedPannsDetector()


def test_labels_contain_target_classes():
    labels = load_labels()
    assert len(labels) == 527
    assert labels[0] == "Speech"
    assert all(c in labels for c in SCREAM_SHOUT_CLASSES)


def test_unknown_class_rejected():
    with pytest.raises(ValueError, match="Screamin"):
        SedPannsDetector(classes=["Screamin"], model=Cnn14_16k())


def test_random_model_shapes_and_range(random_detector):
    det = random_detector
    window = np.random.default_rng(0).uniform(-0.5, 0.5, SR).astype(np.float32)

    assert isinstance(det, Detector)
    assert det.name == "sed_panns"
    probs = det.class_probs(window)
    assert probs.shape == (527,)
    score = det.process(window)
    assert 0.0 <= score <= 1.0
    assert det.process(window) == score  # детерминированно в режиме eval


def test_missing_and_wrong_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        load_model(tmp_path / "nope.pth")
    fake = tmp_path / "fake.pth"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        load_model(fake)


def test_registry():
    assert make_detector("energy").name == "energy"
    with pytest.raises(ValueError, match="неизвестный"):
        make_detector("nope")


@needs_weights
def test_real_model_silence_is_low(real_detector):
    assert real_detector.process(np.zeros(SR, dtype=np.float32)) < 0.1


@needs_weights
def test_real_model_recognizes_speech(real_detector):
    speech, sr = sf.read(str(SPEECH_WAV), dtype="float32")
    assert sr == SR
    probs = real_detector.class_probs(speech[2 * SR : 5 * SR])

    assert real_detector.labels[int(probs.argmax())] == "Speech"
