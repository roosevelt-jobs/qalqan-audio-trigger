import numpy as np

from qat.detectors.base import Detector
from qat.detectors.energy import EnergyDetector


def test_energy_detector_matches_protocol():
    det = EnergyDetector()
    assert isinstance(det, Detector)
    assert det.name == "energy"
    assert det.window_sec > 0 and det.hop_sec > 0


def test_energy_silence_and_full_scale():
    det = EnergyDetector()
    n = round(det.window_sec * 16000)

    assert det.process(np.zeros(n, dtype=np.float32)) == 0.0
    assert det.process(np.ones(n, dtype=np.float32)) == 1.0


def test_energy_score_grows_with_level_and_stays_in_range():
    det = EnergyDetector()
    rng = np.random.default_rng(0)
    base = rng.uniform(-1, 1, round(det.window_sec * 16000)).astype(np.float32)

    scores = [det.process(base * gain) for gain in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)]

    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores)
    assert scores[0] < scores[-1]
