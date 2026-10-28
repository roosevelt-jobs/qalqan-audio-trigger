from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from qat.detectors.energy import EnergyDetector
from qat.eval.bench import _AlwaysAwake
from qat.eval.tuning import mask_scores, pick_best, split_groups, wake_activity
from qat.runtime.audio_io import WavReplaySource
from qat.runtime.pipeline import Score, run_scores
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import SileroVad, Wake

SPEECH_WAV = Path(__file__).parent / "data" / "speech_ru.wav"


@pytest.fixture(scope="module")
def vad():
    return SileroVad()


@pytest.mark.parametrize("loud", [-20.0, -35.0])
def test_mask_equals_real_pipeline_for_stateless_detector(tmp_path, vad, loud):
    # фраза + тихий шум между порогами, чтобы решал и VAD, и порог громкого звука
    speech, sr = sf.read(str(SPEECH_WAV), dtype="float32")
    rng = np.random.default_rng(0)
    signal = np.concatenate([speech, (rng.standard_normal(3 * sr) * 0.01).astype(np.float32)])
    path = tmp_path / "mix.wav"
    sf.write(str(path), signal, sr, subtype="FLOAT")
    det = EnergyDetector()  # без состояния, как sed_panns

    real = list(run_scores(WavReplaySource(path), det, Wake(loud_threshold_dbfs=loud, vad=vad),
                           RingBuffer()))
    everything = list(run_scores(WavReplaySource(path), det, _AlwaysAwake(), RingBuffer()))
    active = wake_activity(path, Wake(loud_threshold_dbfs=loud, vad=vad))

    assert mask_scores(everything, active) == real
    assert 0 < len(real) < len(everything)


def test_mask_scores_block_rule():
    scores = [Score(480, 0.1), Score(481, 0.2), Score(960, 0.3), Score(1440, 0.4)]
    active = np.array([True, False, True])

    # конец окна 480 — блок 0, 481 и 960 — блок 1, 1440 — блок 2
    assert mask_scores(scores, active) == [Score(480, 0.1), Score(1440, 0.4)]


def test_split_groups_disjoint_and_reproducible():
    groups = ["a", "b", "c", "d", "a", None, "e", "f"]

    tune = split_groups(groups, seed=0)

    assert tune == split_groups(groups, seed=0)
    assert len(tune) == 3  # половина от 6 авторов
    assert tune <= {"a", "b", "c", "d", "e", "f"}


def test_pick_best_known_answer():
    rows = [
        {"frr": 0.2, "false_alarms": 1, "awake_fraction": 0.1, "threshold": 0.1},
        {"frr": 0.4, "false_alarms": 0, "awake_fraction": 0.3, "threshold": 0.2},
        {"frr": 0.4, "false_alarms": 0, "awake_fraction": 0.1, "threshold": 0.3},
        {"frr": 0.4, "false_alarms": 0, "awake_fraction": 0.1, "threshold": 0.5},
        {"frr": 0.6, "false_alarms": 0, "awake_fraction": 0.0, "threshold": 0.9},
    ]

    # лучший FRR без ложных тревог; затем меньше пробуждения; затем выше порог
    assert pick_best(rows) == rows[3]
    assert pick_best(rows[:1]) is None
