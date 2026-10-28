"""Признаки, модель и обучение A на синтетических сигналах с известным ответом."""

import json

import numpy as np
import pytest
import soundfile as sf

from qat.detectors.base import Detector
from qat.detectors.kws import KwsDetector
from qat.features import N_FRAMES, N_MELS, WINDOW_SAMPLES, fit_window, log_mel
from qat.models.dscnn import DSCNN
from qat.train import crop, load_config, train

SR = 16000


def test_log_mel_shape_and_determinism():
    x = np.random.default_rng(0).uniform(-0.5, 0.5, WINDOW_SAMPLES).astype(np.float32)
    f = log_mel(x)
    assert f.shape == (N_FRAMES, N_MELS) == (151, 40)
    assert f.dtype == np.float32
    np.testing.assert_array_equal(f, log_mel(x))


def test_fit_window_pads_and_cuts():
    a = np.arange(1, 1001, dtype=np.float32)
    w = fit_window(a, -10)
    assert len(w) == WINDOW_SAMPLES
    assert (w[:10] == 0).all() and w[10] == 1 and w[1009] == 1000 and (w[1010:] == 0).all()
    long = np.ones(WINDOW_SAMPLES * 2, dtype=np.float32)
    assert (fit_window(long, 5) == 1).all()


def test_crop_positive_offset_range():
    a = np.ones(SR, dtype=np.float32)
    for seed in range(20):
        w = crop(a, True, np.random.default_rng(seed), max_offset=8000)
        lead = int(np.argmax(w > 0))
        assert len(w) == WINDOW_SAMPLES and 0 <= lead <= 8000


def test_model_under_size_limit():
    n = sum(p.numel() for p in DSCNN(3).parameters())
    assert n * 4 < 1_000_000  # < 1 МБ float32 (CLAUDE.md, раздел 6)


def make_toy(root, n=60):
    """«ru» — тон 600 Гц, «kk» — 1500 Гц, негативы — шум. Валидация — отдельные seed."""
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    rows = []
    for i in range(n):
        for kind in ("ru", "kk", "neg"):
            if kind == "neg":
                x = rng.normal(0, 0.1, SR)
            else:
                f = 600 if kind == "ru" else 1500
                x = 0.5 * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) + rng.normal(0, 0.02, SR)
            p = root / f"{kind}_{i}.wav"
            sf.write(str(p), x.astype(np.float32), SR)
            rows.append({"path": p.name, "label": "negative" if kind == "neg" else "positive",
                         "lang": None if kind == "neg" else kind,
                         "split": "val" if i % 5 == 0 else "train"})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


@pytest.fixture(scope="module")
def toy_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("toy")
    make_toy(root)
    cfg = {**load_config(), "manifest": "manifest.jsonl", "out_dir": "out", "name": "toy",
           "batch_size": 32, "lr": 0.01}
    report = train(cfg, root=root, epochs=8)
    return root / "out" / "toy.pt", report


def test_training_learns_toy_task(toy_model):
    path, report = toy_model
    assert path.is_file()
    assert report["best"]["macro_recall"] >= 0.9
    assert report["best"]["negative_fpr_at_threshold"] <= 0.1
    assert (path.parent / "toy_report.json").is_file()


def test_detector_uses_trained_model(toy_model):
    det = KwsDetector(toy_model[0])
    t = np.arange(SR) / SR
    # как в обучении: «фраза» ~1 с внутри окна 1,5 с
    tone = fit_window((0.5 * np.sin(2 * np.pi * 600 * t)).astype(np.float32), -2000)
    noise = np.random.default_rng(1).normal(0, 0.1, WINDOW_SAMPLES).astype(np.float32)

    assert isinstance(det, Detector) and det.name == "kws"
    assert det.process(tone) > 0.5
    assert det.process(noise) < 0.5
    assert det.class_probs(tone).argmax() == 1  # класс «ru»


def test_detector_requires_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="qat.train"):
        KwsDetector(tmp_path / "nope.pt")
