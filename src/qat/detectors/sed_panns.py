"""Детектор B: sound event detection на предобученной PANNs Cnn14_16k (AudioSet, 527 классов).

Оценка окна = максимум вероятностей по целевым классам (по умолчанию «крик и возглас»).
Веса и их происхождение: models/panns/README.md.
"""

import csv
import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from qat.detectors.panns_cnn14 import CLASSES_NUM, Cnn14_16k

PANNS_DIR = Path(__file__).resolve().parents[3] / "models" / "panns"
WEIGHTS_PATH = PANNS_DIR / "Cnn14_16k_mAP=0.438.pth"
# md5 файла совпадает с записью Zenodo: 362fc5ff18f1d6ad2f6d464b45893f2c
WEIGHTS_SHA256 = "e2ee543a27919542c2ea03eabaa70b24dcd4e6c8e05621de6b67a94e4c5058e6"
LABELS_PATH = PANNS_DIR / "class_labels_indices.csv"

SCREAM_SHOUT_CLASSES = (
    "Screaming",
    "Shout",
    "Yell",
    "Bellow",
    "Battle cry",
    "Children shouting",
)


def load_labels(path: str | Path = LABELS_PATH) -> list[str]:
    with Path(path).open(newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["index"]))
    labels = [r["display_name"] for r in rows]
    if len(labels) != CLASSES_NUM:
        raise ValueError(f"{path}: {len(labels)} классов, ожидалось {CLASSES_NUM}")
    return labels


def load_model(weights_path: str | Path = WEIGHTS_PATH) -> Cnn14_16k:
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"нет весов {weights_path}; как скачать — models/panns/README.md"
        )
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"{weights_path}: SHA-256 {digest} не совпадает с {WEIGHTS_SHA256}")
    # В чекпойнте кроме весов лежит состояние сэмплера обучения с массивами numpy.
    # Безопасная загрузка (weights_only) с разрешением ровно этих типов numpy;
    # _reconstruct записан в файле под старым (numpy 1.x) именем.
    numpy_globals = [
        (np._core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        np.dtypes.Int64DType,
    ]
    with torch.serialization.safe_globals(numpy_globals):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = Cnn14_16k()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


class SedPannsDetector:
    name = "sed_panns"
    version = "0.1.0"
    # ASSUMPTION: окно 1 с и шаг 0.25 с — компромисс между задержкой и нагрузкой на CPU.
    window_sec = 1.0
    hop_sec = 0.25

    def __init__(
        self,
        classes: Sequence[str] = SCREAM_SHOUT_CLASSES,
        weights_path: str | Path = WEIGHTS_PATH,
        model: Cnn14_16k | None = None,
        threads: int = 1,
    ):
        self.labels = load_labels()
        missing = [c for c in classes if c not in self.labels]
        if missing:
            raise ValueError(f"нет таких классов AudioSet: {missing}")
        self.classes = tuple(classes)
        self._idx = [self.labels.index(c) for c in classes]
        # Один поток: нагрузка честно сравнивается с целью «< 10% одного ядра».
        torch.set_num_threads(threads)
        self.model = model if model is not None else load_model(weights_path)
        self.model.eval()

    def class_probs(self, audio_window: np.ndarray) -> np.ndarray:
        """Вероятности всех 527 классов AudioSet для окна."""
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(audio_window, dtype=np.float32))[None, :]
            return self.model(x)[0].numpy()

    def process(self, audio_window: np.ndarray) -> float:
        # ASSUMPTION: оценка — максимум вероятностей по целевым классам.
        return float(self.class_probs(audio_window)[self._idx].max())

    def reset(self) -> None:
        pass
