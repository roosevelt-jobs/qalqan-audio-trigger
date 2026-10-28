"""Детектор A: KWS-модель (DS-CNN) на кодовые фразы «на помощь» / «көмектесіңдер».

Признаки — qat.features (тот же код, что при обучении). Работает на CPU.
Оценка окна = максимум вероятностей классов фраз.
"""

from pathlib import Path

import numpy as np
import torch

from qat.features import N_FRAMES, WINDOW_SEC, log_mel
from qat.models.dscnn import DSCNN

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "kws" / "kws_dscnn_v1.pt"


class KwsDetector:
    name = "kws"
    version = "0.1.0"
    window_sec = WINDOW_SEC
    # ASSUMPTION: шаг 0,1 с — фраза (~1 с) несколько раз целиком попадает в окно 1,5 с.
    hop_sec = 0.1

    def __init__(self, model_path: str | Path = MODEL_PATH, threads: int = 1):
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"нет модели {model_path}: обучите (uv run python -m qat.train, README «S12»)"
            )
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        if ckpt["features"]["n_frames"] != N_FRAMES:
            raise ValueError("признаки модели не совпадают с qat.features")
        self.classes = ckpt["classes"]
        self.model = DSCNN(len(self.classes), **ckpt["model"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        torch.set_num_threads(threads)

    def class_probs(self, audio_window: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            x = torch.from_numpy(log_mel(audio_window))[None]
            return torch.softmax(self.model(x), 1)[0].numpy()

    def process(self, audio_window: np.ndarray) -> float:
        return float(self.class_probs(audio_window)[1:].max())

    def reset(self) -> None:
        pass
