"""Пробуждение: порог громкости + Silero VAD. Решает, будить ли детектор на блоке.

Логика по громкости блока (dBFS):
- ниже energy_threshold_dbfs — тихо, VAD не вызывается;
- выше loud_threshold_dbfs — громкий звук (крик, удар), активно без VAD;
- между ними — решает VAD.
После активности состояние держится ещё hangover_sec.
"""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from qat.runtime.audio_io import SAMPLE_RATE

MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "vad" / "silero_vad.onnx"
# silero_vad.onnx из пакета silero-vad 6.2.3 (MIT), см. models/vad/LICENSE
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

VAD_CHUNK = 512  # модель принимает ровно 512 сэмплов при 16 кГц
VAD_CONTEXT = 64  # хвост предыдущего куска, который модель ждёт перед новым


class SileroVad:
    """Обёртка над ONNX-моделью Silero VAD по образцу OnnxWrapper из silero-vad 6.2.3."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        model_path = Path(model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != MODEL_SHA256:
            raise ValueError(f"{model_path}: SHA-256 {digest} не совпадает с {MODEL_SHA256}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        # Только CPU: офлайн и одинаково на любой машине.
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(VAD_CONTEXT, dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        """Вероятность речи 0..1 для куска из 512 сэмплов."""
        if chunk.shape != (VAD_CHUNK,):
            raise ValueError(f"нужно {VAD_CHUNK} сэмплов, получено {chunk.shape}")
        x = np.concatenate([self._context, chunk.astype(np.float32)])[None, :]
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[0, -VAD_CONTEXT:]
        return float(out[0, 0])


def block_dbfs(block: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-10))


class Wake:
    # ASSUMPTION: пороги -50 / -20 dBFS и hangover 1 с — стартовые, подбираются на живых записях.

    def __init__(
        self,
        energy_threshold_dbfs: float = -50.0,
        loud_threshold_dbfs: float = -20.0,
        vad_threshold: float = 0.5,
        hangover_sec: float = 1.0,
        vad: SileroVad | None = None,
        sample_rate: int = SAMPLE_RATE,
    ):
        self.energy_threshold_dbfs = energy_threshold_dbfs
        self.loud_threshold_dbfs = loud_threshold_dbfs
        self.vad_threshold = vad_threshold
        self.hangover_samples = round(hangover_sec * sample_rate)
        self.vad = vad if vad is not None else SileroVad()
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self._pending = np.zeros(0, dtype=np.float32)
        self._last_prob = 0.0
        self._hangover_left = 0

    def process(self, block: np.ndarray) -> bool:
        """True — будить детектор на этом блоке."""
        level = block_dbfs(block)
        if level >= self.loud_threshold_dbfs:
            active = True
        elif level < self.energy_threshold_dbfs:
            # ASSUMPTION: в тишине VAD не получает данных, его состояние сбрасывается.
            if len(self._pending) or self._last_prob:
                self.vad.reset()
                self._pending = np.zeros(0, dtype=np.float32)
                self._last_prob = 0.0
            active = False
        else:
            active = self._run_vad(block)

        if active:
            self._hangover_left = self.hangover_samples
            return True
        if self._hangover_left > 0:
            self._hangover_left -= len(block)
            return True
        return False

    def _run_vad(self, block: np.ndarray) -> bool:
        """Перекладывает блоки в куски по 512 и прогоняет VAD."""
        self._pending = np.concatenate([self._pending, block])
        probs = []
        while len(self._pending) >= VAD_CHUNK:
            probs.append(self.vad(self._pending[:VAD_CHUNK]))
            self._pending = self._pending[VAD_CHUNK:]
        if probs:
            self._last_prob = max(probs)
        return self._last_prob >= self.vad_threshold
