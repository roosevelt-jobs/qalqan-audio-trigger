"""Синтез фраз голосами Piper для обучения модели A (TTS — только для обучения, CLAUDE.md 7).

Piper (piper-tts, GPL-3.0) вызывается как отдельная программа (`python -m piper`) и в код
продукта не импортируется: GPL касается программы, сгенерированный звук под неё не подпадает.
Один запуск = один голос, один диктор, одни параметры интонации, весь список фраз.
"""

import hashlib
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from qat.runtime.audio_io import SAMPLE_RATE


@dataclass(frozen=True)
class SynthJob:
    voice: str  # имя модели, например ru_RU-denis-medium
    speaker_id: int
    length_scale: float
    noise_scale: float
    noise_w_scale: float


def run_piper(job: SynthJob, texts: Sequence[str], voices_dir: Path, out_dir: Path) -> dict:
    """Озвучить тексты одним запуском Piper. Возвращает {текст: путь к WAV 16 кГц}.

    Файлы Piper именуются по времени (--output-dir-naming timestamp) и сопоставляются с
    текстами по порядку: Piper пишет их строго по строкам входа. Имена «по тексту» не
    годятся — на Linux длинная кириллическая фраза даёт имя длиннее 255 байт.
    """
    texts = [t.strip() for t in texts]
    if any(not t or "\n" in t for t in texts) or len(set(texts)) != len(texts):
        raise ValueError("тексты должны быть непустыми, однострочными и без повторов")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
        model = voices_dir / f"{job.voice}.onnx"
        proc = subprocess.run(
            [sys.executable, "-m", "piper", "-m", str(model), "-c", f"{model}.json",
             "-i", str(tmp / "in.txt"), "-d", str(tmp / "wav"),
             "--output-dir-naming", "timestamp",
             "-s", str(job.speaker_id), "--length-scale", str(job.length_scale),
             "--noise-scale", str(job.noise_scale), "--noise-w-scale", str(job.noise_w_scale)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Piper ({job}) завершился с кодом {proc.returncode}:\n"
                               f"{proc.stderr[-2000:]}")
        wavs = sorted((tmp / "wav").glob("*.wav"), key=lambda p: int(p.stem))
        if len(wavs) != len(texts):
            raise RuntimeError(f"Piper записал {len(wavs)} файлов на {len(texts)} фраз")
        result = {}
        for text, wav in zip(texts, wavs, strict=True):
            audio, sr = sf.read(str(wav), dtype="float32")
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            digest = hashlib.sha1(f"{job}|{text}".encode()).hexdigest()[:12]
            dst = out_dir / f"{digest}.wav"
            sf.write(str(dst), np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
            result[text] = dst
    return result
