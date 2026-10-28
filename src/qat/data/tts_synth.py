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


def piper_file_name(text: str) -> str:
    """Имя файла, которое Piper даёт строке в режиме --output-dir-naming text."""
    from pathvalidate import sanitize_filename  # зависимость piper-tts (только dev)

    return f"{sanitize_filename(text.strip())}.wav"


def run_piper(job: SynthJob, texts: Sequence[str], voices_dir: Path, out_dir: Path) -> dict:
    """Озвучить тексты одним запуском Piper. Возвращает {текст: путь к WAV 16 кГц}."""
    names = [piper_file_name(t) for t in texts]
    if len(set(names)) != len(names):
        raise ValueError("тексты дают одинаковые имена файлов Piper")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.txt").write_text("\n".join(t.strip() for t in texts) + "\n")
        model = voices_dir / f"{job.voice}.onnx"
        subprocess.run(
            [sys.executable, "-m", "piper", "-m", str(model), "-c", f"{model}.json",
             "-i", str(tmp / "in.txt"), "-d", str(tmp / "wav"), "--output-dir-naming", "text",
             "-s", str(job.speaker_id), "--length-scale", str(job.length_scale),
             "--noise-scale", str(job.noise_scale), "--noise-w-scale", str(job.noise_w_scale)],
            check=True, capture_output=True,
        )
        result = {}
        for text, name in zip(texts, names, strict=True):
            audio, sr = sf.read(str(tmp / "wav" / name), dtype="float32")
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            digest = hashlib.sha1(f"{job}|{text}".encode()).hexdigest()[:12]
            dst = out_dir / f"{digest}.wav"
            sf.write(str(dst), np.clip(audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
            result[text] = dst
    return result
