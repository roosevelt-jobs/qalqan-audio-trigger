"""FSD50K eval — тестовый набор для детектора B (крик и возглас).

Первоисточник: Zenodo, DOI 10.5281/zenodo.4060432 (версия 1.0). Проверка по CLAUDE.md 7.2 —
docs/data_card.md. Оригинальные файлы не меняются, перевод в 16 кГц — в data/processed/.

Шаги CLI:
  verify    — сверить md5 скачанных архивов с Zenodo
  extract   — склеить архив из частей, проверить содержимое (только .wav), распаковать
  convert   — перевести клипы в 16 кГц моно
  manifest  — манифест JSONL для стенда S8
"""

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from qat.logging_setup import setup_logging
from qat.runtime.audio_io import SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "fsd50k" / "raw"
EXTRACTED_DIR = ROOT / "data" / "fsd50k" / "extracted"
PROCESSED_DIR = ROOT / "data" / "processed" / "fsd50k_eval"

POSITIVE_LABELS = ("Screaming", "Shout", "Yell")
# ASSUMPTION: спорные для «крика» классы не считаются ни позитивом, ни негативом:
# плач, болельщики и толпа часто содержат выкрики.
EXCLUDED_LABELS = ("Crying_and_sobbing", "Cheering", "Crowd")


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_md5(raw_dir: Path, expected_file: Path) -> list[str]:
    """Список несовпадений (пустой — всё совпало). Формат файла: '<md5>  <имя>'."""
    problems = []
    for line in expected_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = raw_dir / name
        if not path.is_file():
            problems.append(f"{name}: нет файла")
        elif (got := md5_file(path)) != expected:
            problems.append(f"{name}: md5 {got}, ожидалось {expected}")
    return problems


def unexpected_members(names: Iterable[str], allowed_suffixes: Sequence[str]) -> list[str]:
    """Файлы архива с недопустимым расширением или путём за пределы папки."""
    bad = []
    for name in names:
        if name.endswith("/"):
            continue
        if name.startswith("/") or ".." in Path(name).parts:
            bad.append(name)
        elif Path(name).suffix.lower() not in allowed_suffixes:
            bad.append(name)
    return bad


def extract_eval_audio(raw_dir: Path, out_dir: Path) -> Path:
    """Склеить FSD50K.eval_audio.z01 + .zip, проверить содержимое, распаковать."""
    merged = raw_dir / "FSD50K.eval_audio.full.zip"
    if not merged.is_file():
        subprocess.run(
            ["zip", "-q", "-s", "0", str(raw_dir / "FSD50K.eval_audio.zip"), "--out", str(merged)],
            check=True,
        )
    with zipfile.ZipFile(merged) as z:
        bad = unexpected_members(z.namelist(), (".wav",))
        if bad:
            raise ValueError(f"в архиве неожиданные файлы, распаковка остановлена: {bad[:10]}")
        z.extractall(out_dir)
    return out_dir / "FSD50K.eval_audio"


def to_16k_mono(src: Path, dst: Path) -> float:
    """Перевод клипа в 16 кГц моно 16 бит. Возвращает длительность в секундах."""
    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), np.clip(mono, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")
    return len(mono) / SAMPLE_RATE


def read_ground_truth(csv_path: Path) -> dict[str, set[str]]:
    with csv_path.open(newline="") as f:
        return {row["fname"]: set(row["labels"].split(",")) for row in csv.DictReader(f)}


def build_manifest(
    ground_truth: dict[str, set[str]],
    positive_labels: Sequence[str] = POSITIVE_LABELS,
    excluded_labels: Sequence[str] = EXCLUDED_LABELS,
    uploaders: dict[str, str] | None = None,
) -> list[dict]:
    """Строки манифеста стенда S8. Пути относительно папки с клипами 16 кГц.

    condition позитива — самая узкая из его меток (Yell и Screaming уже Shout, метки
    распространены вверх по онтологии). group — автор клипа на Freesound, если передан.
    """
    rows = []
    for fname, labels in sorted(ground_truth.items(), key=lambda kv: int(kv[0])):
        if labels & set(positive_labels):
            # Разметка на весь клип: положение крика неизвестно, target не задаётся.
            primary = next(c for c in ("Screaming", "Yell", "Shout", *positive_labels)
                           if c in labels)
            row = {"path": f"{fname}.wav", "label": "positive", "condition": primary}
        elif labels & set(excluded_labels):
            continue
        else:
            row = {"path": f"{fname}.wav", "label": "negative"}
        if uploaders is not None:
            row["group"] = uploaders[fname]
        rows.append(row)
    return rows


def sample_negatives(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Все позитивы + n случайных негативов (воспроизводимо по seed), порядок сохранён."""
    neg_idx = [i for i, r in enumerate(rows) if r["label"] == "negative"]
    if n >= len(neg_idx):
        return list(rows)
    keep = set(np.random.default_rng(seed).choice(neg_idx, size=n, replace=False).tolist())
    return [r for i, r in enumerate(rows) if r["label"] == "positive" or i in keep]


def _main() -> None:
    parser = argparse.ArgumentParser(description="FSD50K eval для стенда S8")
    parser.add_argument("step", choices=["verify", "extract", "convert", "manifest"])
    parser.add_argument("--negatives", type=int, help="manifest: взять столько случайных негативов")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", help="convert: переводить только клипы из этого манифеста")
    args = parser.parse_args()
    setup_logging()

    if args.step == "verify":
        problems = verify_md5(RAW_DIR, RAW_DIR / "expected.md5")
        print("\n".join(problems) or "md5 всех файлов совпадают с Zenodo")
        raise SystemExit(1 if problems else 0)
    if args.step == "extract":
        print("распаковано в", extract_eval_audio(RAW_DIR, EXTRACTED_DIR))
    if args.step == "convert":
        src_dir = EXTRACTED_DIR / "FSD50K.eval_audio"
        files = sorted(src_dir.glob("*.wav"))
        if args.only:
            wanted = {json.loads(line)["path"] for line in Path(args.only).read_text().splitlines()}
            files = [f for f in files if f.name in wanted]
        total = 0.0
        for i, src in enumerate(files, 1):
            dst = PROCESSED_DIR / src.name
            total += sf.info(str(dst)).duration if dst.is_file() else to_16k_mono(src, dst)
            if i % 1000 == 0:
                print(f"{i}/{len(files)}")
        print(f"переведено {len(files)} клипов, {total / 3600:.1f} ч, в {PROCESSED_DIR}")
    if args.step == "manifest":
        gt = read_ground_truth(EXTRACTED_DIR / "FSD50K.ground_truth" / "eval.csv")
        info_path = EXTRACTED_DIR / "FSD50K.metadata" / "eval_clips_info_FSD50K.json"
        uploaders = {k: v["uploader"] for k, v in json.loads(info_path.read_text()).items()}
        rows = build_manifest(gt, uploaders=uploaders)
        name = "manifest_sed.jsonl"
        if args.negatives is not None:
            rows = sample_negatives(rows, args.negatives, args.seed)
            name = f"manifest_sed_neg{args.negatives}_seed{args.seed}.jsonl"
        out = PROCESSED_DIR / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        pos = sum(r["label"] == "positive" for r in rows)
        print(f"{out}: позитивов {pos}, негативов {len(rows) - pos}")


if __name__ == "__main__":
    _main()
