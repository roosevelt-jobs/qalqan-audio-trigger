"""Сборка обучающего набора для модели A (KWS) по configs/data.yaml.

1. synth   — синтез Piper: позитивы, трудные негативы, обычная речь (теми же голосами);
             каждый синтезированный файл проверяется Vosk: в позитиве фраза есть, в негативах нет.
2. build   — живые негативы (Kazakh Speech Commands, FSD50K), разбиение train/val
             (отложенные люди TTS; KSCD — случайно, ID дикторов нет; FSD50K — по авторам),
             аугментация, манифест data/processed/kws/manifest.jsonl.

Тест модели A — только живые записи (CLAUDE.md 7), этот набор — для обучения и validation.
"""

import argparse
import csv
import hashlib
import json
import random
from itertools import product
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from qat.data.augment import augment
from qat.data.fsd50k import read_ground_truth
from qat.data.tts_synth import SynthJob, run_piper
from qat.detectors.asr_keywords import MODEL_PATHS, normalize
from qat.logging_setup import setup_logging
from qat.runtime.audio_io import SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "configs" / "data.yaml"
FLEURS_LANG = {"ru": "ru_ru", "kk": "kk_kz"}


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())["kws"]


def contains_phrase(text: str, phrases: list[str]) -> bool:
    tokens = [normalize(w) for w in text.split()]
    for p in phrases:
        words = [normalize(w) for w in p.split()]
        n = len(words)
        if any(tokens[i : i + n] == words for i in range(len(tokens) - n + 1)):
            return True
    return False


def fleurs_sentences(fleurs_dir: Path, lang: str, phrases: list[str], n: int, seed: int) -> list:
    """n случайных предложений FLEURS без кодовой фразы, ≤ 15 слов."""
    out = []
    for f in sorted(fleurs_dir.glob(f"{FLEURS_LANG[lang]}_*.tsv")):
        for line in f.read_text(encoding="utf-8").splitlines():
            cols = line.split("\t")
            if len(cols) >= 3 and 3 <= len(cols[2].split()) <= 15 and not contains_phrase(
                cols[2], phrases
            ):
                out.append(cols[2].strip())
    out = sorted(set(out))
    return random.Random(seed).sample(out, n)


class VoskCheck:
    """Распознавание синтезированного файла для проверки текста (полный словарь)."""

    def __init__(self, langs):
        import vosk

        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self.models = {lang: vosk.Model(str(MODEL_PATHS[lang])) for lang in langs}

    def text(self, lang: str, path: Path) -> str:
        audio, _ = sf.read(str(path), dtype="int16")
        rec = self._vosk.KaldiRecognizer(self.models[lang], SAMPLE_RATE)
        rec.AcceptWaveform(audio.tobytes())
        return json.loads(rec.FinalResult()).get("text", "")


_WORKER_VOSK: VoskCheck | None = None


def _synth_job(args) -> list[dict]:
    """Один запуск Piper + проверка Vosk; выполняется в отдельном процессе."""
    global _WORKER_VOSK
    voice, spk, person, ls, ns, nw, texts, phrases, voices_dir, out = args
    if _WORKER_VOSK is None:
        _WORKER_VOSK = VoskCheck(["ru", "kk"])
    lang = voice["lang"]
    job = SynthJob(voice["name"], int(spk), ls, ns, nw)
    files = run_piper(job, [t for t, _ in texts], voices_dir, out / "tts" / person)
    rows = []
    for text, kind in texts:
        has = contains_phrase(_WORKER_VOSK.text(lang, files[text]), phrases)
        rows.append({
            "path": str(files[text].relative_to(out)), "kind": kind, "lang": lang,
            "source": "piper", "voice": voice["name"], "speaker": person,
            "length_scale": ls, "noise_scale": ns, "license": voice["license"],
            "research_only": bool(voice.get("research_only", False)),
            # брак синтеза: позитив без фразы или негатив с фразой
            "verified": has if kind == "positive" else not has,
        })
    print(f"{voice['name']} {person} ls={ls} ns={ns}: готово", flush=True)
    return rows


def synth(cfg: dict, out: Path, workers: int = 4) -> list[dict]:
    from concurrent.futures import ProcessPoolExecutor

    seed = cfg["seed"]
    langs = sorted({v["lang"] for v in cfg["voices"]})
    texts = {}
    for lang in langs:
        phrases = cfg["phrases"][lang]
        neg = fleurs_sentences(ROOT / cfg["negative_sentences"]["source"], lang, phrases,
                               cfg["negative_sentences"]["per_voice"], seed)
        texts[lang] = ([(t, "positive") for t in cfg["positive_texts"][lang]]
                       + [(t, "hard_negative") for t in cfg["hard_negative_texts"][lang]]
                       + [(t, "speech") for t in neg])
    pr = cfg["prosody"]
    jobs = [
        (voice, spk, person, ls, ns, pr["noise_w_scale"], texts[voice["lang"]],
         cfg["phrases"][voice["lang"]], ROOT / cfg["voices_dir"], out)
        for voice in cfg["voices"]
        for (spk, person), ls, ns in product(voice["speakers"].items(), pr["length_scale"],
                                              pr["noise_scale"])
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job_rows in pool.map(_synth_job, jobs):  # порядок результатов = порядок jobs
            rows += job_rows
    (out / "synth.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    return rows


def _split_by(key: str, frac: float, seed: int) -> str:
    h = int(hashlib.sha1(f"{seed}|{key}".encode()).hexdigest(), 16) % 10_000
    return "val" if h < frac * 10_000 else "train"


def build(cfg: dict, out: Path) -> list[dict]:
    seed = cfg["seed"]
    base = [json.loads(line) for line in (out / "synth.jsonl").read_text().splitlines()]
    base = [r for r in base if r["verified"]]
    for r in base:
        r["label"] = "positive" if r["kind"] == "positive" else "negative"
        r["split"] = "val" if r["speaker"] in cfg["val_speakers"] else "train"

    # Kazakh Speech Commands — живая казахская речь (ID дикторов нет: только обучение).
    kscd = sorted((ROOT / "data/kscd/extracted/data").glob("*/*.wav"))
    for f in kscd:
        base.append({"path": str(f.relative_to(ROOT)), "kind": "kscd", "lang": "kk",
                     "source": "kscd", "label": "negative", "license": "MIT",
                     "split": _split_by(f.stem, cfg["kscd"]["val_speaker_fraction"], seed)})

    # FSD50K: часть негативов — как клипы, остальные неречевые — как фоновый шум.
    fs = cfg["fsd50k"]
    man = [json.loads(line) for line in (ROOT / fs["manifest"]).read_text().splitlines()]
    negs = [r for r in man if r["label"] == "negative"]
    random.Random(seed).shuffle(negs)
    gt = read_ground_truth(ROOT / "data/fsd50k/extracted/FSD50K.ground_truth/eval.csv")
    as_clips, noise_pool = negs[: fs["negatives_as_clips"]], negs[fs["negatives_as_clips"]:]
    for r in as_clips:
        base.append({"path": str(Path(fs["clips_dir"]) / r["path"]), "kind": "fsd50k",
                     "lang": None, "source": "fsd50k", "label": "negative",
                     "split": _split_by(r["group"], fs["val_group_fraction"], seed)})
    noise_files = [ROOT / fs["clips_dir"] / r["path"] for r in noise_pool
                   if "Human_voice" not in gt[Path(r["path"]).stem]][:300]
    noises = [sf.read(str(f), dtype="float32")[0] for f in noise_files]

    rows, aug_dir = [], out / "aug"
    aug_dir.mkdir(parents=True, exist_ok=True)
    for r in base:
        src = (out / r["path"]) if r["source"] == "piper" else ROOT / r["path"]
        rows.append({**r, "path": str(src.relative_to(ROOT)), "augment": None})
        copies = cfg["augment"]["copies_positive" if r["label"] == "positive"
                                else "copies_negative"]
        audio = None
        for k in range(copies):
            if audio is None:
                audio, _ = sf.read(str(src), dtype="float32")
            digest = hashlib.sha1(f"{seed}|{r['path']}|{k}".encode()).hexdigest()[:16]
            rng = np.random.default_rng(int(digest, 16) % 2**32)
            y, params = augment(audio, noises, rng)
            dst = aug_dir / f"{digest}.wav"
            sf.write(str(dst), y, SAMPLE_RATE, subtype="PCM_16")
            rows.append({**r, "path": str(dst.relative_to(ROOT)), "augment": params})
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    (out / "noise_files.txt").write_text("\n".join(str(f.relative_to(ROOT)) for f in noise_files))
    return rows


def summary(rows: list[dict]) -> str:
    table: dict = {}
    for r in rows:
        key = (r["split"], r["label"], r["kind"], r.get("lang") or "-")
        table[key] = table.get(key, 0) + 1
    lines = [f"{s:5} {lab:8} {k:13} {lang:3} {n}" for (s, lab, k, lang), n in sorted(table.items())]
    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Набор для модели A (KWS)")
    parser.add_argument("step", choices=["synth", "build", "summary"])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config()
    out = ROOT / cfg["out_dir"]
    if args.step == "synth":
        rows = synth(cfg, out)
        bad = [r for r in rows if not r["verified"]]
        print(f"синтезировано {len(rows)}, брак (Vosk) {len(bad)}")
        with (out / "synth_rejected.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["voice", "speaker", "kind", "length_scale", "noise_scale", "path"])
            for r in bad:
                w.writerow([r["voice"], r["speaker"], r["kind"], r["length_scale"],
                            r["noise_scale"], r["path"]])
    elif args.step == "build":
        print(summary(build(cfg, out)))
    else:
        rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
        print(summary(rows))


if __name__ == "__main__":
    _main()
