"""Обучение модели A (DS-CNN, 3 класса: другое / «на помощь» / «көмектесіңдер»).

    uv run python -m qat.train

Признаки — только qat.features (один код для обучения и детектора). Они считаются один раз
и кешируются рядом с моделью. Устройство: cuda -> mps -> cpu. Validation — отложенные люди
(configs/data.yaml); настоящий тест — только живые записи (CLAUDE.md, раздел 7).
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from qat.features import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MELS,
    WIN_LENGTH,
    WINDOW_SAMPLES,
    fit_window,
    log_mel,
)
from qat.logging_setup import setup_logging
from qat.models.dscnn import DSCNN
from qat.runtime.audio_io import SAMPLE_RATE

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "train.yaml"
FEATURE_PARAMS = {"sample_rate": SAMPLE_RATE, "n_mels": N_MELS, "win_length": WIN_LENGTH,
                  "hop_length": HOP_LENGTH, "n_fft": N_FFT, "fmin": FMIN, "fmax": FMAX,
                  "window_samples": WINDOW_SAMPLES, "n_frames": N_FRAMES}


def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())["kws"]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def label_of(row: dict, cfg: dict) -> int:
    return cfg["class_of_lang"][row["lang"]] if row["label"] == "positive" else 0


def crop(
    audio: np.ndarray, positive: bool, rng: np.random.Generator, max_offset: int
) -> np.ndarray:
    """Окно модели: позитив — с начала клипа, сдвинутый на случайные 0..max_offset сэмплов
    тишины (фраза в синтезе начинается сразу); негатив — случайный кусок клипа."""
    if positive:
        return fit_window(audio, -int(rng.integers(0, max_offset + 1)))
    if len(audio) > WINDOW_SAMPLES:
        return fit_window(audio, int(rng.integers(0, len(audio) - WINDOW_SAMPLES + 1)))
    return fit_window(audio, -int(rng.integers(0, WINDOW_SAMPLES - len(audio) + 1)))


def build_features(rows: list[dict], cfg: dict, root: Path = ROOT):
    """X (N, кадры, 40) float16, y, is_val. Кроп детерминирован по seed и пути файла."""
    max_offset = round(cfg["positive_max_offset_sec"] * SAMPLE_RATE)
    X = np.zeros((len(rows), N_FRAMES, N_MELS), dtype=np.float16)
    y = np.zeros(len(rows), dtype=np.int64)
    is_val = np.zeros(len(rows), dtype=bool)
    for i, r in enumerate(rows):
        audio, sr = sf.read(str(root / r["path"]), dtype="float32", always_2d=True)
        assert sr == SAMPLE_RATE, r["path"]
        digest = hashlib.sha1(f"{cfg['seed']}|{r['path']}".encode()).hexdigest()
        rng = np.random.default_rng(int(digest[:8], 16))
        y[i] = label_of(r, cfg)
        X[i] = log_mel(crop(audio.mean(axis=1), y[i] > 0, rng, max_offset))
        is_val[i] = r["split"] == "val"
        if (i + 1) % 2000 == 0:
            print(f"признаки: {i + 1}/{len(rows)}", flush=True)
    return X, y, is_val


def spec_augment(x: torch.Tensor, time_mask: int, freq_mask: int) -> torch.Tensor:
    """По одной маске времени и частоты на пример (обнуление после bn_in-нормировки — 0)."""
    b, t, f = x.shape
    x = x.clone()
    tw = torch.randint(0, time_mask + 1, (b,))
    t0 = (torch.rand(b) * (t - tw)).long()
    fw = torch.randint(0, freq_mask + 1, (b,))
    f0 = (torch.rand(b) * (f - fw)).long()
    for i in range(b):
        x[i, t0[i] : t0[i] + tw[i], :] = x[i].mean()
        x[i, :, f0[i] : f0[i] + fw[i]] = x[i].mean()
    return x


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, threshold: float,
             device: torch.device) -> dict:
    model.eval()
    probs = torch.cat([torch.softmax(model(X[i : i + 1024].to(device).float()), 1).cpu()
                       for i in range(0, len(X), 1024)])
    pred = probs.argmax(1)
    score = probs[:, 1:].max(1).values
    out = {"n": int(len(y))}
    for c in range(probs.shape[1]):
        m = y == c
        out[f"recall_{c}"] = float((pred[m] == c).float().mean()) if m.any() else None
    neg = y == 0
    out["negative_fpr_at_threshold"] = float((score[neg] >= threshold).float().mean())
    pos = y > 0
    out["positive_recall_at_threshold"] = float((score[pos] >= threshold).float().mean())
    out["ru_kk_confusion"] = int(((pred == 2) & (y == 1)).sum() + ((pred == 1) & (y == 2)).sum())
    recalls = [v for k, v in out.items() if k.startswith("recall_") and v is not None]
    out["macro_recall"] = float(np.mean(recalls))
    return out


def train(cfg: dict, root: Path = ROOT, epochs: int | None = None) -> dict:
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_dir = root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = root / cfg["manifest"]
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
    key = hashlib.sha1((manifest.read_text() + json.dumps(FEATURE_PARAMS) + json.dumps(
        {k: cfg[k] for k in ("seed", "positive_max_offset_sec", "class_of_lang")})).encode()
    ).hexdigest()[:12]
    cache = out_dir / f"features_{key}.npz"
    if cache.is_file():
        d = np.load(cache)
        X, y, is_val = d["X"], d["y"], d["is_val"]
    else:
        X, y, is_val = build_features(rows, cfg, root)
        np.savez(cache, X=X, y=y, is_val=is_val)
    print(f"примеров {len(y)} (val {int(is_val.sum())}), по классам: "
          f"{np.bincount(y[~is_val], minlength=3).tolist()} train / "
          f"{np.bincount(y[is_val], minlength=3).tolist()} val", flush=True)

    device = pick_device()
    Xt, yt = torch.from_numpy(X[~is_val]), torch.from_numpy(y[~is_val])
    Xv, yv = torch.from_numpy(X[is_val]), torch.from_numpy(y[is_val])
    counts = np.bincount(y[~is_val], minlength=len(cfg["classes"])).astype(np.float64)
    # Балансировка: каждый класс попадает в батчи одинаково часто.
    weights = torch.from_numpy((1.0 / np.maximum(counts, 1))[y[~is_val]])
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg["batch_size"], generator=g,
                        sampler=WeightedRandomSampler(weights, len(yt), generator=g))

    model = DSCNN(len(cfg["classes"]), **cfg["model"]).to(device)
    n_epochs = epochs or cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=n_epochs *
                                                len(loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    best, history = None, []
    for epoch in range(1, n_epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = spec_augment(xb.float(), **cfg["specaugment"]).to(device)
            loss = loss_fn(model(xb), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(yb)
        val = evaluate(model, Xv, yv, cfg["score_threshold"], device)
        # Детектору важна уверенность выше порога: фразы — выше, негативы — ниже.
        val["detection_score"] = (val["positive_recall_at_threshold"]
                                  - val["negative_fpr_at_threshold"])
        history.append({"epoch": epoch, "train_loss": total / len(yt), **val})
        print(f"эпоха {epoch:2d}: loss {total / len(yt):.3f}, val: фраз выше порога "
              f"{val['positive_recall_at_threshold']:.3f}, негативов выше порога "
              f"{val['negative_fpr_at_threshold']:.4f}, ru {val['recall_1']}, "
              f"kk {val['recall_2']}", flush=True)
        # при равенстве — более поздняя эпоха (скорость обучения уже снижена)
        if best is None or val["detection_score"] >= best["detection_score"]:
            best = {"epoch": epoch, **val}
            ckpt = {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "classes": cfg["classes"], "model": cfg["model"], "features": FEATURE_PARAMS,
                "train_config": cfg, "manifest_sha1": hashlib.sha1(manifest.read_bytes())
                .hexdigest(), "val_metrics": best,
            }
            torch.save(ckpt, out_dir / f"{cfg['name']}.pt")
    report = {"device": str(device), "best": best, "history": history,
              "params": sum(p.numel() for p in model.parameters())}
    (out_dir / f"{cfg['name']}_report.json").write_text(json.dumps(report, indent=2))
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели A")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    setup_logging()
    report = train(load_config(), epochs=args.epochs)
    print("лучшая эпоха:", json.dumps(report["best"], ensure_ascii=False))


if __name__ == "__main__":
    _main()
