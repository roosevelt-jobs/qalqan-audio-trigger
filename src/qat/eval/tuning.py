"""Подбор пробуждения и постобработки для детектора без состояния (B, sed_panns).

У такого детектора оценка окна не зависит от того, будили его раньше или нет. Поэтому:
1. детектор один раз прогоняется по ВСЕМ окнам (без пробуждения), оценки кэшируются;
2. для каждой настройки пробуждения считается, на каких блоках оно активно;
3. маска накладывается на кэш — ровно те окна, что оценил бы конвейер (run_scores);
4. постобработка и метрики перебираются на этих оценках.

Защита от подгонки: клипы делятся по авторам (group) на половину для подбора и половину для
итога; один автор не попадает в обе.
"""

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qat.detectors.base import Detector
from qat.detectors.registry import make_detector
from qat.eval.bench import Item, _AlwaysAwake, load_manifest, score_file
from qat.eval.metrics import ScoredFile, det_curve
from qat.logging_setup import setup_logging
from qat.runtime.audio_io import BLOCK_SIZE, SAMPLE_RATE, WavReplaySource
from qat.runtime.pipeline import Score
from qat.runtime.wake import SileroVad, Wake

THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
LOUD_THRESHOLDS = (-20.0, -25.0, -30.0, -35.0)  # None в переборе = без пробуждения
POSTPROC_GRID = [
    {"smooth_windows": s, "consecutive": c} for s in (1, 3) for c in (1, 2)
]


def load_manifest_groups(path: Path) -> list[str | None]:
    return [json.loads(line).get("group") for line in path.read_text().splitlines() if line]


def score_all_windows(items: Sequence[Item], detector: Detector, cache: Path) -> dict[str, dict]:
    """Оценки детектора по всем окнам каждого клипа (без пробуждения), с кэшем на диске."""
    data = json.loads(cache.read_text()) if cache.is_file() else {}
    for i, item in enumerate(items, 1):
        key = item.path.name
        if key in data:
            continue
        f = score_file(item, detector, vad=None)
        data[key] = {"duration": f.duration_sec, "scores": [[s.sample, s.score] for s in f.scores]}
        if i % 200 == 0:
            cache.write_text(json.dumps(data))
            print(f"оценено {i}/{len(items)}", flush=True)
    cache.write_text(json.dumps(data))
    return data


def wake_activity(path: Path, wake: Wake | _AlwaysAwake) -> np.ndarray:
    """Решение пробуждения по каждому блоку источника (как в run_scores)."""
    return np.array([wake.process(b) for b in WavReplaySource(path)], dtype=bool)


def mask_scores(
    scores: Sequence[Score], active: np.ndarray, block: int = BLOCK_SIZE
) -> list[Score]:
    """Окна, которые оценил бы конвейер: конец окна E проверяется на блоке, где записан
    сэмпл E (блок с номером ceil(E / block) - 1), и только если wake там активен."""
    out = []
    for s in scores:
        k = math.ceil(s.sample / block) - 1
        if 0 <= k < len(active) and active[k]:
            out.append(s)
    return out


def split_groups(groups: Sequence[str | None], seed: int = 0) -> set[str]:
    """Половина авторов (воспроизводимо по seed) — для подбора; остальные — для итога."""
    uniq = sorted({g for g in groups if g is not None})
    order = np.random.default_rng(seed).permutation(len(uniq))
    return {uniq[i] for i in order[: len(uniq) // 2]}


@dataclass(frozen=True)
class Choice:
    loud_dbfs: float | None
    smooth_windows: int
    consecutive: int
    threshold: float
    frr: float
    false_alarms: int
    awake_fraction: float


def pick_best(rows: Sequence[dict]) -> dict | None:
    """Максимум пойманных при 0 ложных тревог; при равенстве — меньше времени в пробуждении,
    затем выше порог."""
    ok = [r for r in rows if r["false_alarms"] == 0]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["frr"], r["awake_fraction"], -r["threshold"]))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Подбор пробуждения и постобработки для B")
    parser.add_argument("step", choices=["score", "tune"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detector", default="sed_panns")
    parser.add_argument("--out", default="reports/tuning_sed_panns")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()

    manifest = Path(args.manifest)
    items = load_manifest(manifest)
    cache = manifest.with_name(f"scores_{args.detector}_all_windows.json")
    if args.step == "score":
        score_all_windows(items, make_detector(args.detector), cache)
        print("кэш оценок:", cache)
        return
    _tune(items, load_manifest_groups(manifest), cache, Path(args.out), args.seed,
          args.detector)


def _tune(items, groups, cache: Path, out: Path, seed: int, detector_name: str) -> None:
    data = json.loads(cache.read_text())
    tune_groups = split_groups(groups, seed)
    hop = round(make_detector_hop(detector_name) * SAMPLE_RATE)
    vad = SileroVad()

    configs: list[float | None] = [*LOUD_THRESHOLDS, None]
    halves = {"tune": [], "test": []}
    for item, g in zip(items, groups, strict=True):
        halves["tune" if g in tune_groups else "test"].append(item)

    def scored(item: Item, loud: float | None) -> tuple[ScoredFile, int, int]:
        d = data[item.path.name]
        all_scores = [Score(int(a), float(b)) for a, b in d["scores"]]
        wake = _AlwaysAwake() if loud is None else Wake(loud_threshold_dbfs=loud, vad=vad)
        kept = mask_scores(all_scores, wake_activity(item.path, wake))
        sf_ = ScoredFile(kept, d["duration"], item.condition, item.target)
        return sf_, len(kept), len(all_scores)

    grid_rows, summary = [], {"tune_groups": len(tune_groups), "halves": {}}
    per_half_cache: dict[tuple[str, float | None], tuple] = {}
    for half, half_items in halves.items():
        pos = [i for i in half_items if i.label == "positive"]
        neg = [i for i in half_items if i.label == "negative"]
        summary["halves"][half] = {"positives": len(pos), "negatives": len(neg)}
        for loud in configs:
            ps = [scored(i, loud)[0] for i in pos]
            ns = [scored(i, loud) for i in neg]
            awake = sum(n for _, n, _ in ns) / max(1, sum(t for _, _, t in ns))
            per_half_cache[(half, loud)] = (ps, [s for s, _, _ in ns], awake)
            print(f"{half}: пробуждение {loud}: активно {awake:.1%} окон негативов", flush=True)

    def evaluate(half: str, loud, pp: dict, thresholds) -> list[dict]:
        ps, ns, awake = per_half_cache[(half, loud)]
        rows = []
        for p in det_curve(ps, ns, thresholds, hop, SAMPLE_RATE, **pp):
            rows.append({
                "half": half, "loud_dbfs": loud, **pp, "threshold": p.threshold,
                "frr": p.frr, "false_alarms": p.false_alarms, "fa_per_hour": p.fa_per_hour,
                "negative_hours": p.negative_hours, "awake_fraction": awake,
                **{f"frr_{c}": v for c, v in p.frr_by_condition.items()},
            })
        return rows

    for loud in configs:
        for pp in POSTPROC_GRID:
            grid_rows += evaluate("tune", loud, pp, THRESHOLDS)

    baseline_rows = [r for r in grid_rows if r["loud_dbfs"] == -20.0
                     and r["smooth_windows"] == 3 and r["consecutive"] == 2]
    choices = {"текущая": pick_best(baseline_rows), "лучшая": pick_best(grid_rows)}
    for name, ch in choices.items():
        if ch is None:
            summary[name] = None
            continue
        pp = {"smooth_windows": ch["smooth_windows"], "consecutive": ch["consecutive"]}
        (test,) = evaluate("test", ch["loud_dbfs"], pp, [ch["threshold"]])
        summary[name] = {"tune": ch, "test": test}

    out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in grid_rows for k in r}, key=lambda k: list(grid_rows[0]).index(k)
                  if k in grid_rows[0] else 99)
    with (out / "grid_tune.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(grid_rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_detector_hop(name: str) -> float:
    return make_detector(name).hop_sec


if __name__ == "__main__":
    _main()
