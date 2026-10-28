"""Стенд оценки любого Detector: манифест JSONL -> отчёт (DET-кривая, FRR по условиям, FA/h).

Каждый файл прогоняется через тот же конвейер, что и в живом режиме
(replay -> буфер -> пробуждение -> детектор), один раз; пороги перебираются на оценках.

Манифест: по строке JSON на файл, пути относительно манифеста:
  {"path": "pos/001.wav", "label": "positive", "condition": "noise",
   "target_start_sec": 1.2, "target_end_sec": 2.0}
  {"path": "neg/stream_01.wav", "label": "negative"}
"""

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from qat.detectors.base import Detector
from qat.detectors.registry import CONFIGS, DETECTORS, make_detector
from qat.eval.metrics import HIT_TOLERANCE_AFTER_SEC, ScoredFile, det_curve, threshold_at_fa
from qat.logging_setup import setup_logging
from qat.runtime.audio_io import SAMPLE_RATE, WavReplaySource
from qat.runtime.pipeline import run_scores
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import SileroVad, Wake

# ASSUMPTION: сетка порогов по умолчанию 0.05 … 0.95 с шагом 0.05.
DEFAULT_THRESHOLDS = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]
TARGET_FA_PER_HOUR = 0.1


@dataclass(frozen=True)
class Item:
    path: Path
    label: str  # "positive" | "negative"
    condition: str = "unspecified"
    target: tuple[float, float] | None = None


def load_manifest(path: str | Path) -> list[Item]:
    path = Path(path)
    items = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("label")
        if label not in ("positive", "negative"):
            raise ValueError(f"{path}:{n}: label должен быть positive или negative, а не {label!r}")
        target = None
        if "target_start_sec" in row or "target_end_sec" in row:
            target = (float(row["target_start_sec"]), float(row["target_end_sec"]))
        items.append(
            Item(
                path=path.parent / row["path"],
                label=label,
                condition=row.get("condition", "unspecified"),
                target=target,
            )
        )
    return items


class _AlwaysAwake:
    """Пробуждение выключено: детектор оценивает каждое окно."""

    def process(self, block: np.ndarray) -> bool:
        return True


def score_file(item: Item, detector: Detector, vad: SileroVad | None) -> ScoredFile:
    """Прогон файла через конвейер. vad=None — без пробуждения."""
    detector.reset()
    wake = Wake(vad=vad) if vad is not None else _AlwaysAwake()
    scores = list(run_scores(WavReplaySource(item.path), detector, wake, RingBuffer()))
    info = sf.info(str(item.path))
    return ScoredFile(scores, info.frames / info.samplerate, item.condition, item.target)


def evaluate(
    items: Sequence[Item],
    detector: Detector,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    use_wake: bool = True,
    tolerance_after_sec: float = HIT_TOLERANCE_AFTER_SEC,
) -> dict:
    positives = [i for i in items if i.label == "positive"]
    negatives = [i for i in items if i.label == "negative"]
    if not positives or not negatives:
        raise ValueError("нужны и позитивные, и негативные файлы")

    vad = SileroVad() if use_wake else None
    pos = [score_file(i, detector, vad) for i in positives]
    neg = [score_file(i, detector, vad) for i in negatives]
    hop = round(detector.hop_sec * SAMPLE_RATE)
    postproc = CONFIGS[detector.name].postproc if detector.name in CONFIGS else {}
    det = det_curve(pos, neg, thresholds, hop, SAMPLE_RATE, tolerance_after_sec, **postproc)
    op = threshold_at_fa(det, TARGET_FA_PER_HOUR)

    neg_hours = det[0].negative_hours
    warnings = []
    if neg_hours < 1 / TARGET_FA_PER_HOUR:
        warnings.append(
            f"негативного потока {neg_hours:.2f} ч: даже 0 тревог не подтверждают "
            f"FA/h <= {TARGET_FA_PER_HOUR} (нужно >= {1 / TARGET_FA_PER_HOUR:.0f} ч)"
        )
    if op is None:
        warnings.append(f"ни один порог не даёт FA/h <= {TARGET_FA_PER_HOUR}")

    return {
        "detector": detector.name,
        "detector_version": detector.version,
        "use_wake": use_wake,
        "postproc": postproc,
        "tolerance_after_sec": tolerance_after_sec,
        "positives": len(positives),
        "negatives": len(negatives),
        "negative_hours": neg_hours,
        "target_fa_per_hour": TARGET_FA_PER_HOUR,
        "operating_point": asdict(op) if op else None,
        "det": [asdict(p) for p in det],
        "warnings": warnings,
    }


def write_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (out / "det.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "frr", "fa_per_hour", "misses", "positives", "false_alarms"])
        for p in report["det"]:
            w.writerow([p["threshold"], p["frr"], p["fa_per_hour"], p["misses"],
                        p["positives"], p["false_alarms"]])


def _main() -> None:
    parser = argparse.ArgumentParser(description="Стенд оценки детектора: FRR, FA/h, DET")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--detector", choices=sorted(DETECTORS), required=True)
    parser.add_argument("--out", required=True, help="папка для report.json и det.csv")
    parser.add_argument("--no-wake", action="store_true", help="оценивать без пробуждения")
    args = parser.parse_args()
    setup_logging()

    report = evaluate(load_manifest(args.manifest), make_detector(args.detector),
                      use_wake=not args.no_wake)
    write_report(report, args.out)

    op = report["operating_point"]
    print(f"детектор {report['detector']} {report['detector_version']}, "
          f"позитивов {report['positives']}, негативный поток {report['negative_hours']:.2f} ч")
    if op:
        print(f"рабочий порог {op['threshold']}: FRR {op['frr']:.3f}, "
              f"FA/h {op['fa_per_hour']:.3f}, по условиям {op['frr_by_condition']}")
    for w in report["warnings"]:
        print("ВНИМАНИЕ:", w)
    print(f"отчёт: {Path(args.out) / 'report.json'}, DET: {Path(args.out) / 'det.csv'}")


if __name__ == "__main__":
    _main()
