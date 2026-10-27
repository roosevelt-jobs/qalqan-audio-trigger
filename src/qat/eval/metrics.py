"""Метрики оценки: FRR, FA/h, DET-кривая. Работают на сохранённых потоках оценок, без аудио.

Порог перебирается применением постобработки (S5) к одним и тем же оценкам,
поэтому аудио прогоняется через детектор один раз.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from qat.runtime.audio_io import SAMPLE_RATE
from qat.runtime.pipeline import Score
from qat.runtime.postproc import PostProcessor, Trigger

# ASSUMPTION: тревога засчитывается как попадание до конца цели + 1 с (задержка распознавания).
HIT_TOLERANCE_AFTER_SEC = 1.0


@dataclass(frozen=True)
class ScoredFile:
    scores: list[Score]
    duration_sec: float
    condition: str = "unspecified"
    target: tuple[float, float] | None = None  # (начало, конец) цели в секундах, для позитивов


@dataclass(frozen=True)
class DetPoint:
    threshold: float
    frr: float
    fa_per_hour: float
    misses: int
    positives: int
    false_alarms: int
    negative_hours: float
    frr_by_condition: dict[str, float] = field(default_factory=dict)


def frr(misses: int, positives: int) -> float:
    if positives <= 0:
        raise ValueError("нет позитивных примеров")
    return misses / positives


def fa_per_hour(false_alarms: int, duration_sec: float) -> float:
    if duration_sec <= 0:
        raise ValueError("длительность негативного потока должна быть > 0")
    return false_alarms / (duration_sec / 3600.0)


def find_triggers(
    scores: Sequence[Score], threshold: float, hop_samples: int, **postproc_kwargs
) -> list[Trigger]:
    pp = PostProcessor(hop_samples, threshold=threshold, **postproc_kwargs)
    return [t for s in scores if (t := pp.process(s)) is not None]


def is_hit(
    triggers: Sequence[Trigger],
    target: tuple[float, float] | None,
    sample_rate: int = SAMPLE_RATE,
    tolerance_after_sec: float = HIT_TOLERANCE_AFTER_SEC,
) -> bool:
    if target is None:
        return len(triggers) > 0
    start, end = target
    return any(start <= t.sample / sample_rate <= end + tolerance_after_sec for t in triggers)


def det_curve(
    positives: Sequence[ScoredFile],
    negatives: Sequence[ScoredFile],
    thresholds: Sequence[float],
    hop_samples: int,
    sample_rate: int = SAMPLE_RATE,
    tolerance_after_sec: float = HIT_TOLERANCE_AFTER_SEC,
    **postproc_kwargs,
) -> list[DetPoint]:
    neg_sec = sum(f.duration_sec for f in negatives)
    points = []
    for thr in sorted(thresholds):
        by_cond: dict[str, list[int]] = {}  # условие -> [промахи, всего]
        for f in positives:
            trig = find_triggers(f.scores, thr, hop_samples, **postproc_kwargs)
            hit = is_hit(trig, f.target, sample_rate, tolerance_after_sec)
            counts = by_cond.setdefault(f.condition, [0, 0])
            counts[0] += 0 if hit else 1
            counts[1] += 1
        misses = sum(c[0] for c in by_cond.values())
        fa = sum(
            len(find_triggers(f.scores, thr, hop_samples, **postproc_kwargs)) for f in negatives
        )
        points.append(
            DetPoint(
                threshold=thr,
                frr=frr(misses, len(positives)),
                fa_per_hour=fa_per_hour(fa, neg_sec),
                misses=misses,
                positives=len(positives),
                false_alarms=fa,
                negative_hours=neg_sec / 3600.0,
                frr_by_condition={c: frr(m, n) for c, (m, n) in sorted(by_cond.items())},
            )
        )
    return points


def threshold_at_fa(det: Sequence[DetPoint], max_fa_per_hour: float = 0.1) -> DetPoint | None:
    """Самый низкий порог с FA/h <= max_fa_per_hour (у него лучший FRR). None, если такого нет."""
    ok = [p for p in det if p.fa_per_hour <= max_fa_per_hour]
    return min(ok, key=lambda p: p.threshold) if ok else None
