import pytest

from qat.eval.metrics import (
    DetPoint,
    ScoredFile,
    det_curve,
    fa_per_hour,
    frr,
    is_hit,
    threshold_at_fa,
)
from qat.runtime.pipeline import Score
from qat.runtime.postproc import Trigger

SR = 16000
HOP = 1600  # 0.1 с


def run(start_sec, value, n=3):
    """n соседних окон с одинаковой оценкой, начиная с start_sec."""
    first = round(start_sec * SR)
    return [Score(first + i * HOP, value) for i in range(n)]


def test_frr_and_fa_per_hour():
    assert frr(3, 10) == 0.3
    assert fa_per_hour(2, 1800) == 4.0
    with pytest.raises(ValueError):
        frr(0, 0)
    with pytest.raises(ValueError):
        fa_per_hour(1, 0)


def test_is_hit_window():
    target = (5.0, 6.0)
    trig = lambda sec: [Trigger(round(sec * SR), 0.9, 0.8)]  # noqa: E731

    assert not is_hit(trig(4.0), target)   # до начала цели — промах
    assert is_hit(trig(5.5), target)
    assert is_hit(trig(6.9), target)       # в пределах +1 с после конца
    assert not is_hit(trig(7.5), target)
    assert not is_hit([], target)
    assert is_hit(trig(0.1), None)         # без разметки — любая тревога


def test_det_curve_known_answer():
    target = (0.0, 10.0)
    positives = [
        ScoredFile(run(1.0, 0.9), 10.0, "quiet", target),
        ScoredFile(run(1.0, 0.6), 10.0, "quiet", target),
        ScoredFile(run(1.0, 0.3), 10.0, "noise", target),
    ]
    # 1 час негатива, два отдельных эпизода: 0.7 и 0.4
    negatives = [ScoredFile(run(100.0, 0.7) + run(2000.0, 0.4), 3600.0)]

    det = det_curve(positives, negatives, [0.95, 0.2, 0.5, 0.8], HOP, SR)

    got = [(p.threshold, p.misses, p.false_alarms, p.fa_per_hour) for p in det]
    assert got == [
        (0.2, 0, 2, 2.0),
        (0.5, 1, 1, 1.0),
        (0.8, 2, 0, 0.0),
        (0.95, 3, 0, 0.0),
    ]
    assert det[1].frr == pytest.approx(1 / 3)
    assert det[2].frr_by_condition == {"noise": 1.0, "quiet": 0.5}
    assert all(p.negative_hours == 1.0 for p in det)
    # монотонность: FRR не падает, FA/h не растёт с порогом
    assert [p.frr for p in det] == sorted(p.frr for p in det)
    assert [p.fa_per_hour for p in det] == sorted((p.fa_per_hour for p in det), reverse=True)


def test_threshold_at_fa_picks_lowest_passing():
    def point(thr, fa):
        return DetPoint(thr, 0.0, fa, 0, 1, 0, 1.0)

    det = [point(0.2, 2.0), point(0.5, 0.1), point(0.8, 0.0)]
    assert threshold_at_fa(det, 0.1).threshold == 0.5
    assert threshold_at_fa([point(0.2, 2.0)], 0.1) is None
