import pytest

from qat.runtime.pipeline import Score
from qat.runtime.postproc import PostProcessor, Trigger

HOP = 1600  # 0.1 с при 16 кГц; дебаунс 5 с = 50 шагов


def feed(pp, values, start=HOP):
    """Подаёт оценки на сетке с шагом HOP, возвращает срабатывания."""
    out = []
    for i, v in enumerate(values):
        t = pp.process(Score(start + i * HOP, v))
        if t is not None:
            out.append(t)
    return out


def test_single_spike_needs_consecutive():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=1, consecutive=2)

    assert feed(pp, [0.1, 0.95, 0.1, 0.1]) == []


def test_smoothing_suppresses_single_spike():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=3, consecutive=1)

    # среднее (0.1, 0.1, 0.95) = 0.38 < 0.8
    assert feed(pp, [0.1, 0.1, 0.95, 0.1, 0.1]) == []


def test_sustained_episode_fires_once_at_nth_window():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=1, consecutive=3)

    triggers = feed(pp, [0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    # окна 2, 3, 4 выше порога -> срабатывание на третьем из них
    assert triggers == [Trigger(4 * HOP, 0.9, 0.8)]


def test_two_episodes_beyond_debounce_fire_twice():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=1, consecutive=2)

    values = [0.9, 0.9] + [0.1] * 60 + [0.9, 0.9]
    triggers = feed(pp, values)

    assert [t.sample for t in triggers] == [2 * HOP, 64 * HOP]


def test_second_episode_within_debounce_is_ignored():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=1, consecutive=2)

    # второй эпизод через ~2 с, всё ещё внутри дебаунса и продолжается за его пределы
    values = [0.9, 0.9] + [0.1] * 20 + [0.9] * 40
    triggers = feed(pp, values)

    assert [t.sample for t in triggers] == [2 * HOP]


def test_gap_in_grid_resets_counter():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=1, consecutive=2)

    assert pp.process(Score(HOP, 0.9)) is None
    # следующее окно не соседнее (wake был неактивен) -> счёт с нуля
    assert pp.process(Score(10 * HOP, 0.9)) is None
    assert pp.process(Score(11 * HOP, 0.9)) == Trigger(11 * HOP, 0.9, 0.8)


def test_gap_resets_smoothing_history():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=3, consecutive=1)

    feed(pp, [0.1, 0.1, 0.1])
    # без сброса среднее было бы (0.1, 0.1, 0.9) = 0.37
    assert pp.process(Score(20 * HOP, 0.9)) == Trigger(20 * HOP, 0.9, 0.8)


def test_score_equal_to_threshold_fires():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=1, consecutive=1)

    assert feed(pp, [0.8]) == [Trigger(HOP, 0.8, 0.8)]


def test_reset_clears_state():
    pp = PostProcessor(HOP, threshold=0.8, smooth_windows=1, consecutive=2)

    assert len(feed(pp, [0.9, 0.9])) == 1
    pp.reset()
    # без reset это было бы внутри дебаунса и того же эпизода
    assert len(feed(pp, [0.9, 0.9], start=3 * HOP)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"hop_samples": 0}, {"smooth_windows": 0}, {"consecutive": 0}],
)
def test_invalid_params(kwargs):
    params = {"hop_samples": HOP} | kwargs
    with pytest.raises(ValueError):
        PostProcessor(**params)
