"""Несколько детекторов параллельно и слияние тревог («любой из двух», S14a)."""

from pathlib import Path

import numpy as np
import pytest

from qat.detectors.asr_keywords import MODEL_PATHS
from qat.detectors.registry import make_detector, make_postproc, parse_detector_list
from qat.runtime.audio_io import SAMPLE_RATE, WavReplaySource
from qat.runtime.events import MERGE_WINDOW_SEC, EventWriter
from qat.runtime.pipeline import _scores_per_block_multi, run_events_multi, run_scores
from qat.runtime.postproc import Trigger
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import Wake

DATA = Path(__file__).parent / "data"
SR = SAMPLE_RATE
needs_vosk = pytest.mark.skipif(
    not all(p.is_dir() for p in MODEL_PATHS.values()), reason="нет моделей Vosk"
)


class Fake:
    """Детектор без состояния: оценка — среднее модуля окна."""

    def __init__(self, name, window_sec, hop_sec, version="1"):
        self.name, self.version, self.window_sec, self.hop_sec = name, version, window_sec, hop_sec

    def process(self, w):
        return float(np.abs(w).mean())

    def reset(self):
        pass


class AlwaysAwake:
    def process(self, block):
        return True


def blocks(seconds=3.0, seed=0):
    x = np.random.default_rng(seed).uniform(-1, 1, int(seconds * SR)).astype(np.float32)
    return [x[i : i + 480] for i in range(0, len(x), 480)]


def test_each_detector_gets_same_scores_as_alone():
    a, b = Fake("a", 1.0, 0.25), Fake("b", 0.25, 0.25)
    src = blocks()

    together = [[], []]
    for per in _scores_per_block_multi(src, [a, b], AlwaysAwake(), RingBuffer()):
        together[0] += per[0]
        together[1] += per[1]

    assert together[0] == list(run_scores(src, a, AlwaysAwake(), RingBuffer()))
    assert together[1] == list(run_scores(src, b, AlwaysAwake(), RingBuffer()))


def make_writer(tmp_path, dets, seconds=30):
    buf = RingBuffer()
    buf.write(np.zeros(seconds * SR, dtype=np.float32))
    return EventWriter(dets, buf, tmp_path / "events", merge_window_sec=MERGE_WINDOW_SEC), buf


def test_merge_within_window_updates_same_event(tmp_path):
    c, b = Fake("asr_keywords", 0.25, 0.25, "0.1.0"), Fake("sed_panns", 1.0, 0.25, "0.2.0")
    writer, _ = make_writer(tmp_path, [c, b])

    first = writer.on_trigger(Trigger(10 * SR, 0.3, 0.15), b)  # сначала B
    joined = writer.on_trigger(Trigger(12 * SR, 0.9, 0.5), c)  # через 2 с — C

    assert joined.event_id == first.event_id
    assert first.detector == "sed_panns"
    assert joined.detector == "asr_keywords+sed_panns"  # порядок списка детекторов: C, затем B
    assert joined.detector_version == "0.1.0+0.2.0"
    assert (joined.score, joined.threshold) == (0.9, 0.5)  # у самой уверенной оценки
    assert writer.on_trigger(Trigger(12 * SR + 100, 0.95, 0.5), c) is None  # C уже в тревоге

    (done,) = writer.flush()
    assert done.event_id == first.event_id
    assert done.detector == "asr_keywords+sed_panns" and done.clip_ready


def test_far_triggers_are_separate_alarms(tmp_path):
    c, b = Fake("asr_keywords", 0.25, 0.25), Fake("sed_panns", 1.0, 0.25)
    writer, _ = make_writer(tmp_path, [c, b])

    e1 = writer.on_trigger(Trigger(10 * SR, 0.9, 0.5), c)
    e2 = writer.on_trigger(Trigger(10 * SR + int(3.5 * SR), 0.3, 0.15), b)  # 3,5 с > 3 с

    assert e1.event_id != e2.event_id
    assert (e1.detector, e2.detector) == ("asr_keywords", "sed_panns")


def test_single_detector_without_merge_unchanged(tmp_path):
    buf = RingBuffer()
    buf.write(np.zeros(20 * SR, dtype=np.float32))
    writer = EventWriter(Fake("energy", 0.5, 0.1), buf, tmp_path / "events")

    e1 = writer.on_trigger(Trigger(5 * SR, 0.9, 0.8))
    e2 = writer.on_trigger(Trigger(5 * SR + 10, 0.9, 0.8))

    assert e1.event_id != e2.event_id  # без окна слияния — как до S14a


def test_registry_lists_and_postproc():
    assert parse_detector_list("asr_keywords, sed_panns") == ["asr_keywords", "sed_panns"]
    for bad in ["", "nope", "energy,energy"]:
        with pytest.raises(ValueError):
            parse_detector_list(bad)

    c = Fake("asr_keywords", 0.25, 0.25)
    pp = make_postproc(c, SR)
    assert (pp.threshold, pp.smooth_windows, pp.consecutive) == (0.5, 1, 1)
    assert make_postproc(c, SR, threshold=0.7).threshold == 0.7
    assert make_postproc(Fake("sed_panns", 1.0, 0.25), SR).threshold == 0.15


@needs_vosk
def test_default_keywords_end_to_end(tmp_path):
    c = make_detector("asr_keywords")

    def alarms(wav):
        c.reset()
        buf = RingBuffer()
        writer = EventWriter([c], buf, tmp_path / wav, merge_window_sec=MERGE_WINDOW_SEC)
        evs = run_events_multi(WavReplaySource(DATA / wav), [c], Wake(), buf,
                               [make_postproc(c, SR)], writer)
        return [e for e in evs if e.clip_ready]

    assert len(alarms("asr_en_kw.wav")) == 1  # «Help! Help, please!»
    assert alarms("asr_en_neutral.wav") == []
    assert alarms("asr_ru_kw.wav") == []  # русский больше не кодовая фраза (S15)
    assert alarms("asr_kk_kw.wav") == []


@needs_vosk
def test_two_detectors_one_alarm(tmp_path):
    c, e = make_detector("asr_keywords"), make_detector("energy")  # energy — вместо B, быстро
    buf = RingBuffer()
    writer = EventWriter([c, e], buf, tmp_path / "ev", merge_window_sec=MERGE_WINDOW_SEC)
    pps = [make_postproc(c, SR), make_postproc(e, SR, threshold=0.6)]

    events = list(run_events_multi(WavReplaySource(DATA / "asr_en_kw.wav"), [c, e],
                                   Wake(), buf, pps, writer))

    final = [x for x in events if x.clip_ready]
    assert len(final) == 1
    assert final[0].detector == "asr_keywords+energy"
    assert len({x.event_id for x in events}) == 1  # все этапы — одна тревога


class Scripted:
    """Пробуждение по заданному списку ответов, по одному на блок."""

    def __init__(self, answers):
        self.answers = iter(answers)

    def process(self, block):
        return next(self.answers)


def test_per_detector_wake():
    a, b = Fake("a", 0.25, 0.25), Fake("b", 0.25, 0.25)
    src = blocks(2.0)
    n = len(src)
    shared = Scripted([False] * n)  # общее пробуждение спит
    own_b = Scripted([True] * n)  # у b своё — бодрствует

    got = [[], []]
    for per in _scores_per_block_multi(src, [a, b], shared, RingBuffer(), [None, own_b]):
        got[0] += per[0]
        got[1] += per[1]

    assert got[0] == []  # a живёт по общему пробуждению
    assert got[1] == list(run_scores(src, b, AlwaysAwake(), RingBuffer()))


def test_registry_wake_for_b_is_loudness_only():
    from qat.detectors.registry import make_wake

    assert make_wake(Fake("asr_keywords", 0.25, 0.25)) is None  # C — общее (с VAD)
    wake_b = make_wake(Fake("sed_panns", 1.0, 0.25))
    t = np.arange(SR) / SR
    # тихий визг-тон ~ -33 dBFS: не речь, ниже громкого порога общего пробуждения
    tone = (0.03 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
    tone_blocks = [tone[i : i + 480] for i in range(0, len(tone) - 479, 480)]

    assert all(wake_b.process(x) for x in tone_blocks)
    shared = Wake()
    assert not any(shared.process(x) for x in tone_blocks)
