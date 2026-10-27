import hashlib
from pathlib import Path

import numpy as np

from qat.detectors.energy import EnergyDetector
from qat.runtime.audio_io import BLOCK_SIZE, SAMPLE_RATE, WavReplaySource
from qat.runtime.events import EventWriter
from qat.runtime.pipeline import run_events, run_scores
from qat.runtime.postproc import PostProcessor
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import Wake

SPEECH_WAV = Path(__file__).parent / "data" / "speech_ru.wav"


class RecordingDetector:
    """Тестовый детектор: запоминает окна, возвращает 0.5."""

    name = "recording"
    version = "0"
    window_sec = 0.5
    hop_sec = 0.1

    def __init__(self):
        self.windows = []
        self.resets = 0

    def process(self, audio_window):
        self.windows.append(audio_window.copy())
        return 0.5

    def reset(self):
        self.resets += 1


class ScriptedWake:
    """Тестовое пробуждение: отвечает по заданному списку, по одному значению на блок."""

    def __init__(self, answers):
        self.answers = iter(answers)

    def process(self, block):
        return next(self.answers)


def blocks_of(signal):
    return [signal[i : i + BLOCK_SIZE] for i in range(0, len(signal), BLOCK_SIZE)]


def test_windows_on_fixed_grid_and_match_source():
    signal = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / (SAMPLE_RATE * 2)
    blocks = blocks_of(signal)
    det = RecordingDetector()

    scores = list(run_scores(blocks, det, ScriptedWake([True] * len(blocks)), RingBuffer()))

    window, hop = 8000, 1600
    ends = [s.sample for s in scores]
    assert ends == list(range(window, len(signal) + 1, hop))
    for end, audio in zip(ends, det.windows, strict=True):
        assert len(audio) == window
        np.testing.assert_array_equal(audio, signal[end - window : end])


def test_inactive_wake_skips_windows_and_resets_detector():
    signal = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    blocks = blocks_of(signal)
    half = len(blocks) // 2
    det = RecordingDetector()
    wake = ScriptedWake([True] * half + [False] * (len(blocks) - half))

    scores = list(run_scores(blocks, det, wake, RingBuffer()))

    assert scores
    assert all(s.sample <= (half + 1) * BLOCK_SIZE for s in scores)
    assert det.resets == 1


def test_replay_speech_with_energy_stub():
    def run():
        source = WavReplaySource(SPEECH_WAV)
        return list(run_scores(source, EnergyDetector(), Wake(), RingBuffer()))

    scores = run()
    t = np.array([s.sample for s in scores]) / SAMPLE_RATE
    values = np.array([s.score for s in scores])

    assert len(scores) > 0
    assert t.min() >= 2.0  # в начальной тишине 0–2 с оценок нет
    assert values[(t >= 2.5) & (t < 5.5)].min() > 0.4
    assert run() == scores  # детерминированно


def test_replay_speech_produces_one_event_with_clip(tmp_path):
    detector = EnergyDetector()
    buffer = RingBuffer()
    postproc = PostProcessor(round(detector.hop_sec * SAMPLE_RATE), threshold=0.6)
    writer = EventWriter(detector, buffer, tmp_path / "events")

    events = list(
        run_events(WavReplaySource(SPEECH_WAV), detector, Wake(), buffer, postproc, writer)
    )

    # этап 1 (без клипа), затем этап 2 (с клипом) для одного и того же события
    assert [e.clip_ready for e in events] == [False, True]
    first, done = events
    assert done.event_id == first.event_id
    assert done.score >= 0.6
    assert -5.0 <= done.clip_start_offset_sec < 0 < done.clip_end_offset_sec <= 5.0
    clip = tmp_path / done.clip_path
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == done.clip_sha256
