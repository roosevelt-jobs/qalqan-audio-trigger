import hashlib
import json

import numpy as np
import soundfile as sf

from qat.detectors.energy import EnergyDetector
from qat.runtime.audio_io import SAMPLE_RATE
from qat.runtime.events import EventWriter
from qat.runtime.postproc import Trigger
from qat.runtime.ring_buffer import RingBuffer

SR = SAMPLE_RATE
CONTRACT_FIELDS = {
    "event_id", "timestamp_utc", "detector", "detector_version", "score", "threshold",
    "clip_path", "clip_sha256", "clip_start_offset_sec", "clip_end_offset_sec",
    "operator_decision",
}


def make_signal(seconds):
    """Различимые значения: сэмпл i = (i % 10000) / 10000."""
    return (np.arange(round(seconds * SR)) % 10000 / 10000).astype(np.float32)


def setup(tmp_path, written_sec):
    signal = make_signal(30)
    buffer = RingBuffer()
    buffer.write(signal[: round(written_sec * SR)])
    writer = EventWriter(EnergyDetector(), buffer, tmp_path / "events")
    return signal, buffer, writer


def read_json(tmp_path, event):
    return json.loads((tmp_path / "events" / f"{event.event_id}.json").read_text())


def test_two_stage_event_with_full_clip(tmp_path):
    signal, buffer, writer = setup(tmp_path, written_sec=12)
    center = 10 * SR

    first = writer.on_trigger(Trigger(center, 0.93, 0.8))

    assert set(read_json(tmp_path, first)) == CONTRACT_FIELDS
    assert first.clip_path is None and first.clip_sha256 is None
    assert read_json(tmp_path, first)["clip_path"] is None
    assert writer.poll() == []  # хвост «+5 с» ещё не записан

    buffer.write(signal[12 * SR : 15 * SR])
    (done,) = writer.poll()

    assert done.event_id == first.event_id
    assert (done.detector, done.score, done.threshold) == ("energy", 0.93, 0.8)
    assert done.operator_decision is None
    assert (done.clip_start_offset_sec, done.clip_end_offset_sec) == (-5.0, 5.0)
    assert done.clip_path == f"events/{done.event_id}.wav"

    clip_file = tmp_path / done.clip_path
    assert done.clip_sha256 == hashlib.sha256(clip_file.read_bytes()).hexdigest()
    audio, sr = sf.read(str(clip_file), dtype="float32")
    assert sr == SR and audio.ndim == 1
    assert len(audio) == 10 * SR
    np.testing.assert_array_equal(audio, signal[5 * SR : 15 * SR])

    assert read_json(tmp_path, done) == done.to_json()
    assert writer.poll() == []  # повторно не выдаётся


def test_clip_trimmed_at_stream_start(tmp_path):
    signal, buffer, writer = setup(tmp_path, written_sec=8)

    writer.on_trigger(Trigger(2 * SR, 0.9, 0.8))
    (done,) = writer.poll()

    assert (done.clip_start_offset_sec, done.clip_end_offset_sec) == (-2.0, 5.0)
    audio, _ = sf.read(str(tmp_path / done.clip_path), dtype="float32")
    np.testing.assert_array_equal(audio, signal[: 7 * SR])


def test_flush_at_stream_end_keeps_available_tail(tmp_path):
    signal, buffer, writer = setup(tmp_path, written_sec=20)

    writer.on_trigger(Trigger(18 * SR, 0.9, 0.8))
    assert writer.poll() == []
    (done,) = writer.flush()

    assert (done.clip_start_offset_sec, done.clip_end_offset_sec) == (-5.0, 2.0)
    audio, _ = sf.read(str(tmp_path / done.clip_path), dtype="float32")
    np.testing.assert_array_equal(audio, signal[13 * SR : 20 * SR])
    assert writer.flush() == []
