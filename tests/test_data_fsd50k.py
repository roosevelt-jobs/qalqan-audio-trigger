import hashlib

import numpy as np
import pytest
import soundfile as sf

from qat.data.fsd50k import (
    build_manifest,
    read_ground_truth,
    sample_negatives,
    to_16k_mono,
    unexpected_members,
    verify_md5,
)


def test_build_manifest_known_answer():
    gt = {
        "3": {"Screaming", "Human_voice"},
        "1": {"Speech", "Male_speech_and_man_speaking"},
        "2": {"Shout", "Yell"},
        "4": {"Crying_and_sobbing"},
        "5": {"Crowd", "Shout"},  # крик в толпе — позитив важнее исключения
        "6": {"Cheering"},
        "7": {"Dog", "Bark"},
    }

    rows = build_manifest(gt)

    assert rows == [
        {"path": "1.wav", "label": "negative"},
        {"path": "2.wav", "label": "positive", "condition": "Yell"},  # самая узкая метка
        {"path": "3.wav", "label": "positive", "condition": "Screaming"},
        {"path": "5.wav", "label": "positive", "condition": "Shout"},
        {"path": "7.wav", "label": "negative"},
    ]


def test_build_manifest_adds_uploader_group():
    gt = {"1": {"Shout"}, "2": {"Dog"}}

    rows = build_manifest(gt, uploaders={"1": "alice", "2": "bob"})

    assert [r["group"] for r in rows] == ["alice", "bob"]


def test_read_ground_truth(tmp_path):
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text('fname,labels,mids\n10,"Shout,Human_voice","/m/1,/m/2"\n11,Dog,/m/3\n')

    assert read_ground_truth(csv_path) == {"10": {"Shout", "Human_voice"}, "11": {"Dog"}}


def test_verify_md5_detects_tampering(tmp_path):
    good, bad = tmp_path / "a.zip", tmp_path / "b.zip"
    good.write_bytes(b"original")
    bad.write_bytes(b"tampered")
    expected = tmp_path / "expected.md5"
    expected.write_text(
        f"{hashlib.md5(b'original').hexdigest()}  a.zip\n"
        f"{hashlib.md5(b'original').hexdigest()}  b.zip\n"
        f"{hashlib.md5(b'x').hexdigest()}  missing.zip\n"
    )

    problems = verify_md5(tmp_path, expected)

    assert len(problems) == 2
    assert problems[0].startswith("b.zip: md5")
    assert problems[1] == "missing.zip: нет файла"


def test_unexpected_members_blocks_non_wav_and_path_escape():
    names = [
        "FSD50K.eval_audio/",
        "FSD50K.eval_audio/1.wav",
        "FSD50K.eval_audio/2.WAV",
        "FSD50K.eval_audio/run.sh",
        "../evil.wav",
        "/abs/x.wav",
    ]
    assert unexpected_members(names, (".wav",)) == [
        "FSD50K.eval_audio/run.sh",
        "../evil.wav",
        "/abs/x.wav",
    ]


def test_to_16k_mono_keeps_duration(tmp_path):
    sr = 44100
    t = np.arange(int(2.5 * sr)) / sr
    stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
    src = tmp_path / "src.wav"
    sf.write(str(src), stereo, sr, subtype="PCM_16")

    duration = to_16k_mono(src, tmp_path / "out" / "dst.wav")

    info = sf.info(str(tmp_path / "out" / "dst.wav"))
    assert (info.samplerate, info.channels, info.subtype) == (16000, 1, "PCM_16")
    assert duration == pytest.approx(2.5, abs=1e-3)
    assert info.duration == pytest.approx(2.5, abs=1e-3)


def test_sample_negatives_keeps_positives_and_is_reproducible():
    rows = [{"path": f"{i}.wav", "label": "positive" if i % 10 == 0 else "negative"}
            for i in range(100)]

    a = sample_negatives(rows, 20, seed=0)
    b = sample_negatives(rows, 20, seed=0)
    c = sample_negatives(rows, 20, seed=1)

    assert a == b
    assert a != c
    assert sum(r["label"] == "positive" for r in a) == 10
    assert sum(r["label"] == "negative" for r in a) == 20
    assert [rows.index(r) for r in a] == sorted(rows.index(r) for r in a)  # порядок сохранён
    assert sample_negatives(rows, 1000) == rows
