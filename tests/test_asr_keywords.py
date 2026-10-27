"""Детектор C. Фразы — голоса macOS (Milena — RU, Aru — KK), только для юнит-тестов.

Ключевые слова здесь тестовые, не итоговый список проекта.
"""

from pathlib import Path

import numpy as np
import pytest

from qat.detectors.asr_keywords import (
    MODEL_PATHS,
    RECOMMENDED_POSTPROC,
    AsrKeywordsDetector,
    match_score,
)
from qat.detectors.base import Detector
from qat.runtime.audio_io import SAMPLE_RATE, WavReplaySource
from qat.runtime.events import EventWriter
from qat.runtime.pipeline import run_events, run_scores
from qat.runtime.postproc import PostProcessor
from qat.runtime.ring_buffer import RingBuffer
from qat.runtime.wake import Wake

DATA = Path(__file__).parent / "data"
TEST_KEYWORDS = {"ru": ["помогите"], "kk": ["көмектесіңдер"]}
needs_models = pytest.mark.skipif(
    not all(p.is_dir() for p in MODEL_PATHS.values()),
    reason="нет моделей Vosk, см. models/vosk/README.md",
)


def max_score(detector, wav):
    detector.reset()
    scores = run_scores(WavReplaySource(DATA / wav), detector, Wake(), RingBuffer())
    return max((s.score for s in scores), default=0.0)


@pytest.fixture(scope="module")
def detector():
    return AsrKeywordsDetector(TEST_KEYWORDS)


def test_match_score_phrases_and_confidence():
    words = [("Позови", 0.9), ("на", 0.8), ("помощь", 0.6), ("ёлка", 1.0)]

    assert match_score(words, [("на", "помощь")]) == 0.6  # минимум по словам фразы
    assert match_score(words, [("помощь", "на")]) == 0.0  # порядок важен
    assert match_score(words, [("позови",)]) == 0.9  # регистр не важен
    assert match_score(words, [("елка",)]) == 1.0  # ё = е
    assert match_score([], [("помощь",)]) == 0.0


def test_keywords_and_languages_validated():
    with pytest.raises(ValueError, match="не заданы"):
        AsrKeywordsDetector({"ru": []})
    with pytest.raises(ValueError, match="нет моделей"):
        AsrKeywordsDetector({"de": ["hilfe"]})
    with pytest.raises(FileNotFoundError, match="README"):
        AsrKeywordsDetector({"ru": ["помогите"]}, model_paths={"ru": "/nonexistent"})


@needs_models
def test_matches_protocol(detector):
    assert isinstance(detector, Detector)
    assert detector.window_sec == detector.hop_sec


@needs_models
@pytest.mark.parametrize("lang", ["ru", "kk"])
def test_keyword_detected(detector, lang):
    assert max_score(detector, f"asr_{lang}_kw.wav") >= 0.9


@needs_models
@pytest.mark.parametrize("lang", ["ru", "kk"])
def test_neutral_speech_not_detected(detector, lang):
    assert max_score(detector, f"asr_{lang}_neutral.wav") == 0.0


@needs_models
def test_keyword_only_for_its_language():
    # только русское слово: казахская фраза с «көмектесіңдер» не срабатывает
    ru_only = AsrKeywordsDetector({"ru": ["помогите"]})
    assert max_score(ru_only, "asr_kk_kw.wav") == 0.0
    assert max_score(ru_only, "asr_ru_kw.wav") >= 0.9


@needs_models
def test_reset_drops_unfinished_speech(detector):
    speech = np.concatenate(list(WavReplaySource(DATA / "asr_ru_kw.wav")))
    window = round(detector.window_sec * SAMPLE_RATE)
    half = len(speech) // 2

    detector.reset()
    for i in range(0, half - window + 1, window):
        detector.process(speech[i : i + window])
    detector.reset()  # недослушанная фраза не должна «досказаться» тишиной
    silence = np.zeros(window, dtype=np.float32)
    assert max(detector.process(silence) for _ in range(8)) == 0.0


@needs_models
def test_pipeline_gives_one_event_per_keyword_file(detector, tmp_path):
    detector.reset()
    buffer = RingBuffer()
    hop = round(detector.hop_sec * SAMPLE_RATE)
    postproc = PostProcessor(hop, threshold=0.5, **RECOMMENDED_POSTPROC)
    writer = EventWriter(detector, buffer, tmp_path / "events")

    events = list(run_events(
        WavReplaySource(DATA / "asr_ru_kw.wav"), detector, Wake(), buffer, postproc, writer
    ))

    # «Помогите! Помогите, пожалуйста!» — одно событие (дебаунс 5 с), два этапа
    assert [e.clip_ready for e in events] == [False, True]
    assert events[0].detector == "asr_keywords"
