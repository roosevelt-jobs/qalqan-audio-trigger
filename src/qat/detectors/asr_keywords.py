"""Детектор C: распознавание речи (Vosk, EN; RU/KK тоже) и поиск ключевых слов в тексте.

Распознаватели работают потоком: окно = шагу, каждому окну подаётся только новый звук.
Оценка окна:
- финальный результат Vosk — уверенность в найденной ключевой фразе (минимум по её словам);
- промежуточный результат — 1.0, если фраза есть в промежуточном тексте;
- 0, если фразы нет.
Промежуточные результаты нужны ради задержки: финальный Vosk выдаёт только после паузы,
а пробуждение к тому времени может погаснуть (проверено на тестовых фразах).

Приватность (CLAUDE.md, раздел 10): распознанный текст живёт только внутри process()
и никуда не пишется и не логируется.

Полный словарь, а не режим grammar: в grammar-режиме Vosk натягивает посторонние слова
на ключевые (проверено: «пожалуйста» -> «помогите» с уверенностью 1.0).
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import vosk

from qat.runtime.audio_io import SAMPLE_RATE

VOSK_DIR = Path(__file__).resolve().parents[3] / "models" / "vosk"
MODEL_PATHS = {
    "en": VOSK_DIR / "vosk-model-small-en-us-0.15",
    "ru": VOSK_DIR / "vosk-model-small-ru-0.22",
    "kk": VOSK_DIR / "vosk-model-small-kz-0.42",
}
# Оценка — редкий одиночный всплеск, сглаживание и «2 подряд» его бы погасили.
RECOMMENDED_POSTPROC = {"smooth_windows": 1, "consecutive": 1}


def normalize(word: str) -> str:
    return word.lower().replace("ё", "е")


def match_score(words: Sequence[tuple[str, float]], phrases: Sequence[Sequence[str]]) -> float:
    """Максимальная уверенность среди вхождений ключевых фраз; фраза — слова подряд."""
    best = 0.0
    tokens = [normalize(w) for w, _ in words]
    for phrase in phrases:
        n = len(phrase)
        for i in range(len(tokens) - n + 1):
            if tokens[i : i + n] == list(phrase):
                best = max(best, min(conf for _, conf in words[i : i + n]))
    return best


class AsrKeywordsDetector:
    name = "asr_keywords"
    version = "0.1.0"
    # ASSUMPTION: окно = шаг = 0.25 с: распознаватель потоковый, перекрытие не нужно.
    window_sec = 0.25
    hop_sec = 0.25

    def __init__(
        self,
        keywords: Mapping[str, Sequence[str]],
        model_paths: Mapping[str, str | Path] = MODEL_PATHS,
    ):
        """keywords: язык -> ключевые фразы, например {"ru": ["помогите"], "kk": ["көмек"]}."""
        if not any(keywords.values()):
            raise ValueError("ключевые слова не заданы")
        unknown = set(keywords) - set(model_paths)
        if unknown:
            raise ValueError(
                f"нет моделей для языков {sorted(unknown)}, есть {sorted(model_paths)}"
            )
        vosk.SetLogLevel(-1)
        self._phrases = {
            lang: [tuple(normalize(w) for w in p.split()) for p in phrases]
            for lang, phrases in keywords.items()
            if phrases
        }
        self._models = {}
        for lang in self._phrases:
            path = Path(model_paths[lang])
            if not path.is_dir():
                raise FileNotFoundError(f"нет модели {path}; как скачать — models/vosk/README.md")
            self._models[lang] = vosk.Model(str(path))
        self.reset()

    def reset(self) -> None:
        self._recognizers = {}
        for lang, model in self._models.items():
            rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)
            rec.SetWords(True)
            self._recognizers[lang] = rec

    def process(self, audio_window: np.ndarray) -> float:
        pcm = (np.clip(audio_window, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        score = 0.0
        for lang, rec in self._recognizers.items():
            if rec.AcceptWaveform(pcm):
                result = json.loads(rec.Result())
                words = [(w["word"], float(w["conf"])) for w in result.get("result", [])]
            else:
                # ASSUMPTION: у промежуточного текста нет уверенности — считаем её 1.0.
                # Гипотеза может измениться, мимолётное ложное слово даст тревогу.
                partial = json.loads(rec.PartialResult()).get("partial", "")
                words = [(w, 1.0) for w in partial.split()]
            score = max(score, match_score(words, self._phrases[lang]))
        return score
