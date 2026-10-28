"""Выбор детектора по имени (стенд оценки, сервер, CLI) и его рабочие настройки."""

from collections.abc import Callable
from dataclasses import dataclass, field

from qat.detectors.base import Detector

# Кодовые фразы детектора C по умолчанию (выбраны в S14a, CLAUDE.md раздел 4.2).
DEFAULT_KEYWORDS = {"ru": ["на помощь"], "kk": ["көмектесіңдер"]}


@dataclass(frozen=True)
class DetectorConfig:
    threshold: float
    postproc: dict = field(default_factory=dict)  # параметры PostProcessor кроме порога


# ASSUMPTION: energy — заглушка (порог как раньше); sed_panns — порог из S9.1;
# asr_keywords — оценка-всплеск, поэтому без сглаживания и «1 подряд».
CONFIGS = {
    "energy": DetectorConfig(threshold=0.8),
    "sed_panns": DetectorConfig(threshold=0.15),
    "asr_keywords": DetectorConfig(
        threshold=0.5, postproc={"smooth_windows": 1, "consecutive": 1}
    ),
}


def _energy() -> Detector:
    from qat.detectors.energy import EnergyDetector

    return EnergyDetector()


def _sed_panns() -> Detector:
    # Импорт внутри: torch и веса грузятся, только если выбран этот детектор.
    from qat.detectors.sed_panns import SedPannsDetector

    return SedPannsDetector()


def _asr_keywords() -> Detector:
    from qat.detectors.asr_keywords import AsrKeywordsDetector

    return AsrKeywordsDetector(DEFAULT_KEYWORDS)


DETECTORS: dict[str, Callable[[], Detector]] = {
    "energy": _energy,
    "sed_panns": _sed_panns,
    "asr_keywords": _asr_keywords,
}


def make_detector(name: str) -> Detector:
    try:
        return DETECTORS[name]()
    except KeyError:
        raise ValueError(f"неизвестный детектор {name!r}, есть: {sorted(DETECTORS)}") from None


def parse_detector_list(value: str) -> list[str]:
    """'asr_keywords,sed_panns' -> ['asr_keywords', 'sed_panns'] с проверкой имён и повторов."""
    names = [n.strip() for n in value.split(",") if n.strip()]
    unknown = [n for n in names if n not in DETECTORS]
    if not names or unknown or len(set(names)) != len(names):
        raise ValueError(
            f"список детекторов {value!r}: неизвестные {unknown}, есть {sorted(DETECTORS)}"
        )
    return names


def make_postproc(detector: Detector, sample_rate: int, threshold: float | None = None):
    """PostProcessor с рабочими настройками детектора; threshold переопределяет порог."""
    from qat.runtime.postproc import PostProcessor

    cfg = CONFIGS.get(detector.name, DetectorConfig(threshold=0.8))
    hop = round(detector.hop_sec * sample_rate)
    return PostProcessor(hop, threshold=cfg.threshold if threshold is None else threshold,
                         **cfg.postproc)
