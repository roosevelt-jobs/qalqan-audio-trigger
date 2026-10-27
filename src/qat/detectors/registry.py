"""Выбор детектора по имени (стенд оценки, сервер, CLI)."""

from collections.abc import Callable

from qat.detectors.base import Detector


def _energy() -> Detector:
    from qat.detectors.energy import EnergyDetector

    return EnergyDetector()


def _sed_panns() -> Detector:
    # Импорт внутри: torch и веса грузятся, только если выбран этот детектор.
    from qat.detectors.sed_panns import SedPannsDetector

    return SedPannsDetector()


DETECTORS: dict[str, Callable[[], Detector]] = {
    "energy": _energy,
    "sed_panns": _sed_panns,
}


def make_detector(name: str) -> Detector:
    try:
        return DETECTORS[name]()
    except KeyError:
        raise ValueError(f"неизвестный детектор {name!r}, есть: {sorted(DETECTORS)}") from None
