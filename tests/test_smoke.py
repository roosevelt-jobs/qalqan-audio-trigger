import logging
from pathlib import Path

import qat
from qat.logging_setup import setup_logging

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert qat.__version__ == "0.1.0"


def test_setup_logging_sets_level():
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG
    setup_logging("warning")
    assert logging.getLogger().level == logging.WARNING


def test_runtime_config_exists():
    assert (ROOT / "configs" / "runtime.yaml").is_file()
