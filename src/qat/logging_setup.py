"""Единая настройка логирования для всех модулей qat."""

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Настраивает корневой логгер: время, уровень, модуль, сообщение."""
    logging.basicConfig(level=level.upper(), format=LOG_FORMAT, force=True)
