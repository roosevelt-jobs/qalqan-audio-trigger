# Qalqan Audio Trigger (MVP)

Прототип звукового триггера тревоги для бейджа Qalqan: слушает микрофон, детектирует
тревожный сигнал и показывает тревогу оператору в браузере. Правила и архитектура —
в [CLAUDE.md](CLAUDE.md).

## Требования

- macOS (Apple Silicon)
- [uv](https://docs.astral.sh/uv/)
- Python 3.11 (`uv python install 3.11`)

## S0: скелет проекта

Установить окружение и инструменты разработки:

```bash
uv sync
```

Запустить тесты:

```bash
uv run pytest
```

Проверить стиль:

```bash
uv run ruff check
```
