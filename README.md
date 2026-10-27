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

## S1: источник звука

Модуль `src/qat/runtime/audio_io.py`: `WavReplaySource` (replay WAV) и `MicrophoneSource`
(микрофон). Оба отдают блоки float32, моно, 16 кГц, по 480 сэмплов (30 мс).

Тесты replay:

```bash
uv run pytest tests/test_audio_io.py
```

Ручная проверка микрофона: раз в секунду печатает число блоков и громкость (RMS).
Во время речи RMS должен расти. При первом запуске macOS попросит доступ к микрофону.

```bash
uv run python -m qat.runtime.audio_io --seconds 5
```

Replay принимает только WAV с частотой 16 кГц. Другие файлы перевести встроенной утилитой macOS:

```bash
afconvert -f WAVE -d LEF32@16000 input.wav input_16k.wav
```

## S2: кольцевой буфер

Модуль `src/qat/runtime/ring_buffer.py`: `RingBuffer` хранит последние 30 с аудио (по умолчанию).
Время адресуется номером сэмпла от старта потока. `clip(center, -5.0, 5.0)` вырезает клип
вокруг события. Если нужных сэмплов уже нет или они ещё не записаны, будет `ValueError`;
что есть в буфере, показывает `available_range()`.

```bash
uv run pytest tests/test_ring_buffer.py
```
