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

## S3: пробуждение

Модуль `src/qat/runtime/wake.py`: `Wake.process(block) -> bool` решает, будить ли детектор.
Ниже −50 dBFS — тихо; выше −20 dBFS — громкий звук будит сразу; между ними решает Silero VAD.
После активности состояние держится ещё 1 с.

Модель: `models/vad/silero_vad.onnx` из пакета silero-vad 6.2.3 (MIT, `models/vad/LICENSE`),
запускается через onnxruntime только на CPU. При загрузке сверяется SHA-256 файла.

Тестовая фраза `tests/data/speech_ru.wav` сгенерирована офлайн голосом macOS и нужна только для
юнит-теста VAD, не для оценки моделей:

```bash
say -v Milena -o phrase.wav --data-format=LEF32@16000 "Внимание! Мне нужна помощь, срочно вызовите скорую."
```

К фразе добавлено по 2 с тишины с каждой стороны, файл сохранён как 16-бит PCM, 16 кГц.

```bash
uv run pytest tests/test_wake.py
```

## S4: интерфейс детектора + заглушка

- `src/qat/detectors/base.py` — протокол `Detector` по контракту.
- `src/qat/detectors/energy.py` — **заглушка**: оценка = громкость окна (−60 dBFS → 0, 0 dBFS → 1).
  Тревогу не детектирует, нужна для отладки конвейера.
- `src/qat/runtime/pipeline.py` — `run_scores(source, detector, wake, buffer)`: поток оценок
  `Score(sample, score)`. Концы окон кратны шагу детектора; при неактивном wake окна пропускаются.

```bash
uv run pytest tests/test_detectors.py tests/test_pipeline.py
```

Ручной прогон WAV через конвейер с заглушкой (максимум оценки по секундам, «-» — детектор не будили):

```bash
uv run python -m qat.runtime.pipeline --wav tests/data/speech_ru.wav
```
