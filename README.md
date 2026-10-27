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

Ручной прогон WAV через весь конвейер — см. раздел S6.

## S5: постобработка

Модуль `src/qat/runtime/postproc.py`: `PostProcessor.process(score) -> Trigger | None`.
Шаги: скользящее среднее по `smooth_windows` окнам → порог (`>=`) → `consecutive` окон подряд →
дебаунс `debounce_sec`. Срабатывает один раз на эпизод превышения порога. Разрыв в сетке окон
(wake был неактивен) сбрасывает сглаживание и счётчик. По умолчанию: 0.8, 3 окна, 2 подряд, 5 с.

```bash
uv run pytest tests/test_postproc.py
```

## S6: события

Модуль `src/qat/runtime/events.py`: `EventWriter` пишет `events/<event_id>.json` и
`events/<event_id>.wav` (16 кГц, моно, float32). Событие выдаётся в два этапа:

1. Сразу при срабатывании — JSON, где `clip_path`, `clip_sha256` и смещения клипа = `null`.
2. Когда записаны 5 с после срабатывания — тот же `event_id` с клипом −5…+5 с и SHA-256 файла клипа.

Если срабатывание было раньше 5 с от старта или поток кончился раньше «+5 с», клип обрезается
до доступного аудио, а в JSON пишутся фактические смещения. Тишиной клип не дополняется.

`pipeline.run_events(...)` связывает весь конвейер: источник → буфер → пробуждение → детектор →
постобработка → события.

```bash
uv run pytest tests/test_events.py tests/test_pipeline.py
```

Ручной прогон WAV с заглушкой `energy`: печатает оба этапа событий, клипы и JSON — в `events/`.

```bash
uv run python -m qat.runtime.pipeline --wav tests/data/speech_ru.wav --threshold 0.6
```

Проверить хеш клипа:

```bash
shasum -a 256 events/<event_id>.wav
```

## S7: сервер и экран оператора

`server/app.py` — FastAPI + WebSocket, `server/static/index.html` — экран оператора. Конвейер
(детектор пока заглушка `energy`) работает в фоновом потоке сервера. Сервер слушает только
`127.0.0.1:8000`. Телеметрия FastAPI (OpenTelemetry) выключена в коде.

| Адрес | Что делает |
|---|---|
| `GET /` | экран оператора |
| `WS /ws` | история (последние 50 событий) при подключении, дальше — новые события и обновления |
| `GET /clips/<event_id>.wav` | клип события |
| `POST /events/<event_id>/decision` | `{"decision": "confirmed" \| "rejected"}` → JSON события и журнал `events/decisions.jsonl` |

На карточке тревоги видна задержка от срабатывания до появления на экране (цель ≤ 500 мс).

Запуск с микрофоном, затем открыть http://127.0.0.1:8000 и громко сказать что-нибудь или хлопнуть:

```bash
uv run python -m server.app --mic
```

Демо на WAV в темпе реального времени:

```bash
uv run python -m server.app --wav tests/data/speech_ru.wav --realtime --threshold 0.6
```

Тесты:

```bash
uv run pytest tests/test_server.py
```

## S8: стенд оценки

`src/qat/eval/metrics.py` — FRR, FA/h, DET-кривая, выбор рабочего порога (самый низкий порог с
FA/h ≤ 0,1). `src/qat/eval/bench.py` — прогон любого `Detector` по манифесту через тот же конвейер,
что и в живом режиме; пороги перебираются на сохранённых оценках постобработкой S5.

Манифест — JSONL, пути относительно манифеста:

```json
{"path": "pos/001.wav", "label": "positive", "condition": "noise", "target_start_sec": 1.2, "target_end_sec": 2.0}
{"path": "neg/stream_01.wav", "label": "negative"}
```

Позитив засчитан, если тревога была от начала цели до её конца + 1 с. FRR считается и по
условиям (`condition`). Если негативного потока меньше 10 ч, в отчёте будет предупреждение:
FA/h ≤ 0,1 на таком объёме не подтвердить.

```bash
uv run python -m qat.eval.bench --manifest manifest.jsonl --detector energy --out reports/energy
```

Результат: `report.json` (рабочий порог, FRR по условиям, вся DET-кривая, предупреждения) и
`det.csv`. Флаг `--no-wake` оценивает детектор без пробуждения.

```bash
uv run pytest tests/test_eval_metrics.py tests/test_eval_bench.py
```

## S9: детектор B (PANNs)

`src/qat/detectors/sed_panns.py` — `SedPannsDetector` на предобученной PANNs Cnn14_16k (AudioSet,
527 классов). Оценка окна = максимум вероятностей по классам «крик и возглас»: Screaming, Shout,
Yell, Bellow, Battle cry, Children shouting. Окно 1 с, шаг 0,25 с, CPU в один поток.
Код модели — `src/qat/detectors/panns_cnn14.py` (перенесён у авторов, MIT).
Источники, лицензии и хеши — `models/panns/README.md`.

Веса (359 МБ) в git не входят, их нужно скачать один раз:

```bash
curl -L -o "models/panns/Cnn14_16k_mAP=0.438.pth" "https://zenodo.org/records/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
```

Детектор выбирается по имени (`src/qat/detectors/registry.py`) в стенде, сервере и CLI конвейера:

```bash
uv run python -m server.app --mic --detector sed_panns --threshold 0.5
```

```bash
uv run python -m qat.eval.bench --manifest manifest.jsonl --detector sed_panns --out reports/sed_panns
```

Тесты (без весов два теста пропускаются):

```bash
uv run pytest tests/test_sed_panns.py
```
