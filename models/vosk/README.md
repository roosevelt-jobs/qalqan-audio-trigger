# Модели Vosk (детектор C)

| Модель | Язык | Размер zip | SHA-256 zip | Лицензия |
|---|---|---|---|---|
| `vosk-model-small-ru-0.22` | русский | 46 236 750 байт | `961d5ff98a17f4aa6de69864d0aa71fa5bac682301d2b5d17a3f24c5c99a46d4` | Apache 2.0 |
| `vosk-model-small-kz-0.42` | казахский | 59 697 294 байт | `b6d6bfe6195c866805bd18b1f56723e4d2ba4866e81df1481006d0c5704fda4f` | Apache 2.0 |

Источник — сайт авторов Vosk (Alpha Cephei): https://alphacephei.com/vosk/models.
На Hugging Face у авторов (`alphacep`) казахской модели нет, а русские там — другого
поколения (ONNX-трансдьюсеры), пакет `vosk` с ними не работает.

В git модели не входят. Скачать и распаковать сюда:

```bash
cd models/vosk && for m in vosk-model-small-ru-0.22 vosk-model-small-kz-0.42; do curl -L -O "https://alphacephei.com/vosk/models/$m.zip" && shasum -a 256 $m.zip && unzip -q $m.zip && rm $m.zip; done
```

Пакет `vosk` закреплён на 0.3.44: у 0.3.45 нет сборки для macOS.
