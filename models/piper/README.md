# Голоса Piper (синтез обучающих данных для модели A)

Источник: `rhasspy/piper-voices@5041a7a49ae5b3a0f30e6837769eb03fdbb231bd` (Hugging Face).
В git не входят (~365 МБ). Рядом с каждым `.onnx` лежат `.onnx.json` и `MODEL_CARD`.

| Файл | Байт | SHA-256 | Лицензия данных |
|---|---|---|---|
| `kk_KZ-issai-high.onnx` | 127 864 258 | `4dee767c893e8535da821447d12cb030e3569e11254c14030a1da5d8b2222c16` | CC BY 4.0 |
| `kk_KZ-iseke-x_low.onnx` | 28 130 791 | `dcf526e1c089823c9f896f8f7fddd9b05d538752e139c8a2b4cf4fba41c9715c` | CC BY 4.0 |
| `kk_KZ-raya-x_low.onnx` | 28 130 791 | `75552af00254baee9f85b976877e039f9530ea86ddb83d70262b72925785a0aa` | CC BY 4.0 |
| `ru_RU-denis-medium.onnx` | 63 201 294 | `15fab56e11a097858ee115545d0f697fc2a316c41a291a5362349fb870411b0a` | CC0 |
| `ru_RU-dmitri-medium.onnx` | 63 201 294 | `f073356ebc4bd0f80c5af58df2953a5988bd5bdab1eb38635ce960b071fbefcb` | CC0 |
| `ru_RU-ruslan-medium.onnx` | 63 201 294 | `72a5f88e0b20928064eb45d88e1daa21f8af62d18613580d32cbb4aed48dcf7f` | CC BY-NC-SA 4.0 — только исследование |

Скачать (из корня репозитория):

```bash
cd models/piper && R=https://huggingface.co/rhasspy/piper-voices/resolve/5041a7a49ae5b3a0f30e6837769eb03fdbb231bd && for v in kk/kk_KZ/issai/high/kk_KZ-issai-high kk/kk_KZ/iseke/x_low/kk_KZ-iseke-x_low kk/kk_KZ/raya/x_low/kk_KZ-raya-x_low ru/ru_RU/denis/medium/ru_RU-denis-medium ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium ru/ru_RU/ruslan/medium/ru_RU-ruslan-medium; do n=$(basename $v); curl -L -o $n.onnx "$R/$v.onnx" && curl -L -o $n.onnx.json "$R/$v.onnx.json" && curl -L -o $n.MODEL_CARD "$R/$(dirname $v)/MODEL_CARD"; done
```

Программа Piper (`piper-tts`, GPL-3.0) стоит только в группе `dev` и вызывается отдельным
процессом (`python -m piper`) — в код продукта не импортируется.
