#!/usr/bin/env bash
# Установка проекта в Google Colab (Linux). Запуск из корня проекта:
#   bash scripts/colab_setup.sh kws
# Профиль ассетов: kws (данные для модели A), detectors (модели B и C), all.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-kws}"

# sounddevice на Linux нужна системная библиотека PortAudio (на Mac она внутри пакета).
apt-get -qq update >/dev/null && apt-get -qq install -y libportaudio2 >/dev/null

# uv и Python 3.11 (проект требует >=3.11,<3.12; в Colab по умолчанию другая версия).
pip -q install uv
uv python install 3.11
uv sync

# Модели и данные из первоисточников, со сверкой хешей.
uv run python -m qat.assets fetch --profile "$PROFILE"
if [ "$PROFILE" = "kws" ] || [ "$PROFILE" = "all" ]; then
  uv run python -m qat.assets prepare-fsd50k
fi
echo "готово: профиль $PROFILE"
