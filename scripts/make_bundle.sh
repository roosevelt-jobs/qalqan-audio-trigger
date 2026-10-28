#!/usr/bin/env bash
# Архив кода для загрузки на Google Диск: только закоммиченные файлы (git archive).
# Данных, весов моделей, событий и личных записей в нём нет — их Colab скачивает сам.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then
  echo "ВНИМАНИЕ: есть незакоммиченные изменения — в архив они не попадут" >&2
fi
mkdir -p dist
OUT="dist/qalqan_$(git rev-parse --short HEAD).zip"
git archive --format=zip --prefix=qalqan/ -o "$OUT" HEAD
echo "$OUT ($(du -h "$OUT" | cut -f1))"
