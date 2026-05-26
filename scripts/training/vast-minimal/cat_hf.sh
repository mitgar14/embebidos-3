#!/bin/bash
# Muestra un archivo del repo de modelos en HF. Uso: (desde la raiz) cat_hf.sh <path_en_repo>
set -euo pipefail
P="${1:?uso: cat_hf.sh <path_en_repo>}"
HF=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '\r\n "')
curl -s -H "Authorization: Bearer $HF" \
  "https://huggingface.co/mitgar14/embebidos-3-models-v1d/resolve/main/$P" | tr -d '\0\r'
