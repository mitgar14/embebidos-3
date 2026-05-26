#!/bin/bash
# Lista los archivos del repo de modelos en HF. Uso: (desde la raiz) check_hf.sh
set -euo pipefail
HF=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '\r\n "')
curl -s -H "Authorization: Bearer $HF" \
  "https://huggingface.co/api/models/mitgar14/embebidos-3-models-v1d/tree/main?recursive=true" \
  | tr -d '\0\r' | python3 scripts/training/vast-minimal/check_hf_tree.py
