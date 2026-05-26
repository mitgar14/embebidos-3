#!/bin/bash
# Crea la instancia Vast minima. Uso: (desde la raiz del repo) create.sh <offer_id>
# Lee HF_TOKEN de .env y lo inyecta como env var de la instancia (no se imprime).
set -euo pipefail
OFFER="${1:?uso: create.sh <offer_id>}"
HF=$(grep -m1 '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '\r\n "')
[ -n "$HF" ] || { echo "ERROR: HF_TOKEN vacio en .env"; exit 1; }
echo "[create] offer=$OFFER token_len=${#HF}"
vastai create instance "$OFFER" \
  --image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  --disk 30 \
  --onstart scripts/training/vast-minimal/onstart.sh \
  --env "-e HF_TOKEN=$HF" \
  --raw
