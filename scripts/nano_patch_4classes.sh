#!/bin/bash
# Patch del Nano para operar con 4 clases (glass/paper/plastic/cardboard) + repo v1d.
# Se ejecuta EN el Nano (via ssh ... | base64 -d | SUDO_PASS=... bash). Idempotente.
set -e
ROOT=/home/jetson/embebidos-3
C=$ROOT/scripts/server/nano_server_constants.py
N=$ROOT/scripts/builder/nano_correctness.py

# 1. CLASSES 3 -> 4 (server + builder)
sed -i 's/CLASSES = \["glass", "paper", "plastic"\]/CLASSES = ["glass", "paper", "plastic", "cardboard"]/' "$C" "$N"
# 2. COLORS: 4to color (cardboard = marron BGR) en el builder
sed -i 's/COLORS = \[(0, 255, 0), (255, 200, 0), (0, 100, 255)\]/COLORS = [(0, 255, 0), (255, 200, 0), (0, 100, 255), (19, 69, 139)]/' "$N"
# 3. Repo HF v1c -> v1d en secrets.env (requiere sudo)
echo "$SUDO_PASS" | sudo -S -p '' sed -i 's#models-v1c#models-v1d#' /etc/embebidos3/secrets.env

echo "== CLASSES =="
grep -n 'CLASSES =' "$C" "$N"
echo "== COLORS =="
grep -n 'COLORS =' "$N"
echo "== py_compile =="
python3 -m py_compile "$C" "$N" && echo COMPILE_OK || echo COMPILE_FAIL
echo "== repo efectivo =="
echo "$SUDO_PASS" | sudo -S -p '' grep EMBEBIDOS3_HF_REPO /etc/embebidos3/secrets.env
echo "== DONE =="
