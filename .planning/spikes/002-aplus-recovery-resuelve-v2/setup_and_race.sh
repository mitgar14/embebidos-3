#!/usr/bin/env bash
# Spike 002: setup + reproducir race con builder A++ (throwaway).
# Aplica los 3 ajustes obligatorios de ronda 2:
#   #1: fsync explícitos (engine_fd + parent_dir_fd)
#   #2: threat model SIGKILL/OOM (power-loss out of scope)
#   #3: validar contenido de .previous en recovery (lo hace reconcile.py)
#
# Caso de oro: SIGKILL DESPUÉS de los renames del previous, ANTES del staging→active.
# Esa es la ventana V-2 donde el .previous queda VÁLIDO + .ready, y el active VACÍO.

set -euo pipefail

ROOT="${SPIKE_ROOT:-/home/jetson/spike-v2/002}"
ENGINES_DIR="${ROOT}/engines"
ACTIVE_ENGINE="${ENGINES_DIR}/best_fp16.engine"
ACTIVE_META="${ENGINES_DIR}/best_fp16.engine.meta.json"
ACTIVE_READY="${ENGINES_DIR}/best_fp16.engine.ready"
PREV_DIR="${ENGINES_DIR}/.previous"
PREV_ENGINE="${PREV_DIR}/best_fp16.engine.old"
PREV_META="${PREV_DIR}/best_fp16.engine.old.meta.json"
PREV_READY="${PREV_DIR}/best_fp16.engine.old.ready"
STAGING_DIR="${ENGINES_DIR}/.staging"
STAGING_ENGINE="${STAGING_DIR}/best_fp16.engine.new"

fsync_path() {
    python3 -c "
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
" "$1"
}

echo "[SPIKE-002] root: $ROOT"
echo "[SPIKE-002] kernel: $(uname -r)"

# === Setup: estado heredado de build anterior con A++ ===
# Active existe + tiene .ready válido. Staging existe (build nuevo terminado).
# Previous NO existe (será creado por el race).
mkdir -p "$ENGINES_DIR" "$PREV_DIR" "$STAGING_DIR"
rm -rf "$ENGINES_DIR"/* "$PREV_DIR"/* "$STAGING_DIR"/*

echo "[SPIKE-002] setup: active engine + .ready (build anterior con A++)"
dd if=/dev/urandom of="$ACTIVE_ENGINE" bs=1M count=10 status=none
echo '{"sha256":"synthetic_active","build_id":"spike002_prev"}' > "$ACTIVE_META"
ACTIVE_SHA_ORIG=$(sha256sum "$ACTIVE_ENGINE" | awk '{print $1}')
echo "{\"committed_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"engine_sha256\":\"$ACTIVE_SHA_ORIG\"}" \
    > "$ACTIVE_READY"
fsync_path "$ACTIVE_ENGINE"
fsync_path "$ACTIVE_READY"
echo "$ACTIVE_SHA_ORIG" > "${ROOT}/.active_sha_orig.txt"
echo "[SPIKE-002]   active sha256 original: ${ACTIVE_SHA_ORIG:0:16}..."

echo "[SPIKE-002] setup: staging engine (build nuevo terminado)"
dd if=/dev/urandom of="$STAGING_ENGINE" bs=1M count=10 status=none
fsync_path "$STAGING_ENGINE"

# === Builder A++ swap, parando antes del staging→active (caso V-2) ===
echo ""
echo "[SPIKE-002] ejecutando builder A++ swap"
echo "[SPIKE-002]   fsync previo del staging (ajuste #1)"
fsync_path "$STAGING_ENGINE"

echo "[SPIKE-002]   rm -f previous (limpiar viejo)"
rm -f "$PREV_ENGINE" "$PREV_META" "$PREV_READY"

if [[ -f "$ACTIVE_ENGINE" ]]; then
    echo "[SPIKE-002]   mv active engine → previous engine"
    mv "$ACTIVE_ENGINE" "$PREV_ENGINE"
    [[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true

    echo "[SPIKE-002]   mv active .ready → previous .ready (heredando validez)"
    if [[ -f "$ACTIVE_READY" ]]; then
        mv "$ACTIVE_READY" "$PREV_READY"
    fi
    echo "[SPIKE-002]   fsync(previous dir) — ajuste #1"
    fsync_path "$PREV_DIR"
fi

echo "[SPIKE-002]   >>> CRASH SIMULADO — staging→active NO se ejecuta <<<"
# El "mv $STAGING_ENGINE $ACTIVE_ENGINE" + escritura del .ready del active
# NO se ejecutan. Este es el bug V-2 con el agregado de que el previous tiene
# su .ready (válido, heredado del active anterior).

echo ""
echo "[SPIKE-002] estado post-race:"
ls -la "$ENGINES_DIR/" 2>&1 | head -10
echo "  .previous/:"
ls -la "$PREV_DIR/" 2>&1 | tail -n +2 | head -10
echo "  .staging/:"
ls -la "$STAGING_DIR/" 2>&1 | tail -n +2 | head -10
