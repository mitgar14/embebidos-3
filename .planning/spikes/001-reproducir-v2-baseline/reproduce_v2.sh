#!/usr/bin/env bash
# Spike 001: reproducir V-2 baseline (sin fix)
# Reproduce el race condition del swap actual de nano_build_engine.sh:191-197
# Variante A: simulamos el SIGKILL no ejecutando el segundo mv.
# Fidelidad: el estado post-crash es idéntico al SIGKILL real (mismo conjunto
# de archivos en disco). La diferencia entre "no se ejecutó" y "se mató durante"
# es irrelevante para validar el RECOVERY (lo único que importa es el estado final).

set -euo pipefail

ROOT="${SPIKE_ROOT:-/home/jetson/spike-v2/001}"
ENGINES_DIR="${ROOT}/engines"
ACTIVE_ENGINE="${ENGINES_DIR}/best_fp16.engine"
ACTIVE_META="${ENGINES_DIR}/best_fp16.engine.meta.json"
PREV_DIR="${ENGINES_DIR}/.previous"
PREV_ENGINE="${PREV_DIR}/best_fp16.engine.old"
PREV_META="${PREV_DIR}/best_fp16.engine.old.meta.json"
STAGING_DIR="${ENGINES_DIR}/.staging"
STAGING_ENGINE="${STAGING_DIR}/best_fp16.engine.new"

echo "[SPIKE-001] root: $ROOT"
echo "[SPIKE-001] kernel: $(uname -r)"
echo "[SPIKE-001] filesystem: $(stat -fc %T "$(dirname "$ROOT")" 2>/dev/null || echo unknown)"

# Setup limpio
mkdir -p "$ENGINES_DIR" "$PREV_DIR" "$STAGING_DIR"
rm -f "$ACTIVE_ENGINE" "$ACTIVE_META" "$PREV_ENGINE" "$PREV_META" "$STAGING_ENGINE"

echo "[SPIKE-001] creando engines sintéticos (~10 MB random)"
dd if=/dev/urandom of="$ACTIVE_ENGINE" bs=1M count=10 status=none
dd if=/dev/urandom of="$STAGING_ENGINE" bs=1M count=10 status=none
echo '{"sha256":"synthetic","build_id":"spike001"}' > "$ACTIVE_META"

ACTIVE_SHA_PRE=$(sha256sum "$ACTIVE_ENGINE" | awk '{print $1}')
STAGING_SHA=$(sha256sum "$STAGING_ENGINE" | awk '{print $1}')
echo "[SPIKE-001] active sha256 pre-race: ${ACTIVE_SHA_PRE:0:16}..."
echo "[SPIKE-001] staging sha256: ${STAGING_SHA:0:16}..."

# Reproducir el bloque swap de nano_build_engine.sh:191-197
# Ejecutar hasta el primer mv (incluido), interrumpir antes del segundo
echo ""
echo "[SPIKE-001] ejecutando swap (mismo flujo que nano_build_engine.sh:191-197)"
echo "[SPIKE-001]   rm -f \"\$PREV_ENGINE\" \"\$PREV_META\""
rm -f "$PREV_ENGINE" "$PREV_META"
echo "[SPIKE-001]   mv \"\$ACTIVE_ENGINE\" \"\$PREV_ENGINE\"           # punto 1 — completado"
mv "$ACTIVE_ENGINE" "$PREV_ENGINE"
[[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true
echo "[SPIKE-001]   >>> CRASH SIMULADO — segundo mv NO se ejecuta <<<"
# El "mv $STAGING_ENGINE $ACTIVE_ENGINE" del código original NO se ejecuta acá

# Persistir sha pre-race para validación cruzada con spike 002
echo "$ACTIVE_SHA_PRE" > "${ROOT}/.active_sha_pre_race.txt"
echo "$STAGING_SHA" > "${ROOT}/.staging_sha.txt"

echo ""
echo "[SPIKE-001] estado post-race:"
ls -la "$ENGINES_DIR/" 2>&1 | head -20
echo "  .previous/:"
ls -la "$PREV_DIR/" 2>&1 | tail -n +2 | head -10
echo "  .staging/:"
ls -la "$STAGING_DIR/" 2>&1 | tail -n +2 | head -10
