#!/usr/bin/env bash
# Spike 003: validar 3 escenarios donde A++ recovery debe comportarse correctamente.
#  A. Estado consistente (no_op): active + .ready coherentes → reconcile NO toca nada
#  B. Degraded sin .ready: active missing + previous engine sin .ready → degraded
#  C. Degraded sin engine: active missing + previous .ready sin engine → degraded
#
# Reutiliza reconcile.py del spike 002.

set -uo pipefail

RECONCILE="/home/jetson/spike-v2/002/reconcile.py"
errors_total=0

# Helper: setup directorio limpio
fresh_root() {
    local root=$1
    rm -rf "$root"
    mkdir -p "$root/engines/.previous" "$root/engines/.staging"
}

fsync_path() {
    python3 -c "
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try: os.fsync(fd)
finally: os.close(fd)
" "$1"
}

# ========================================
# Escenario A: estado consistente (no_op)
# ========================================
ROOT_A="/home/jetson/spike-v2/003/A_consistent"
echo ""
echo "========================================="
echo "[SPIKE-003-A] estado consistente → no_op"
echo "========================================="
fresh_root "$ROOT_A"
dd if=/dev/urandom of="$ROOT_A/engines/best_fp16.engine" bs=1M count=10 status=none
SHA_A_PRE=$(sha256sum "$ROOT_A/engines/best_fp16.engine" | awk '{print $1}')
SIZE_A_PRE=$(stat -c %s "$ROOT_A/engines/best_fp16.engine")
echo "{\"committed_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"engine_sha256\":\"$SHA_A_PRE\"}" \
    > "$ROOT_A/engines/best_fp16.engine.ready"
fsync_path "$ROOT_A/engines/best_fp16.engine"
fsync_path "$ROOT_A/engines/best_fp16.engine.ready"

echo "[SPIKE-003-A] estado pre-reconcile: active sha=${SHA_A_PRE:0:16}... size=$SIZE_A_PRE"
RESULT_A=$(python3 "$RECONCILE" "$ROOT_A")
EXIT_A=$?
echo "[SPIKE-003-A] reconcile retornó (exit $EXIT_A):"
echo "$RESULT_A"

# Asserts A
errors_A=0
if echo "$RESULT_A" | grep -q '"action": "no_op"'; then
    echo "[ASSERT-A] PASS: action=no_op"
else
    echo "[ASSERT-A] FAIL: action != no_op"
    errors_A=$((errors_A+1))
fi
SHA_A_POST=$(sha256sum "$ROOT_A/engines/best_fp16.engine" | awk '{print $1}')
SIZE_A_POST=$(stat -c %s "$ROOT_A/engines/best_fp16.engine")
if [[ "$SHA_A_PRE" == "$SHA_A_POST" && "$SIZE_A_PRE" == "$SIZE_A_POST" ]]; then
    echo "[ASSERT-A] PASS: active engine NO modificado (sha + size matches)"
else
    echo "[ASSERT-A] FAIL: active engine modificado por reconcile (false positive)"
    errors_A=$((errors_A+1))
fi
if [[ -f "$ROOT_A/engines/best_fp16.engine.ready" ]]; then
    echo "[ASSERT-A] PASS: .ready intacto"
else
    echo "[ASSERT-A] FAIL: .ready borrado por reconcile (false positive)"
    errors_A=$((errors_A+1))
fi
echo "[SPIKE-003-A] errors: $errors_A"
errors_total=$((errors_total+errors_A))

# ============================================================
# Escenario B: degraded — active missing + previous SIN .ready
# ============================================================
ROOT_B="/home/jetson/spike-v2/003/B_degraded_no_ready"
echo ""
echo "============================================================="
echo "[SPIKE-003-B] degraded sin .ready → degraded (no auto-promote)"
echo "============================================================="
fresh_root "$ROOT_B"
# Active missing, previous engine SIN .ready (caso: SIGKILL durante swap del .ready)
dd if=/dev/urandom of="$ROOT_B/engines/.previous/best_fp16.engine.old" bs=1M count=10 status=none

echo "[SPIKE-003-B] estado pre-reconcile: active missing, previous engine sin .ready"
RESULT_B=$(python3 "$RECONCILE" "$ROOT_B")
EXIT_B=$?
echo "[SPIKE-003-B] reconcile retornó (exit $EXIT_B):"
echo "$RESULT_B"

errors_B=0
if echo "$RESULT_B" | grep -q '"action": "degraded"'; then
    echo "[ASSERT-B] PASS: action=degraded"
else
    echo "[ASSERT-B] FAIL: action != degraded (debería NO auto-promover sin .ready)"
    errors_B=$((errors_B+1))
fi
if [[ -f "$ROOT_B/engines/.previous/best_fp16.engine.old" ]] && [[ ! -f "$ROOT_B/engines/best_fp16.engine" ]]; then
    echo "[ASSERT-B] PASS: previous engine NO movido (correcto, no debe auto-promover sin .ready)"
else
    echo "[ASSERT-B] FAIL: reconcile alteró estado pese a previous inválido"
    errors_B=$((errors_B+1))
fi
if [[ $EXIT_B -ne 0 ]]; then
    echo "[ASSERT-B] PASS: exit code != 0 (señala degraded al caller)"
else
    echo "[ASSERT-B] FAIL: exit 0 cuando debería ser != 0"
    errors_B=$((errors_B+1))
fi
echo "[SPIKE-003-B] errors: $errors_B"
errors_total=$((errors_total+errors_B))

# ============================================================
# Escenario C: degraded — previous .ready existe SIN engine
# ============================================================
ROOT_C="/home/jetson/spike-v2/003/C_degraded_no_engine"
echo ""
echo "==============================================================="
echo "[SPIKE-003-C] degraded sin engine en previous → degraded"
echo "==============================================================="
fresh_root "$ROOT_C"
# Active missing, previous tiene .ready pero NO tiene engine
# (caso teórico: alguien borró el engine pero dejó el .ready)
echo "{\"committed_at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" \
    > "$ROOT_C/engines/.previous/best_fp16.engine.old.ready"

echo "[SPIKE-003-C] estado pre-reconcile: active missing, previous .ready sin engine"
RESULT_C=$(python3 "$RECONCILE" "$ROOT_C")
EXIT_C=$?
echo "[SPIKE-003-C] reconcile retornó (exit $EXIT_C):"
echo "$RESULT_C"

errors_C=0
if echo "$RESULT_C" | grep -q '"action": "degraded"'; then
    echo "[ASSERT-C] PASS: action=degraded (validó CONTENIDO de previous, no solo .ready)"
else
    echo "[ASSERT-C] FAIL: action != degraded — ajuste #3 NO está validando contenido"
    errors_C=$((errors_C+1))
fi
if [[ ! -f "$ROOT_C/engines/best_fp16.engine" ]]; then
    echo "[ASSERT-C] PASS: no se inventó active engine"
else
    echo "[ASSERT-C] FAIL: reconcile creó active engine de la nada"
    errors_C=$((errors_C+1))
fi
echo "[SPIKE-003-C] errors: $errors_C"
errors_total=$((errors_total+errors_C))

echo ""
echo "========================================="
echo "[SPIKE-003] RESUMEN GLOBAL"
echo "========================================="
echo "[SPIKE-003] errors totales: $errors_total"
if [[ $errors_total -eq 0 ]]; then
    echo "[SPIKE-003] A++ COMPORTAMIENTO CORRECTO en los 3 escenarios ✓"
    exit 0
else
    echo "[SPIKE-003] FAIL: $errors_total errores en validación"
    exit 1
fi
