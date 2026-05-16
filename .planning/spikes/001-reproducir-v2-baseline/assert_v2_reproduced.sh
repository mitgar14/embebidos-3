#!/usr/bin/env bash
# Spike 001: assert que V-2 fue reproducido correctamente
# Verifica que el estado del filesystem matches el bug V-2 exacto.

set -uo pipefail

ROOT="${SPIKE_ROOT:-/home/jetson/spike-v2/001}"
ACTIVE_ENGINE="${ROOT}/engines/best_fp16.engine"
PREV_ENGINE="${ROOT}/engines/.previous/best_fp16.engine.old"
STAGING_ENGINE="${ROOT}/engines/.staging/best_fp16.engine.new"

errors=0
echo "[ASSERT] verificando estado post-V-2 en $ROOT"

# Assert 1: active engine NO debe existir (eso es el bug)
if [[ -f "$ACTIVE_ENGINE" ]]; then
    echo "[ASSERT] FAIL: active engine existe (no debería)"
    errors=$((errors+1))
else
    echo "[ASSERT] PASS: active engine MISSING (V-2 confirmado)"
fi

# Assert 2: previous engine SÍ debe existir (rescatable)
if [[ -f "$PREV_ENGINE" ]]; then
    PREV_SIZE=$(stat -c %s "$PREV_ENGINE")
    echo "[ASSERT] PASS: previous engine EXISTS (${PREV_SIZE} bytes)"
else
    echo "[ASSERT] FAIL: previous engine NO existe"
    errors=$((errors+1))
fi

# Assert 3: staging engine SÍ debe existir (build válido)
if [[ -f "$STAGING_ENGINE" ]]; then
    STAGING_SIZE=$(stat -c %s "$STAGING_ENGINE")
    echo "[ASSERT] PASS: staging engine EXISTS (${STAGING_SIZE} bytes)"
else
    echo "[ASSERT] FAIL: staging engine NO existe"
    errors=$((errors+1))
fi

# Assert 4: cross-check sha del previous vs el sha pre-race del active
PRE_RACE_SHA_FILE="${ROOT}/.active_sha_pre_race.txt"
if [[ -f "$PRE_RACE_SHA_FILE" && -f "$PREV_ENGINE" ]]; then
    EXPECTED=$(cat "$PRE_RACE_SHA_FILE")
    ACTUAL=$(sha256sum "$PREV_ENGINE" | awk '{print $1}')
    if [[ "$EXPECTED" == "$ACTUAL" ]]; then
        echo "[ASSERT] PASS: previous sha matches active pre-race (${ACTUAL:0:16}...)"
    else
        echo "[ASSERT] FAIL: sha mismatch — expected ${EXPECTED:0:16}..., got ${ACTUAL:0:16}..."
        errors=$((errors+1))
    fi
fi

echo ""
if [[ $errors -eq 0 ]]; then
    echo "[ASSERT] V-2 REPRODUCIDO ✓ (4/4 asserts)"
    exit 0
else
    echo "[ASSERT] $errors errores — V-2 NO reproducido correctamente"
    exit 1
fi
