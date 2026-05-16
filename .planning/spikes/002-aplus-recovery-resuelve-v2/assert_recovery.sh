#!/usr/bin/env bash
# Spike 002: assert post-recovery
# Verifica que A++ recovery dejó el sistema en estado healthy:
#  - active engine existe (auto-promovido desde previous)
#  - active engine.ready existe
#  - sha256 del active matches el sha256 del active ORIGINAL pre-race (integridad)
#  - .previous quedó vacío (engine y .ready movidos)

set -uo pipefail

ROOT="${SPIKE_ROOT:-/home/jetson/spike-v2/002}"
ACTIVE_ENGINE="${ROOT}/engines/best_fp16.engine"
ACTIVE_READY="${ROOT}/engines/best_fp16.engine.ready"
PREV_ENGINE="${ROOT}/engines/.previous/best_fp16.engine.old"
PREV_READY="${ROOT}/engines/.previous/best_fp16.engine.old.ready"
ORIG_SHA_FILE="${ROOT}/.active_sha_orig.txt"

errors=0
echo "[ASSERT] verificando estado post-recovery en $ROOT"

# Assert 1: active engine EXISTE (recuperado)
if [[ -f "$ACTIVE_ENGINE" ]]; then
    SIZE=$(stat -c %s "$ACTIVE_ENGINE")
    echo "[ASSERT] PASS: active engine EXISTS (${SIZE} bytes)"
else
    echo "[ASSERT] FAIL: active engine NO existe (recovery falló)"
    errors=$((errors+1))
fi

# Assert 2: active .ready EXISTE (commit marker)
if [[ -f "$ACTIVE_READY" ]]; then
    echo "[ASSERT] PASS: active .ready EXISTS"
else
    echo "[ASSERT] FAIL: active .ready NO existe"
    errors=$((errors+1))
fi

# Assert 3: integridad sha256 (active matches el original pre-race)
if [[ -f "$ORIG_SHA_FILE" && -f "$ACTIVE_ENGINE" ]]; then
    EXPECTED=$(cat "$ORIG_SHA_FILE")
    ACTUAL=$(sha256sum "$ACTIVE_ENGINE" | awk '{print $1}')
    if [[ "$EXPECTED" == "$ACTUAL" ]]; then
        echo "[ASSERT] PASS: integridad sha256 (${ACTUAL:0:16}...)"
    else
        echo "[ASSERT] FAIL: sha256 mismatch — expected ${EXPECTED:0:16}, got ${ACTUAL:0:16}"
        errors=$((errors+1))
    fi
fi

# Assert 4: .previous quedó vacío (engine y .ready movidos al active)
if [[ ! -f "$PREV_ENGINE" ]]; then
    echo "[ASSERT] PASS: previous engine MOVED (ya no en .previous/)"
else
    echo "[ASSERT] FAIL: previous engine sigue en .previous/ (debería haberse movido)"
    errors=$((errors+1))
fi
if [[ ! -f "$PREV_READY" ]]; then
    echo "[ASSERT] PASS: previous .ready MOVED"
else
    echo "[ASSERT] FAIL: previous .ready sigue en .previous/"
    errors=$((errors+1))
fi

echo ""
if [[ $errors -eq 0 ]]; then
    echo "[ASSERT] A++ RECOVERY VALIDADO ✓ (5/5 asserts)"
    exit 0
else
    echo "[ASSERT] $errors errores — A++ recovery NO funcionó como esperado"
    exit 1
fi
