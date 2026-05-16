#!/usr/bin/env bash
# nano_build_engine.sh <job_id>
# Orquestador del build TRT FP16. Invocado por la unit systemd embebidos3-builder@<jobid>.service.
# Asume sudoers granular ya configurado.
set -euo pipefail

BUILD_START_UNIX=$(date +%s)

JOB_ID="${1:-}"
if [[ -z "$JOB_ID" ]]; then
    echo "[BUILD] FATAL: falta job_id" >&2
    exit 2
fi

ROOT=/home/jetson/embebidos-3
LOG_FILE="${ROOT}/logs/jobs/${JOB_ID}.log"
TEGRA_LOG="${ROOT}/logs/jobs/${JOB_ID}.tegrastats.log"
STAGING_DIR="${ROOT}/engines/.staging"
PREV_DIR="${ROOT}/engines/.previous"
STAGING_ENGINE="${STAGING_DIR}/best_fp16.engine.new"
ACTIVE_ENGINE="${ROOT}/engines/best_fp16.engine"
ACTIVE_META="${ROOT}/engines/best_fp16.engine.meta.json"
PREV_ENGINE="${PREV_DIR}/best_fp16.engine.old"
PREV_META="${PREV_DIR}/best_fp16.engine.old.meta.json"
WORKSPACE="${EMBEBIDOS3_TRTEXEC_WORKSPACE:-512}"
HF_TOKEN="${HF_TOKEN:-}"

mkdir -p "${ROOT}/logs/jobs" "${STAGING_DIR}" "${PREV_DIR}" "${ROOT}/onnx"

# Lock exclusivo: fd se libera automáticamente al morir el proceso
exec {LOCK_FD}<>/run/embebidos3/builder.lock
flock -n "$LOCK_FD" || { echo "[BUILD] otro builder en curso, abort" >&2; exit 1; }
echo "$$" > /run/embebidos3/builder.lock

JS() { python3 "${ROOT}/scripts/builder_state.py" "${JOB_ID}" "$@"; }

cleanup() {
    local code=$?
    [[ -n "${TEGRA_PID:-}" ]] && kill "$TEGRA_PID" 2>/dev/null || true
    rm -f "$STAGING_ENGINE"
    # restore Nano (idempotente)
    sudo systemctl start lightdm.service 2>/dev/null || true
    sudo sysctl vm.swappiness=60 >/dev/null 2>&1 || true
    if [[ $code -ne 0 ]]; then
        echo "[BUILD] FAILED exit=$code" >&2
        JS finalize --phase failed --exit-code "$code" 2>/dev/null || true
        systemctl is-active --quiet embebidos3-server.service \
            || sudo systemctl start embebidos3-server.service
    fi
}
trap cleanup EXIT

JS phase --name acquired_lock --pct 5

# 1. download manifest
JS phase --name downloaded_manifest --pct 8
HF_TOKEN="$HF_TOKEN" python3 "${ROOT}/scripts/hf_rest.py" download \
    manifests/manifest.json /tmp/manifest.json

# 2. parse rev + esperado_sha
HF_REV=$(python3 -c "import json; m=json.load(open('/tmp/manifest.json')); print(m.get('recovery',{}).get('hf_revision') or m.get('hf_revision') or 'main')")
EXPECTED_SHA=$(python3 -c "import json; print(json.load(open('/tmp/manifest.json'))['artifacts']['best_onnx']['sha256'])")
HF_COMMIT_DATE=$(python3 -c "import json; print(json.load(open('/tmp/manifest.json')).get('hf_commit_date','') or '')")

# 3. download onnx
JS phase --name downloaded_onnx --pct 12
HF_TOKEN="$HF_TOKEN" python3 "${ROOT}/scripts/hf_rest.py" download \
    exports/best.onnx "${ROOT}/onnx/best.onnx.tmp" --revision "$HF_REV"

# 4. verify SHA
JS phase --name verified_sha --pct 15
ACTUAL_SHA=$(sha256sum "${ROOT}/onnx/best.onnx.tmp" | awk '{print $1}')
if [[ "$EXPECTED_SHA" != "$ACTUAL_SHA" ]]; then
    echo "[BUILD] SHA mismatch: expected=$EXPECTED_SHA actual=$ACTUAL_SHA" >&2
    rm -f "${ROOT}/onnx/best.onnx.tmp"
    exit 2
fi
mv "${ROOT}/onnx/best.onnx.tmp" "${ROOT}/onnx/best.onnx"

# 5. stop server
JS phase --name stopped_server --pct 18
sudo systemctl stop embebidos3-server.service
sleep 3

# 6. prep Nano
JS phase --name prep_nano --pct 22
sudo systemctl stop lightdm.service 2>/dev/null || true
sudo systemctl disable nvzramconfig 2>/dev/null || true
sudo swapoff -a
if [[ ! -f /mnt/swap.img ]]; then
    sudo fallocate -l 8G /mnt/swap.img
    sudo chmod 600 /mnt/swap.img
    sudo mkswap /mnt/swap.img
fi
sudo swapon /mnt/swap.img
sudo sysctl vm.swappiness=100 >/dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

# 7. trtexec
JS phase --name trtexec_started --pct 25
tegrastats --interval 2000 > "$TEGRA_LOG" 2>&1 &
TEGRA_PID=$!

set +e
timeout --kill-after=30s 40m \
    /usr/src/tensorrt/bin/trtexec \
        --onnx="${ROOT}/onnx/best.onnx" \
        --saveEngine="$STAGING_ENGINE" \
        --fp16 \
        --workspace="$WORKSPACE" \
        --buildOnly \
        --verbose \
        2>&1 | tee "$LOG_FILE" | python3 "${ROOT}/scripts/parse_trtexec_progress.py" "$JOB_ID"
TRTEXEC_EXIT="${PIPESTATUS[0]}"
set -e

kill "$TEGRA_PID" 2>/dev/null || true; unset TEGRA_PID

case "$TRTEXEC_EXIT" in
    0)   JS phase --name trtexec_built --pct 75 ;;
    124) echo "[BUILD] timeout 40m" >&2; exit 124 ;;
    137) echo "[BUILD] OOM-killed (exit 137)" >&2; exit 137 ;;
    *)   echo "[BUILD] trtexec error $TRTEXEC_EXIT" >&2; exit "$TRTEXEC_EXIT" ;;
esac

# sanity check
SIZE=$(stat -c %s "$STAGING_ENGINE")
if [[ "$SIZE" -lt 1000000 ]]; then
    echo "[BUILD] engine sospechosamente chico: $SIZE bytes" >&2
    exit 1
fi

# 8. validate
JS phase --name validating --pct 80
VAL_JSON="${ROOT}/logs/jobs/${JOB_ID}.validation.json"
python3 "${ROOT}/scripts/validate_engine.py" "$STAGING_ENGINE" > "$VAL_JSON" || \
    { echo "[BUILD] validación falló" >&2; cat "$VAL_JSON" >&2 || true; exit 3; }
JS phase --name validated --pct 85

# 9. backup viejo a HF (si existe .previous)
if [[ -f "$PREV_ENGINE" ]]; then
    JS phase --name backing_up_previous --pct 88
    TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    OLD_SHA=$(sha256sum "$PREV_ENGINE" | awk '{print $1}' | head -c 8)
    HF_TOKEN="$HF_TOKEN" python3 "${ROOT}/scripts/hf_rest.py" upload \
        "$PREV_ENGINE" \
        "engines-archive/${TIMESTAMP}__${OLD_SHA}/best_fp16.engine" \
        --message "embebidos3 backup engine ${OLD_SHA}" \
        || { echo "[BUILD] backup HF falló, abort cleanup" >&2; exit 4; }
    if [[ -f "$PREV_META" ]]; then
        HF_TOKEN="$HF_TOKEN" python3 "${ROOT}/scripts/hf_rest.py" upload \
            "$PREV_META" \
            "engines-archive/${TIMESTAMP}__${OLD_SHA}/meta.json" \
            --message "embebidos3 backup meta ${OLD_SHA}" || true
    fi
fi
JS phase --name backed_up_previous --pct 92

# 10. swap atómico
rm -f "$PREV_ENGINE" "$PREV_META"
if [[ -f "$ACTIVE_ENGINE" ]]; then
    mv "$ACTIVE_ENGINE" "$PREV_ENGINE"
    [[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true
fi
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"
BUILD_DUR=$(( $(date +%s) - BUILD_START_UNIX ))
python3 "${ROOT}/scripts/write_engine_meta.py" \
    "$ACTIVE_ENGINE" "$HF_REV" "$EXPECTED_SHA" "$WORKSPACE" \
    --build-duration-s "$BUILD_DUR" \
    --validation-json "$VAL_JSON" \
    ${HF_COMMIT_DATE:+--hf-commit-date "$HF_COMMIT_DATE"}
JS phase --name swapped --pct 95

# 11. restore Nano
JS phase --name restoring_nano --pct 97
sudo systemctl start lightdm.service 2>/dev/null || true
sudo sysctl vm.swappiness=60 >/dev/null

# 12. start server
JS phase --name starting_server --pct 99
sudo systemctl start embebidos3-server.service

JS finalize --phase done --exit-code 0
