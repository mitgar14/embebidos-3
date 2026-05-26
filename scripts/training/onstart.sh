#!/usr/bin/env bash
# =============================================================================
# onstart.sh — Vast.ai container bootstrap para training v1d
# =============================================================================
# Ejecutado UNA vez por --onstart-cmd cuando el container arranca.
#  1) apt deps minimos
#  2) Descarga bootstrap.sh + notebook desde HF (mitgar14/embebidos3-raw-batches/.notebook/)
#  3) Ejecuta bootstrap.sh (instala stack Track B en /opt/venv/trackb + cron watchdog)
#  4) Lanza training nbconvert dentro de tmux 'training' (sobrevive a SSH disconnect)
#
# Logs:
#   /workspace/onstart.log     -- este script
#   /workspace/bootstrap.log   -- bootstrap.sh
#   /workspace/nbconvert.log   -- training
#
# Variables esperadas (vastai create --env):
#   HF_TOKEN, VAST_API_KEY
# =============================================================================

set -ex
# Algunas imagenes de Vast NO traen /workspace al arrancar el onstart: crearlo
# ANTES de redirigir, o el `exec >` falla y (con set -e) mata el script aca.
mkdir -p /workspace
# tee: deja el log en archivo Y en el log del contenedor (visible con `vastai logs`).
exec > >(tee /workspace/onstart.log) 2>&1

echo "=== onstart.sh start $(date -Iseconds) ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends curl ca-certificates tmux

# Verificar env
: "${HF_TOKEN:?HF_TOKEN no inyectado por vastai --env}"

NOTEBOOK_BASE="https://huggingface.co/datasets/mitgar14/embebidos3-raw-batches/resolve/main/.notebook"

mkdir -p /workspace/embebidos-3/notebooks

echo "=== Descargando bootstrap.sh ==="
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
  "$NOTEBOOK_BASE/bootstrap.sh" \
  -o /workspace/embebidos-3/bootstrap.sh
chmod +x /workspace/embebidos-3/bootstrap.sh

echo "=== Descargando train_v1d_vastai.ipynb ==="
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" \
  "$NOTEBOOK_BASE/train_v1d_vastai.ipynb" \
  -o /workspace/embebidos-3/notebooks/train_v1d_vastai.ipynb

echo "=== Ejecutando bootstrap.sh ==="
cd /workspace/embebidos-3
./bootstrap.sh > /workspace/bootstrap.log 2>&1
echo "=== bootstrap.sh OK ==="

# Persist env para shells subsecuentes (debug por SSH)
cat > /etc/profile.d/embebidos3-env.sh <<EOF
export HF_TOKEN='${HF_TOKEN}'
export VAST_API_KEY='${VAST_API_KEY:-}'
export VAST_CONTAINERLABEL='${VAST_CONTAINERLABEL:-}'
EOF
chmod +x /etc/profile.d/embebidos3-env.sh

echo "=== Lanzando training en tmux 'training' ==="
tmux kill-session -t training 2>/dev/null || true
tmux new-session -d -s training \
  "cd /workspace/embebidos-3 && \
   export HF_TOKEN='$HF_TOKEN' && \
   export VAST_CONTAINERLABEL='${VAST_CONTAINERLABEL:-}' && \
   /opt/venv/trackb/bin/jupyter nbconvert \
     --to notebook --execute --inplace \
     --ExecutePreprocessor.timeout=7200 \
     --ExecutePreprocessor.kernel_name=trackb \
     notebooks/train_v1d_vastai.ipynb 2>&1 | tee /workspace/nbconvert.log; \
   echo TRAINING_EXIT=\$? >> /workspace/nbconvert.log"

# Marker file para que verificacion externa sepa que onstart completo
touch /workspace/.onstart_done
echo "=== onstart.sh OK $(date -Iseconds) ==="
