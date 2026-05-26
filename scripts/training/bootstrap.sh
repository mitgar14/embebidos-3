#!/usr/bin/env bash
# =============================================================================
# bootstrap.sh — Setup Track B (YOLOv8n -> ONNX -> TensorRT FP16) en Vast.ai
# =============================================================================
# Spec:       HANDOFF-track-b-2026-05-13.md §5.1 Etapa 2
# Container:  vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310
# Decisiones: D2 (stack), D3 (Vast.ai), D5 (uv), D6 (venv trackb),
#             D8 (HF Hub), D9 (tmux), D11 (cron watchdog), D17, D18,
#             D30 (torch cu124 pin), D33 (/etc/vast-env para cron),
#             D34 (pin torch==2.4.1 version, no --extra-index-url torch)
# Idempotente: sí
# Gotchas:    G-VAST-02 (tmux antes de tmux new), G-VAST-03 (uv no pip),
#             G-VAST-04 (LF endings — usar `dos2unix` si copiaste desde Win),
#             G-VAST-05 (cu130 wheel rompe con driver <12.9 → usar cu124),
#             G-VAST-06 (cron sin /etc/vast-env no ve VAST_API_KEY)
# Auth:       requiere env vars VAST_API_KEY (para watchdog), HF_TOKEN y
#             ROBOFLOW_API_KEY (para los notebooks, no para el bootstrap).
#             Pasarlas con `vastai create instance ... --env '-e VAST_API_KEY=... -e HF_TOKEN=...'`
# =============================================================================

set -euo pipefail

# ---- Variables --------------------------------------------------------------
VENV=/opt/venv/trackb
WORKDIR=/workspace/embebidos-3
PYTHON_VER=3.10

# ---- 1) apt deps (G-VAST-02) ------------------------------------------------
echo "[1/8] apt deps"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  tmux git curl wget unzip ca-certificates \
  ffmpeg libgl1-mesa-glx libglib2.0-0 \
  cron >/dev/null

# ---- 2) uv (D5) -------------------------------------------------------------
echo "[2/8] uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
uv --version

# ---- 3) venv Track B (D6) ---------------------------------------------------
echo "[3/8] venv $VENV"
mkdir -p "$(dirname "$VENV")"
if [ ! -d "$VENV" ]; then
  uv venv "$VENV" --python "$PYTHON_VER"
fi

# ---- 4) Stack Track B (D2, D30) — versiones congeladas ----------------------
# D30 (2026-05-14): pinear torch al wheel cu124 explícito. Driver de Vast.ai
# (mayo 2026) está en la franja CUDA 12.4-12.8; cu130 requiere >=12.9 y falla.
# D34 (2026-05-20): además de --index-url cu124, PINEAR versiones de torch y
# torchvision. Sin pin, uv resuelve a torch 2.12 (más nuevo en pypi via
# --extra-index-url) que viene como +cu130 y no carga con driver 12.4.
echo "[4/8] stack Track B"

# 4a) PyTorch primero, pineado al wheel CUDA explícito (cu128).
# D35 (2026-05-25): Vast.ai entrega GPUs Blackwell (sm_120, p.ej. RTX PRO 6000)
# incluso bajo etiquetas como "RTX_4090", con driver 580+. El wheel cu124
# (torch 2.4.1) NO trae kernels sm_120 y revienta con "no kernel image
# available". cu128 (torch 2.7.0) cubre sm_120 y tambien Ada, Hopper y Ampere,
# así que es la opción universal y segura sin importar qué GPU caiga.
# NO usar extra-index aquí: pypi tiene torch más reciente y uv prefiere version
# sobre index. Resto del stack instala desde pypi en 4b.
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.0 torchvision==0.22.0

# 4b) Resto del stack — Ultralytics ya no toca torch (lo encuentra instalado).
uv pip install --python "$VENV/bin/python" \
  "ultralytics>=8.4.46,<8.5" \
  "numpy<2.0" \
  "onnx" \
  "onnxslim>=0.1.34" \
  "onnxruntime-gpu" \
  "huggingface_hub" \
  "wandb" \
  "roboflow>=1.1.27" \
  "jupyter" \
  "jupyterlab" \
  "ipykernel"

# 4c) Refuerzo numpy<2 — el resolver de uv prefiere wheels nuevos y a veces
# deja numpy 2.x si una dep transitiva no pinea hard. Forzar al final.
uv pip install --python "$VENV/bin/python" --force-reinstall "numpy<2.0"

# ---- 5) Registrar ipykernel "Track B (YOLOv8)" (D6) -------------------------
echo "[5/8] ipykernel"
"$VENV/bin/python" -m ipykernel install --user \
  --name trackb --display-name "Track B (YOLOv8)"

# ---- 6-7) vastai CLI + cron watchdog: omitibles con SKIP_WATCHDOG -----------
# Bajo un orquestador externo (p.ej. SkyPilot autodown) el teardown lo maneja
# el orquestador, no este watchdog. SKIP_WATCHDOG=1 salta ambas secciones.
if [ "${SKIP_WATCHDOG:-0}" != "1" ]; then

# ---- 6) vastai CLI para watchdog (D11) --------------------------------------
echo "[6/8] vastai CLI"
uv tool install vastai >/dev/null 2>&1 || uv tool upgrade vastai >/dev/null 2>&1 || true

# ---- 7) Cron watchdog auto-destroy (D11, D33) -------------------------------
# D33 (2026-05-14): cron NO hereda env del shell del operador. Escribimos
# /etc/vast-env (root:600) con VAST_API_KEY + VAST_CONTAINERLABEL + PATH y
# el cron line hace `. /etc/vast-env` antes de invocar el watchdog. Sin esto
# el watchdog corría pero `vastai destroy` fallaba por API key ausente.
echo "[7/8] cron watchdog"
mkdir -p /opt/scripts

cat > /etc/vast-env <<EOF
VAST_API_KEY=${VAST_API_KEY:-}
VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-}
HOME=${HOME}
PATH=${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin
EOF
chmod 600 /etc/vast-env

cat > /opt/scripts/check-gpu-idle.sh <<'WATCHDOG'
#!/usr/bin/env bash
# Cada 5 min mide util GPU. <5% durante 6 muestreos seguidos => destroy.
# Requiere /etc/vast-env con VAST_API_KEY + VAST_CONTAINERLABEL.
set -euo pipefail
COUNTER=/tmp/gpu_idle_count
LOG=/var/log/gpu-idle-watchdog.log
THRESHOLD=5
WINDOW=6

# Validar env antes de hacer cualquier cosa.
if [ -z "${VAST_API_KEY:-}" ] || [ -z "${VAST_CONTAINERLABEL:-}" ]; then
  echo "$(date -Iseconds) skip: VAST_API_KEY o VAST_CONTAINERLABEL ausentes" >> "$LOG"
  exit 0
fi

util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
       | head -1 | tr -d ' ')
[ -z "$util" ] && exit 0

if [ "$util" -lt "$THRESHOLD" ]; then
  count=$(cat "$COUNTER" 2>/dev/null || echo 0)
  count=$((count + 1))
  echo "$count" > "$COUNTER"
  if [ "$count" -ge "$WINDOW" ]; then
    INSTANCE_ID="${VAST_CONTAINERLABEL#C.}"
    echo "$(date -Iseconds) destroying idle instance $INSTANCE_ID (util=$util)" >> "$LOG"
    vastai destroy instance "$INSTANCE_ID" >> "$LOG" 2>&1 || true
  fi
else
  echo 0 > "$COUNTER"
fi
WATCHDOG
chmod +x /opt/scripts/check-gpu-idle.sh

# Cron cada 5 min — idempotente. `. /etc/vast-env` carga API key + label antes
# de invocar el script. Sin esto, cron corre con env mínimo y `vastai destroy`
# falla silenciosamente por falta de credenciales.
( crontab -l 2>/dev/null | grep -v 'check-gpu-idle.sh' || true
  echo "*/5 * * * * . /etc/vast-env && /opt/scripts/check-gpu-idle.sh"
) | crontab -

service cron start >/dev/null 2>&1 || /etc/init.d/cron start >/dev/null 2>&1 || true

else
  echo "[6-7/8] vastai CLI + cron watchdog OMITIDOS (SKIP_WATCHDOG=1; teardown vía orquestador)"
fi

# ---- 8) Workdir + JupyterLab en tmux (D9, D18) ------------------------------
echo "[8/8] workdir + JupyterLab"
mkdir -p "$WORKDIR/runs" "$WORKDIR/logs" "$WORKDIR/models" "$WORKDIR/notebooks" "$WORKDIR/datasets"

# SKIP_JUPYTERLAB=1 omite el lab interactivo (al correr nbconvert headless bajo
# un orquestador). Los mkdir de arriba se conservan.
if [ "${SKIP_JUPYTERLAB:-0}" != "1" ] && ! tmux has-session -t jupyter 2>/dev/null; then
  tmux new-session -d -s jupyter \
    "cd $WORKDIR && $VENV/bin/jupyter lab \
       --no-browser --port=8888 --ip=0.0.0.0 --allow-root \
       --ServerApp.token='' --ServerApp.password='' \
       --ServerApp.root_dir=$WORKDIR 2>&1 | tee -a $WORKDIR/logs/jupyter.log"
fi

# ---- Resumen ----------------------------------------------------------------
cat <<EOF

==============================================================================
  Bootstrap Track B completo
==============================================================================
  venv         : $VENV ($("$VENV/bin/python" --version 2>&1))
  kernel       : Track B (YOLOv8)
  JupyterLab   : http://0.0.0.0:8888  (sin token, tmux session 'jupyter')
                 attach: tmux attach -t jupyter   |   detach: Ctrl+B D
  watchdog     : /opt/scripts/check-gpu-idle.sh
                 cron */5 min — destroy tras 30 min idle (<5% util)
                 log: /var/log/gpu-idle-watchdog.log
  workdir      : $WORKDIR
  vastai CLI   : $(command -v vastai 2>/dev/null || echo '(no en PATH; usar /root/.local/bin/vastai)')

  Próximos pasos (desde el operador Win11):
    1) ssh -L 8888:localhost:8888 root@INSTANCE_IP   # túnel JupyterLab
    2) abrir http://localhost:8888 en el navegador
    3) crear notebook con kernel "Track B (YOLOv8)"
==============================================================================
EOF
