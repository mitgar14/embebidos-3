#!/usr/bin/env bash
# install_wsl2.sh — bootstrap idempotente para el Plan B (RTX 3060 de Nicolas).
#
# Uso (dentro de WSL2 Ubuntu 22.04 de Nicolas):
#   cd /ruta/al/repo/embebidos-3
#   bash scripts/labeling/install/install_wsl2.sh
#
# Lo que hace:
#   1. Verifica systemd activo en WSL2 (sino, da instrucciones para activarlo)
#   2. Verifica CUDA passthrough (nvidia-smi via /usr/lib/wsl/lib)
#   3. Instala miniconda si no esta + crea env embebidos3-label
#   4. Copia el repo a /opt/embebidos3-label
#   5. Genera /etc/embebidos3-label.env con HF_TOKEN (preguntado)
#   6. Instala unit systemd con __USER__ sustituido
#   7. Habilita + arranca servicio
#   8. Imprime URL Tailscale (si esta instalado) o instruye como exponerlo
#
# Re-ejecutable: cada paso comprueba si ya esta hecho.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TARGET_USER="${SUDO_USER:-$USER}"
ENV_NAME="embebidos3-label"

log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[install]\033[0m %s\n' "$*" >&2; }
err() { printf '\033[1;31m[install]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 1. systemd en WSL2 ----------------------------------------------
log "Verificando systemd..."
if ! pidof systemd >/dev/null 2>&1; then
  warn "systemd NO esta activo en esta WSL2. Pasos para activarlo:"
  cat <<'EOF'

  1) En WSL2:  sudo tee /etc/wsl.conf > /dev/null <<EOL
[boot]
systemd=true
EOL
  2) En PowerShell del host Windows:  wsl --shutdown
  3) Reabrir WSL2 (este script se vuelve a correr).

EOF
  exit 1
fi
log "systemd OK (PID $(pidof systemd))"

# ---------- 2. CUDA passthrough ---------------------------------------------
log "Verificando CUDA passthrough..."
NVSMI=""
if command -v nvidia-smi >/dev/null 2>&1; then
  NVSMI="$(command -v nvidia-smi)"
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  NVSMI="/usr/lib/wsl/lib/nvidia-smi"
  warn "nvidia-smi no esta en PATH. Lo usaremos via ruta absoluta."
else
  err "nvidia-smi no encontrado. Driver NVIDIA del host Windows desactualizado?"
fi
"$NVSMI" --query-gpu=name,memory.total --format=csv,noheader || err "nvidia-smi fallo"
log "GPU disponible OK"

# ---------- 3. Miniconda + conda env ----------------------------------------
CONDA_DIR="$HOME/miniconda3"
if [[ ! -d "$CONDA_DIR" ]]; then
  log "Miniconda no encontrado en $CONDA_DIR. Instalando..."
  TMP=$(mktemp -d)
  wget -q -O "$TMP/miniconda.sh" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$TMP/miniconda.sh" -b -p "$CONDA_DIR"
  rm -rf "$TMP"
fi

CONDA_BIN="$CONDA_DIR/bin/conda"
if ! "$CONDA_BIN" env list | grep -q "^$ENV_NAME "; then
  log "Creando conda env '$ENV_NAME' (puede tardar 5-10 min)..."
  "$CONDA_BIN" env create -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml"
else
  log "Conda env '$ENV_NAME' ya existe. Actualizando deps..."
  "$CONDA_BIN" env update -n "$ENV_NAME" -f "$SCRIPT_DIR/environment.yml" --prune
fi

ENV_PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"
"$ENV_PY" -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()}')" \
  || err "PyTorch no detecta CUDA. Revisar driver del host Windows."

# ---------- 4. Copiar repo a /opt/embebidos3-label -------------------------
log "Sincronizando repo a /opt/embebidos3-label (no toca tu copia de trabajo)..."
sudo mkdir -p /opt/embebidos3-label
sudo rsync -a --delete \
  --include='scripts/' \
  --include='scripts/labeling/' \
  --include='scripts/labeling/**' \
  --include='pyproject.toml' \
  --exclude='*' \
  "$REPO_ROOT/" /opt/embebidos3-label/
sudo chown -R "$TARGET_USER:$TARGET_USER" /opt/embebidos3-label

# ---------- 5. Generar /etc/embebidos3-label.env ----------------------------
if [[ ! -f /etc/embebidos3-label.env ]]; then
  log "Generando /etc/embebidos3-label.env (HF_TOKEN requerido)"
  read -r -s -p "HF_TOKEN (input oculto): " HF_TOKEN_VALUE
  echo
  sudo tee /etc/embebidos3-label.env >/dev/null <<EOF
HF_TOKEN=$HF_TOKEN_VALUE
EOF
  sudo chmod 600 /etc/embebidos3-label.env
  sudo chown root:root /etc/embebidos3-label.env
else
  log "/etc/embebidos3-label.env ya existe (no se sobrescribe)"
fi

# ---------- 6. Instalar systemd unit con USER sustituido --------------------
log "Instalando unit systemd..."
sudo sed "s|__USER__|$TARGET_USER|g" \
  "$SCRIPT_DIR/embebidos3-label.service" | \
  sudo tee /etc/systemd/system/embebidos3-label.service >/dev/null

sudo mkdir -p /var/lib/embebidos3-label/jobs
sudo chown -R "$TARGET_USER:$TARGET_USER" /var/lib/embebidos3-label

sudo systemctl daemon-reload
sudo systemctl enable embebidos3-label
sudo systemctl restart embebidos3-label
sleep 2

# ---------- 7. Verificar arranque -------------------------------------------
if systemctl is-active --quiet embebidos3-label; then
  log "embebidos3-label esta ACTIVO"
else
  warn "embebidos3-label no arranco. Logs:"
  sudo journalctl -u embebidos3-label -n 30 --no-pager
  exit 1
fi

# Smoke test
sleep 1
if curl -sf http://127.0.0.1:8765/health >/dev/null; then
  log "/health OK"
  curl -s http://127.0.0.1:8765/system | python3 -m json.tool || true
else
  warn "/health fallo. Revisar logs: sudo journalctl -u embebidos3-label -f"
fi

# ---------- 8. Tailscale serve (opcional, solo si esta instalado) ----------
if command -v tailscale >/dev/null 2>&1; then
  log "Tailscale detectado. Exponiendo en tailnet..."
  sudo tailscale serve --bg --tls-terminated-tcp 8765 || \
    sudo tailscale serve --bg 8765 || \
    warn "tailscale serve fallo (revisar permisos)"
  TS_DOMAIN="$(tailscale status --json 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("Self",{}).get("DNSName","").rstrip("."))' 2>/dev/null)"
  if [[ -n "$TS_DOMAIN" ]]; then
    log "URL del servicio: https://$TS_DOMAIN/health"
  fi
else
  warn "Tailscale no esta instalado en el host Windows."
  warn "Para que mitgar14 acceda al servicio:"
  warn "  1) Instalar Tailscale en Windows host de Nicolas (tailscale.com/download)"
  warn "  2) Login con cuenta compartida o invitar a mitgar14 al tailnet"
  warn "  3) Desde WSL2:  sudo tailscale serve --bg 8765"
fi

log "Listo. Servicio escuchando en localhost:8765"
log "Comandos utiles:"
log "  sudo systemctl status embebidos3-label"
log "  sudo journalctl -u embebidos3-label -f"
log "  curl http://localhost:8765/system"
