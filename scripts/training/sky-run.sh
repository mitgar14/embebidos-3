#!/usr/bin/env bash
# =============================================================================
# sky-run.sh — Lanza o re-ejecuta el entrenamiento v1d vía SkyPilot (Vast.ai)
# =============================================================================
# Pensado para correrse DENTRO de WSL (donde vive SkyPilot), evitando el quoting
# fragil del boundary MINGW->WSL: aca usamos comillas simples y finales LF, asi
# que la extraccion del token desde el .env es robusta (el intento inline desde
# el Bash tool mutilaba el patron "^HF_TOKEN=" y mandaba el secret vacio).
#
# Modos:
#   check   solo extrae el token y reporta su longitud (no toca Vast)
#   exec    re-ejecuta el bloque `run` en el cluster existente (setup ya hecho)
#   launch  provisiona un cluster nuevo (setup + run completos)
#
# Uso (desde WSL, no hace falta cd):
#   bash /mnt/c/Users/mitgar14/Documentos/embebidos-3/scripts/training/sky-run.sh check
#   bash .../sky-run.sh exec
#   bash .../sky-run.sh launch
# =============================================================================
set -euo pipefail

REPO=/mnt/c/Users/mitgar14/Documentos/embebidos-3
ENVF="$REPO/.env"
YAML="$REPO/train-v1d.sky.yaml"
CLUSTER=v1d
MODE="${1:-exec}"

export PATH="$HOME/.local/bin:$PATH"

# Extraccion robusta: patron en comillas SIMPLES, primera coincidencia, sin CR/LF.
HFTOK="$(grep '^HF_TOKEN=' "$ENVF" | head -1 | cut -d= -f2- | tr -d '\r\n')"
if [ "${#HFTOK}" -lt 10 ]; then
  echo "ERROR: HF_TOKEN vacio o no hallado en $ENVF (len=${#HFTOK})" >&2
  exit 1
fi
echo "HF_TOKEN ok: len=${#HFTOK} prefijo=${HFTOK:0:3}"

case "$MODE" in
  check)
    echo "check ok (no se toco Vast)"
    ;;
  launch)
    cd "$REPO"
    exec sky launch -y -c "$CLUSTER" "$YAML" --secret HF_TOKEN="$HFTOK"
    ;;
  exec)
    cd "$REPO"
    exec sky exec "$CLUSTER" "$YAML" --secret HF_TOKEN="$HFTOK"
    ;;
  fixtorch)
    # Reinstala torch del index cu128 (kernels sm_120 para Blackwell; tambien
    # cubre Ada/Hopper). Vast entrego una RTX PRO 6000 Blackwell bajo la etiqueta
    # "RTX_4090", y torch 2.4.1+cu124 no tiene kernels para sm_120. Termina con un
    # sanity check de CUDA (alloc + matmul) en la GPU real.
    sky exec "$CLUSTER" 'export PATH="$HOME/.local/bin:$PATH"; set -e; uv pip install --python /opt/venv/trackb/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0; uv pip install --python /opt/venv/trackb/bin/python --force-reinstall "numpy<2.0"; /opt/venv/trackb/bin/python -c "import torch; print(\"VER\", torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(\"ARCH\", torch.cuda.get_arch_list()); x=torch.rand(16,16,device=\"cuda\"); print(\"MATMUL_OK\", float((x@x).sum()))"'
    ;;
  *)
    echo "modo desconocido: $MODE (usa check|exec|launch)" >&2
    exit 2
    ;;
esac
