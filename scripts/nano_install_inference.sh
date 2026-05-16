#!/usr/bin/env bash
# =============================================================================
# nano_install_inference.sh — Setup inferencia in-process + FastAPI server en
# Jetson Nano B01 (JetPack 4.6.1, L4T R32.7.1, Python 3.6.9, TRT 8.2.1.8).
#
# Investigación: investigaciones/2026-05-14-deploy-streaming-nano-webcam-web.md
# Plan A: pycuda 2019.1.2 (jaybdub NVIDIA moderator + jkjung-avt comunidad).
# Fix raíz: variables CUDA env ANTES de pip install + flags include/lib.
# =============================================================================
set -euo pipefail

# ---- 0) Env CUDA (raíz del fallo previo) -----------------------------------
export PATH=/usr/local/cuda/bin:${HOME}/.local/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_INC_DIR=/usr/local/cuda/include
export CUDA_ROOT=/usr/local/cuda

echo "[0/6] env CUDA configurado"
nvcc --version | tail -3
which python3 && python3 --version

# ---- 1) pip3 bootstrap (Python 3.6 EOL) ------------------------------------
echo "[1/6] pip3 bootstrap"
if ! command -v pip3 >/dev/null 2>&1; then
    cd /tmp
    wget -q https://bootstrap.pypa.io/pip/3.6/get-pip.py -O get-pip-36.py
    python3 get-pip-36.py --user --quiet
fi
pip3 --version

# ---- 2) Build deps para pycuda 2019.1.2 ------------------------------------
echo "[2/6] verificar build deps (boost-python, build-essential)"
DEPS_MISSING=()
dpkg -l 2>/dev/null | grep -q "libboost-python-dev" || DEPS_MISSING+=("libboost-python-dev")
dpkg -l 2>/dev/null | grep -q "libboost-thread-dev" || DEPS_MISSING+=("libboost-thread-dev")
dpkg -l 2>/dev/null | grep -q "python3-dev" || DEPS_MISSING+=("python3-dev")
dpkg -l 2>/dev/null | grep -q "build-essential" || DEPS_MISSING+=("build-essential")
if [ ${#DEPS_MISSING[@]} -ne 0 ]; then
    echo "  faltan: ${DEPS_MISSING[*]}"
    echo "  ejecutá manualmente: sudo apt-get install -y ${DEPS_MISSING[*]}"
    echo "  (este script requiere sudo passwordless para auto-instalar; saltando)"
fi

# ---- 3) pycuda 2019.1.2 (plan A, jaybdub NVIDIA) ---------------------------
echo "[3/6] pycuda 2019.1.2"
if python3 -c "import pycuda; print(pycuda.VERSION_TEXT)" 2>/dev/null; then
    echo "  ya instalado"
else
    # Usar global-option para apuntar al cuda include/lib explícito.
    # Ref: jaybdub forum, ratificado por docenas 2021-2024.
    pip3 install --user --no-cache-dir \
        --global-option=build_ext \
        --global-option="-I/usr/local/cuda/include" \
        --global-option="-L/usr/local/cuda/lib64" \
        "pycuda==2019.1.2" 2>&1 | tail -10
fi
python3 -c "import pycuda; print('pycuda', pycuda.VERSION_TEXT)"

# ---- 4) numpy + FastAPI stack (compat py3.6) -------------------------------
echo "[4/6] FastAPI server stack (versiones compat py3.6)"
# numpy: el system 1.13.3 es muy viejo; necesitamos >=1.17 pero <1.20 (último py3.6).
# Si numpy 1.19.5 vuelve a dar SIGILL, mantenemos el system 1.13.3 (TRT funciona).
pip3 install --user --no-cache-dir \
    "fastapi==0.65.3" \
    "uvicorn[standard]==0.13.4" \
    "websockets==9.1" \
    "starlette==0.14.2" \
    "pydantic<2"  # FastAPI 0.65 requiere pydantic v1

python3 -c "
import fastapi, uvicorn, starlette, websockets
print('fastapi', fastapi.__version__)
print('uvicorn', uvicorn.__version__)
print('starlette', starlette.__version__)
print('websockets', websockets.__version__)
"

# ---- 5) Dependencias adicionales del pipeline (Fase D) ---------------------
echo "[5/6] dependencias adicionales para pipeline"
python3 -m pip install --user --no-cache-dir "requests>=2.25,<3"
# jq via apt si no está
command -v jq >/dev/null 2>&1 || sudo apt-get install -y jq

python3 -c "import requests; print('requests', requests.__version__)"
jq --version

# ---- 6) Verificación final -------------------------------------------------
echo "[6/6] verificación stack completo"
python3 -c "
import tensorrt as trt; print('tensorrt', trt.__version__)
import pycuda.driver as cuda; print('pycuda driver OK')
import cv2; print('opencv', cv2.__version__)
import fastapi, uvicorn; print('fastapi/uvicorn OK')
"

cat <<'EOF'

==============================================================================
  Setup inferencia in-process completo
==============================================================================
  pycuda       : 2019.1.2 (con CUDA env exportado)
  TensorRT     : 8.2.1.8 (system)
  FastAPI      : 0.65.3 + uvicorn 0.13.4 + websockets 9.1
  OpenCV       : 4.1.1 (system, NMSBoxes V0)

  Próximo paso:
    cd /home/jetson/embebidos-3/scripts
    ENGINE_PATH=/home/jetson/embebidos-3/engines/best_fp16.engine \
      /home/jetson/.local/bin/uvicorn nano_server:app \
        --host 0.0.0.0 --port 8000 --workers 1

  Health check:
    curl http://localhost:8000/health

  Logs server: redirigir a /home/jetson/embebidos-3/logs/server.log

==============================================================================
EOF
