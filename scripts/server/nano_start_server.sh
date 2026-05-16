#!/usr/bin/env bash
# Inicia el server FastAPI/WS en foreground (para systemd Type=simple).
# El proceso queda en foreground; systemd captura stdout/stderr y maneja restart.
set -euo pipefail

ROOT=/home/jetson/embebidos-3
ENGINE="${ROOT}/engines/best_fp16.engine"

# Env CUDA
export PATH=/home/jetson/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONIOENCODING=utf-8
export ENGINE_PATH="${ENGINE}"

# Foreground: NO nohup, NO `&`. systemd captura stdout/stderr y maneja PID.
exec python3 -c "import uvicorn; uvicorn.run('nano_server:app', host='0.0.0.0', port=8000, workers=1, log_level='info')"
