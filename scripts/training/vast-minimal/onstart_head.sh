#!/bin/bash
# onstart minimo Vast.ai (generado: NO editar onstart.sh a mano; editar train.py + regenerar con gen.sh)
mkdir -p /workspace
exec > >(tee -a /workspace/onstart.log) 2>&1
set -uo pipefail
export PYTHONUNBUFFERED=1
cd /workspace
echo "[onstart] start $(date -u)"
cat > /workspace/train.py <<'PYEOF'
