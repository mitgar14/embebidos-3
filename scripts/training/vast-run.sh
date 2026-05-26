#!/usr/bin/env bash
# =============================================================================
# vast-run.sh — Lanzador Vast DIRECTO (sin SkyPilot) con tope de precio
# =============================================================================
# SkyPilot no controla el precio en Vast: su catalogo decia "RTX_3090 $0.25"
# pero Vast aprovisionaba una RTX PRO 6000 Blackwell a $1.478/h (3 veces). Aca
# usamos la CLI de Vast: buscamos ofertas BAJO un tope (dph_total), elegimos una
# por ID y la creamos exactamente, viendo el precio real ANTES. torch cu128
# (bootstrap) corre en cualquier GPU, asi que filtramos por PRECIO, no por modelo.
#
# Teardown doble: el notebook se autodestruye al terminar (vastai destroy) y el
# cron watchdog del bootstrap destruye si la GPU queda idle ~30 min.
#
# Correr en WSL (donde vive uv/uvx):
#   bash vast-run.sh sync             sube bootstrap.sh + notebook a HF .notebook/
#   bash vast-run.sh offers [MAXDPH]  lista ofertas baratas (default 0.20)
#   bash vast-run.sh launch OFFER_ID  crea la instancia con onstart
# =============================================================================
set -euo pipefail

REPO=/mnt/c/Users/mitgar14/Documentos/embebidos-3
ENVF="$REPO/.env"
HF_REPO=mitgar14/embebidos3-raw-batches
MODEL_REPO=mitgar14/embebidos-3-models-v1d
IMAGE="vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310"
DISK=40
MODE="${1:-offers}"

export PATH="$HOME/.local/bin:$PATH"

# Extraccion robusta desde el .env (comillas simples, sin CR/LF).
val() { (grep "^$1=" "$ENVF" 2>/dev/null || true) | head -1 | cut -d= -f2- | tr -d '\r\n'; }
HF_TOKEN="$(val HF_TOKEN)"
VAST_API_KEY="$(val VAST_API_KEY)"
ROBOFLOW_API_KEY="$(val ROBOFLOW_API_KEY)"
[ "${#HF_TOKEN}" -lt 10 ] && { echo "ERROR: HF_TOKEN ausente en $ENVF"; exit 1; }
# VAST_API_KEY no esta en el .env: cae al config de la CLI (~/.config/vastai/vast_api_key).
if [ -z "$VAST_API_KEY" ] && [ -f "$HOME/.config/vastai/vast_api_key" ]; then
  VAST_API_KEY="$(tr -d '\r\n' < "$HOME/.config/vastai/vast_api_key")"
fi

case "$MODE" in
  sync)
    # Normaliza a LF (G-VAST-04) antes de subir: si el .sh viaja con CRLF, el
    # bash del contenedor revienta con errores de \r aleatorios.
    sed 's/\r$//' "$REPO/scripts/training/bootstrap.sh" > /tmp/bootstrap.sh
    echo "Subiendo bootstrap.sh (cu128) + notebook a $HF_REPO/.notebook/ ..."
    HF_TOKEN="$HF_TOKEN" HF_REPO="$HF_REPO" NB="$REPO/notebooks/train_v1d_vastai.ipynb" \
      uvx --from huggingface_hub python -c '
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_REPO"]
api.upload_file(path_or_fileobj="/tmp/bootstrap.sh", path_in_repo=".notebook/bootstrap.sh", repo_id=repo, repo_type="dataset")
api.upload_file(path_or_fileobj=os.environ["NB"], path_in_repo=".notebook/train_v1d_vastai.ipynb", repo_id=repo, repo_type="dataset")
print("sync OK")
'
    ;;
  offers)
    MAXDPH="${2:-0.20}"
    echo "Ofertas: 1 GPU, dph<$MAXDPH, driver CUDA>=12.8 (cu128-ready), VRAM>=12, red>=100Mbps:"
    uvx vastai search offers "rentable=true num_gpus=1 dph_total<$MAXDPH cuda_max_good>=12.8 gpu_ram>=12 inet_down>=100 disk_space>=40" -o dph_total --raw \
      | python3 -c "import json,sys
d=json.load(sys.stdin)
print(f'{\"OFFER_ID\":>9}  {\"DPH\":>6}  {\"GPU\":<18} {\"CUDA\":>5} {\"VRAM\":>5} {\"REGION\"}')
for o in d[:8]:
    print(f'{o[\"id\"]:>9}  {round(o[\"dph_total\"],4):>6}  {o[\"gpu_name\"][:18]:<18} {o[\"cuda_max_good\"]:>5} {round(o.get(\"gpu_ram\",0)):>5} {o.get(\"geolocation\",\"?\")}')"
    ;;
  launch)
    OFFER_ID="${2:?Falta OFFER_ID}"
    sed 's/\r$//' "$REPO/scripts/training/onstart.sh" > /tmp/onstart.sh
    echo "Creando instancia en oferta $OFFER_ID (disco ${DISK}GB)..."
    uvx vastai create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK" --ssh \
      --env "-e HF_TOKEN=$HF_TOKEN -e VAST_API_KEY=$VAST_API_KEY -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -e SKIP_JUPYTERLAB=1" \
      --onstart /tmp/onstart.sh
    ;;
  cheapest)
    MAXDPH="${2:-0.20}"
    sed 's/\r$//' "$REPO/scripts/training/onstart.sh" > /tmp/onstart.sh
    # Busca fresco y elige la oferta mas barata con arch cu128-segura (Ampere+;
    # excluye Pascal/Volta/Turing viejas que torch 2.7 cu128 ya no compila).
    read -r OFFER_ID DPH GPU < <(uvx vastai search offers "rentable=true num_gpus=1 dph_total<$MAXDPH cuda_max_good>=12.8 gpu_ram>=12 inet_down>=100 disk_space>=40" -o dph_total --raw | python3 -c '
import json, sys, re
offers = json.load(sys.stdin)
SAFE = re.compile(r"RTX *30|RTX *40|RTX *50|A4000|A5000|A6000|A100|H100|H200|L4|L40|RTX_?6000|RTX_?5000", re.I)
ok = [o for o in offers if SAFE.search(o.get("gpu_name",""))]
if ok:
    o = ok[0]
    print(o["id"], round(o["dph_total"], 4), o["gpu_name"].replace(" ", "_"))
')
    [ -z "${OFFER_ID:-}" ] && { echo "No hubo oferta segura bajo \$$MAXDPH/h"; exit 1; }
    echo "Oferta elegida: id=$OFFER_ID  precio=\$$DPH/h  gpu=$GPU"
    uvx vastai create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK" --ssh \
      --env "-e HF_TOKEN=$HF_TOKEN -e VAST_API_KEY=$VAST_API_KEY -e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -e SKIP_JUPYTERLAB=1" \
      --onstart /tmp/onstart.sh
    ;;
  peek)
    # Entra por SSH a la instancia y muestra util real de GPU + cola del log de
    # nbconvert (que vive dentro de tmux, no en el log del contenedor).
    URL=$(uvx vastai ssh-url "${2:?Falta id}" 2>/dev/null | tr -d '\r\n')
    HP=${URL#ssh://root@}; HOST=${HP%:*}; PORT=${HP##*:}
    echo "ssh root@$HOST:$PORT"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -p "$PORT" "root@$HOST" \
      'echo "UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)"; echo "--- nbconvert.log (cola) ---"; tail -25 /workspace/nbconvert.log 2>&1' 2>&1
    ;;
  hf)
    # Chequeo HF de un disparo (sin loop): codigos HTTP de cada artefacto + la
    # ultima linea del heartbeat (epoch actual). Corre DENTRO del script (LF,
    # comillas correctas) para evitar el destrozo de variables al pasar un for
    # inline por el doble borde Bash-tool -> MINGW -> wsl.exe -> Ubuntu.
    BASE="https://huggingface.co/$MODEL_REPO/resolve/main"
    echo "HF check  repo=$MODEL_REPO  token_len=${#HF_TOKEN}"
    for f in runs/heartbeat.jsonl manifests/eval_summary.json manifests/manifest.json runs/results.csv models/best.onnx; do
      code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" "$BASE/$f")
      printf "  %-30s %s\n" "$f" "$code"
    done
    last=$(curl -s -H "Authorization: Bearer $HF_TOKEN" "$BASE/runs/heartbeat.jsonl" | tail -1)
    [ -n "$last" ] && echo "ultimo heartbeat: $last"
    ;;
  ensure)
    # "Ponelo a entrenar y ya": inyecta mi clave SSH en la instancia CORRIENDO
    # (vastai attach ssh; no reinicia), entra, y si el nbconvert NO esta vivo lo
    # relanza en tmux. Todo dentro del script (LF, heredoc 'REMOTE' con comillas
    # simples) para no destrozar comillas en el doble borde.
    ID="${2:?Falta id}"
    [ -f "$HOME/.ssh/id_ed25519" ] || ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -q
    PUB="$(cat "$HOME/.ssh/id_ed25519.pub")"
    echo "Inyectando clave SSH en instancia $ID ..."
    uvx vastai attach ssh "$ID" "$PUB" 2>&1 | grep -vi deprecat || true
    URL="$(uvx vastai ssh-url "$ID" 2>/dev/null | tr -d '\r\n')"
    HP="${URL#ssh://root@}"; HOST="${HP%:*}"; PORT="${HP##*:}"
    echo "ssh root@$HOST:$PORT (esperando a que la clave propague)"
    ok=0
    for t in 1 2 3 4 5; do
      sleep 6
      if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=12 -p "$PORT" "root@$HOST" 'bash -s' <<'REMOTE'
echo "=== tmux ==="; tmux ls 2>&1 || echo "(sin sesiones tmux)"
echo "=== procesos nbconvert ==="; pgrep -af "jupyter nbconvert" || echo "(ninguno)"
echo "=== nbconvert.log (cola) ==="; tail -n 45 /workspace/nbconvert.log 2>&1 || echo "(sin log)"
echo "=== GPU ==="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1
if pgrep -f "jupyter nbconvert.*train_v1d" >/dev/null; then
  echo "STATUS=ENTRENANDO_O_EN_SETUP (nbconvert vivo)"
else
  echo "STATUS=NBCONVERT_MUERTO -> RELANZANDO"
  . /etc/profile.d/embebidos3-env.sh 2>/dev/null || true
  tmux kill-session -t training 2>/dev/null || true
  tmux new-session -d -s training "cd /workspace/embebidos-3 && /opt/venv/trackb/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=trackb notebooks/train_v1d_vastai.ipynb 2>&1 | tee /workspace/nbconvert.log; echo TRAINING_EXIT=\$? >> /workspace/nbconvert.log"
  sleep 4
  echo "--- relanzado ---"; tmux ls 2>&1; tail -n 15 /workspace/nbconvert.log 2>&1
fi
REMOTE
      then ok=1; break; fi
      echo "  intento $t: la clave aun no propaga, reintento..."
    done
    [ "$ok" = 1 ] || echo "ERROR: no pude entrar por SSH tras el attach (revisar manualmente)"
    ;;
  fixtrain)
    # Arregla el segfault del DataLoader (PyTorch + /dev/shm 64MB de Docker) y
    # relanza. Doble arreglo para certeza: (1) remonta /dev/shm grande si el
    # contenedor lo permite; (2) parchea el notebook a workers=0 (sin shared
    # memory => segfault imposible; con 163 imgs el costo de velocidad es nimio).
    # Requiere que la clave SSH ya este inyectada (correr `ensure` antes).
    ID="${2:?Falta id}"
    URL="$(uvx vastai ssh-url "$ID" 2>/dev/null | tr -d '\r\n')"
    HP="${URL#ssh://root@}"; HOST="${HP%:*}"; PORT="${HP##*:}"
    echo "fixtrain en root@$HOST:$PORT"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" "root@$HOST" 'bash -s' <<'REMOTE' || true
set +e
echo "=== matando tmux training condenado ==="
tmux kill-session -t training 2>/dev/null || true
pkill -f "jupyter nbconvert.*train_v1d" 2>/dev/null || true
sleep 2

echo "=== remontando /dev/shm a 8g (si el contenedor lo permite) ==="
mount -o remount,size=8g /dev/shm 2>&1 || echo "(remount fallo; workers=0 cubre el caso)"
df -h /dev/shm 2>&1 | tail -1

echo "=== parcheando notebook -> workers=0 (idempotente) ==="
python3 -c '
import json
p="/workspace/embebidos-3/notebooks/train_v1d_vastai.ipynb"
nb=json.load(open(p))
patched=False
for c in nb["cells"]:
    if c.get("cell_type")!="code": continue
    src=c["source"]
    joined="".join(src)
    if "model.train(" in joined and "workers=" not in joined:
        out=[]
        for line in src:
            out.append(line)
            if "model.train(" in line:
                out.append("        workers=0,\n")
        c["source"]=out
        patched=True
json.dump(nb, open(p,"w"))
print("patched=",patched)
'

echo "=== relanzando training en tmux ==="
. /etc/profile.d/embebidos3-env.sh 2>/dev/null || true
tmux new-session -d -s training "cd /workspace/embebidos-3 && /opt/venv/trackb/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 --ExecutePreprocessor.kernel_name=trackb notebooks/train_v1d_vastai.ipynb 2>&1 | tee /workspace/nbconvert.log; echo TRAINING_EXIT=\$? >> /workspace/nbconvert.log"
sleep 5
echo "=== tmux ==="; tmux ls 2>&1
echo "=== nbconvert.log (cola) ==="; tail -n 12 /workspace/nbconvert.log 2>&1
REMOTE
    ;;
  gpucheck)
    # Test de sanidad CUDA: ¿el backward basico crashea en esta GPU? Si imprime
    # BACKWARD_OK la GPU esta sana y el segfault es de YOLO/datos; si crashea, la
    # GPU/driver de esta instancia barata esta flaky => destruir y relanzar.
    ID="${2:?Falta id}"
    URL="$(uvx vastai ssh-url "$ID" 2>/dev/null | tr -d '\r\n')"
    HP="${URL#ssh://root@}"; HOST="${HP%:*}"; PORT="${HP##*:}"
    echo "gpucheck en root@$HOST:$PORT"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$PORT" "root@$HOST" 'bash -s' <<'REMOTE' || true
echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
echo "=== test backward CUDA ==="
/opt/venv/trackb/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cap', torch.cuda.get_device_capability()); x=torch.randn(2000,2000,device='cuda',requires_grad=True); (x*x).sum().backward(); torch.cuda.synchronize(); print('BACKWARD_OK')" 2>&1
echo "exit=$?"
REMOTE
    ;;
  wait)
    # Sondea HF por los artefactos que publica el notebook (CommitScheduler cada
    # ~10 min). heartbeat=200 => entrenando (paso la carga en GPU, sin Blackwell
    # roto); manifest=200 => terminado OK. Pensado para correr en background.
    BASE="https://huggingface.co/$MODEL_REPO/resolve/main"
    echo "waiter: token_len=${#HF_TOKEN} repo=$MODEL_REPO"
    for i in $(seq 1 40); do
      hb=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" "$BASE/runs/heartbeat.jsonl" || echo "ERR")
      ev=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" "$BASE/manifests/eval_summary.json" || echo "ERR")
      mf=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" "$BASE/manifests/manifest.json" || echo "ERR")
      echo "[$i $(date +%H:%M:%S)] heartbeat=$hb eval=$ev manifest=$mf"
      [ "$mf" = "200" ] && { echo "=== DONE: manifest.json presente, entrenamiento completo ==="; break; }
      sleep 60
    done
    echo "waiter fin"
    ;;
  *)
    echo "modo desconocido: $MODE (usa sync|offers|launch|cheapest|wait)" >&2; exit 2 ;;
esac
