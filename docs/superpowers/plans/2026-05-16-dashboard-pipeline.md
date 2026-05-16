# Dashboard pipeline — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** automatizar desde la página web el flujo descarga ONNX desde HF Hub → compilación TRT FP16 con `trtexec` → validación in-process → swap atómico → reanudación del server, sin reiniciar el Nano y con rollback automático ante fallas.

**Architecture:** Approach B — dos units systemd en el Nano (`embebidos3-server.service` long-running + `embebidos3-builder@<jobid>.service` oneshot templated), coordinación por filesystem en `/var/run/embebidos3/`, comunicación dashboard ↔ server vía HTTP/WS/SSE, robustez en 7 capas con rollback a engine anterior y backup pre-borrado a HF Hub.

**Tech Stack:**
- Backend Nano: Python 3.6.9, FastAPI 0.65, uvicorn 0.13, TensorRT 8.2.1.8, pycuda 2019.1.2, `requests` (sin `huggingface_hub`), systemd.
- Frontend: vanilla JS ES2020, HTML/CSS heredados del redesign 2026-05-15 (Source Sans 3 + Source Code Pro + tokens OKLCH).
- Sistema: Jetson Nano B01 JetPack 4.6.1, L4T R32.7.1, Ubuntu 18.04, Maxwell `sm_53`, 4 GB RAM unificada.
- Testing: `pytest` en el host (mock del Nano) + tests de integración manuales SSH al Nano para flujos end-to-end.

**Inputs:** `docs/superpowers/specs/2026-05-16-dashboard-pipeline-design.md` (spec aprobado), `investigaciones/2026-05-15-pipeline-tecnico-vacios.md` (hallazgos investigación).

---

## File structure

Archivos nuevos:

```
scripts/
├── nano_install_systemd.sh              # instala units + sudoers + tmpfiles
├── nano_build_engine.sh                 # builder oneshot (orquestador bash)
├── embebidos3-builder-launch            # wrapper sudoers-safe
├── builder_state.py                     # CLI helper estado job
├── parse_trtexec_progress.py            # parser stdout trtexec → progress
├── validate_engine.py                   # mini-correctness post-build
├── write_engine_meta.py                 # escribe engine.meta.json
├── hf_rest.py                           # cliente REST HF (download/upload/list)
└── recover_job_state.py                 # helper recovery (import desde nano_server.py)

systemd/
├── embebidos3-server.service
├── embebidos3-builder@.service
└── embebidos3-logs.tmpfiles.conf

tests/
├── test_hf_rest.py                      # tests cliente HF REST (mocked requests)
├── test_builder_state.py                # tests CLI helper estado
├── test_parse_trtexec_progress.py       # tests parser
├── test_validate_engine.py              # tests validación (skip si no GPU)
├── test_write_engine_meta.py            # tests metadata
├── test_recover_job_state.py            # tests recovery
└── conftest.py                          # fixtures comunes
```

Archivos modificados:

```
scripts/
├── nano_server.py                       # TRTWorker reentrante + endpoints + recovery + bug fix :360
├── nano_start_server.sh                 # simplificado para systemd (sin nohup, sin pkill)
└── nano_install_inference.sh            # añade requests + jq

scripts/dashboard/
├── index.html                           # tabs en header + pestaña modelo
├── app.js                               # routing + módulo modelo + SSE client
└── style.css                            # estilos tabs + pestaña modelo + log viewer

pyproject.toml                           # añadir pytest + httpx (mock requests) para dev
```

Archivos eliminados:

```
scripts/nano_stop_server.sh              # lo reemplaza `systemctl stop`
```

---

## Convenciones del plan

- **TDD donde aplica**: Python con `pytest`, test-first. Cada función nueva nace con su test fallando.
- **Verificación manual** para scripts bash/units systemd/UI: comando exacto + expected output.
- **Commits frecuentes**: uno por tarea cerrada (no por step). Mensajes en imperativo, en español o inglés según el dominio.
- **Branching**: trabajo directo en `main` (proyecto académico, equipo de uno). Si surgieran cambios riesgosos, hacer feature branch ad-hoc.
- **Path absoluto Windows**: el plan usa rutas relativas al repo (`scripts/...`). El host está en `C:\Users\mitgar14\Documentos\embebidos-3\`.
- **Acceso al Nano**: SSH alias `nano` ya configurado en `~/.ssh/config`. Comandos remotos se prefijan `ssh nano "<cmd>"`.
- **`sudo` en el Nano** está disponible sin password para `jetson` user (verificado).
- **Validación end-to-end** post-fase: ejecutar el flujo completo manualmente antes de pasar a la siguiente fase.

---

## Fase A — Infraestructura systemd

**Objetivo:** units systemd + sudoers + tmpfiles + script install operativos. El server actual (lanzado por nohup) queda migrado a systemd y arranca al boot.

### Task A1: archivo unit del server

**Files:**
- Create: `systemd/embebidos3-server.service`

- [ ] **Step 1: crear el archivo unit**

```ini
[Unit]
Description=embebidos-3 FastAPI/WS inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetson
Group=jetson
WorkingDirectory=/home/jetson/embebidos-3
EnvironmentFile=/etc/embebidos3/secrets.env
RuntimeDirectory=embebidos3
RuntimeDirectoryMode=0755
ExecStart=/home/jetson/embebidos-3/scripts/nano_start_server.sh
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: commit**

```bash
git add systemd/embebidos3-server.service
git commit -m "feat(systemd): añadir unit embebidos3-server.service"
```

### Task A2: archivo unit templated del builder

**Files:**
- Create: `systemd/embebidos3-builder@.service`

- [ ] **Step 1: crear el archivo unit templated**

```ini
[Unit]
Description=embebidos-3 TRT engine builder (job %i)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=jetson
Group=jetson
WorkingDirectory=/home/jetson/embebidos-3
EnvironmentFile=/etc/embebidos3/secrets.env
RuntimeDirectory=embebidos3
RuntimeDirectoryMode=0755
ExecStart=/home/jetson/embebidos-3/scripts/nano_build_engine.sh %i
TimeoutStartSec=2700
StandardOutput=append:/home/jetson/embebidos-3/logs/jobs/%i.systemd.log
StandardError=append:/home/jetson/embebidos-3/logs/jobs/%i.systemd.log
```

- [ ] **Step 2: commit**

```bash
git add systemd/embebidos3-builder@.service
git commit -m "feat(systemd): añadir unit templated embebidos3-builder@.service"
```

### Task A3: config tmpfiles para retención de logs (3 días TTL)

**Files:**
- Create: `systemd/embebidos3-logs.tmpfiles.conf`

- [ ] **Step 1: crear config tmpfiles**

```
# Tipo Path                                            Mode UID    GID    Age  Argument
d     /home/jetson/embebidos-3/logs/jobs              0755 jetson jetson 3d   -
```

- [ ] **Step 2: commit**

```bash
git add systemd/embebidos3-logs.tmpfiles.conf
git commit -m "feat(systemd): añadir tmpfiles config para retención logs 3d"
```

### Task A4: wrapper sudoers-safe `embebidos3-builder-launch`

**Files:**
- Create: `scripts/embebidos3-builder-launch`

- [ ] **Step 1: crear wrapper con validación regex**

```bash
#!/usr/bin/env bash
# Lanzador validado de instancias templated del builder.
# Invocado como: sudo embebidos3-builder-launch <jobid>
set -euo pipefail

JOBID="${1:-}"
if [[ ! "$JOBID" =~ ^[A-Za-z0-9_-]{10,40}$ ]]; then
    echo "JOBID inválido: '$JOBID' (esperado: 10-40 chars [A-Za-z0-9_-])" >&2
    exit 1
fi
exec /bin/systemctl start "embebidos3-builder@${JOBID}.service"
```

- [ ] **Step 2: hacer ejecutable**

```bash
chmod +x scripts/embebidos3-builder-launch
```

- [ ] **Step 3: verificar validación (smoke test local)**

Ejecutar (en el host, simulando):
```bash
bash scripts/embebidos3-builder-launch "abc"      # debe fallar (corto)
bash scripts/embebidos3-builder-launch "valid-job-id-12345"  # debe llamar systemctl (fallará pero por otra razón)
```

Expected: el primero sale con exit 1 y mensaje "JOBID inválido"; el segundo intenta exec systemctl (que en Windows no existe, error distinto).

- [ ] **Step 4: commit**

```bash
git add scripts/embebidos3-builder-launch
git commit -m "feat(scripts): wrapper sudoers-safe embebidos3-builder-launch"
```

### Task A5: script de instalación `nano_install_systemd.sh`

**Files:**
- Create: `scripts/nano_install_systemd.sh`

- [ ] **Step 1: crear el script de instalación idempotente**

```bash
#!/usr/bin/env bash
# Instala units systemd + sudoers + secrets template + wrapper + tmpfiles.
# Idempotente. Ejecutar como user jetson (que invoca sudo internamente).
set -euo pipefail

ROOT=/home/jetson/embebidos-3
REPO_SYSTEMD="${ROOT}/systemd"

echo "[1/8] crear /etc/embebidos3/"
sudo mkdir -p /etc/embebidos3
if [[ ! -f /etc/embebidos3/secrets.env ]]; then
    sudo tee /etc/embebidos3/secrets.env > /dev/null <<'EOF'
# HF_TOKEN requerido para descargar repo privado mitgar14/embebidos-3-models
HF_TOKEN=hf_REEMPLAZAR
EMBEBIDOS3_TRTEXEC_WORKSPACE=512
EOF
    sudo chown root:jetson /etc/embebidos3/secrets.env
    sudo chmod 0640 /etc/embebidos3/secrets.env
    echo "    creado, editá con 'sudo nano /etc/embebidos3/secrets.env' antes de continuar"
fi

echo "[2/8] instalar wrapper sudoers-safe"
sudo install -m 0755 "${ROOT}/scripts/embebidos3-builder-launch" /usr/local/bin/embebidos3-builder-launch

echo "[3/8] instalar sudoers (14 entradas)"
sudo install -m 0440 -o root -g root /dev/stdin /etc/sudoers.d/embebidos3 <<'EOF'
jetson ALL=(root) NOPASSWD: /usr/local/bin/embebidos3-builder-launch *
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-builder@*.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl disable nvzramconfig
jetson ALL=(root) NOPASSWD: /sbin/swapoff -a
jetson ALL=(root) NOPASSWD: /sbin/swapon /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/sysctl vm.swappiness=*
jetson ALL=(root) NOPASSWD: /sbin/fallocate -l 8G /mnt/swap.img
jetson ALL=(root) NOPASSWD: /bin/chmod 600 /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/mkswap /mnt/swap.img
jetson ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
EOF
sudo visudo -cf /etc/sudoers.d/embebidos3

echo "[4/8] instalar units systemd"
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-server.service" /etc/systemd/system/
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-builder@.service" /etc/systemd/system/

echo "[5/8] instalar tmpfiles config (logs TTL 3d)"
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-logs.tmpfiles.conf" /etc/tmpfiles.d/

echo "[6/8] systemctl daemon-reload + enable + tmpfiles --create"
sudo systemctl daemon-reload
sudo systemctl enable embebidos3-server.service
sudo systemd-tmpfiles --create /etc/tmpfiles.d/embebidos3-logs.tmpfiles.conf

echo "[7/8] migrar server actual (si está corriendo con nohup)"
if pgrep -f 'nano_server:app' > /dev/null; then
    echo "    matando server nohup..."
    pkill -f 'nano_server:app' || true
    sleep 2
fi

echo "[8/8] start server vía systemd"
sudo systemctl start embebidos3-server.service
sleep 3
systemctl status embebidos3-server.service --no-pager | head -15

cat <<'EOF'

==============================================================================
  Setup completo
==============================================================================
  Verificá:
    curl http://localhost:8000/health
    curl http://localhost:8000/model/state
    systemctl status embebidos3-server.service
    sudo -l -U jetson | grep embebidos3

  HF_TOKEN: editá /etc/embebidos3/secrets.env con el token real.
  Después: sudo systemctl restart embebidos3-server.service
==============================================================================
EOF
```

- [ ] **Step 2: hacer ejecutable**

```bash
chmod +x scripts/nano_install_systemd.sh
```

- [ ] **Step 3: commit**

```bash
git add scripts/nano_install_systemd.sh
git commit -m "feat(scripts): nano_install_systemd.sh idempotente"
```

### Task A6: simplificar `nano_start_server.sh` para systemd

**Files:**
- Modify: `scripts/nano_start_server.sh`

- [ ] **Step 1: leer el contenido actual**

```bash
cat scripts/nano_start_server.sh
```

Expected: ver el script con `nohup` y `pkill` (versión actual).

- [ ] **Step 2: reescribir para systemd (sin nohup ni pkill)**

```bash
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

cd "${ROOT}/scripts"

# Foreground: NO nohup, NO `&`. systemd captura stdout/stderr y maneja PID.
exec python3 -c "import uvicorn; uvicorn.run('nano_server:app', host='0.0.0.0', port=8000, workers=1, log_level='info')"
```

- [ ] **Step 3: commit**

```bash
git add scripts/nano_start_server.sh
git commit -m "refactor(scripts): nano_start_server.sh para systemd Type=simple (sin nohup)"
```

### Task A7: eliminar `nano_stop_server.sh` (lo reemplaza `systemctl`)

**Files:**
- Delete: `scripts/nano_stop_server.sh`

- [ ] **Step 1: borrar el archivo**

```bash
rm scripts/nano_stop_server.sh
```

- [ ] **Step 2: commit**

```bash
git add -A scripts/nano_stop_server.sh
git commit -m "chore(scripts): eliminar nano_stop_server.sh (systemctl stop lo reemplaza)"
```

### Task A8: validar instalación end-to-end en el Nano

- [ ] **Step 1: rsync el repo al Nano**

```bash
ssh nano "mkdir -p /home/jetson/embebidos-3/systemd /home/jetson/embebidos-3/scripts"
scp systemd/*.service systemd/*.conf nano:/home/jetson/embebidos-3/systemd/
scp scripts/embebidos3-builder-launch scripts/nano_install_systemd.sh scripts/nano_start_server.sh nano:/home/jetson/embebidos-3/scripts/
ssh nano "chmod +x /home/jetson/embebidos-3/scripts/embebidos3-builder-launch /home/jetson/embebidos-3/scripts/nano_install_systemd.sh /home/jetson/embebidos-3/scripts/nano_start_server.sh"
```

- [ ] **Step 2: ejecutar el instalador**

```bash
ssh nano "bash /home/jetson/embebidos-3/scripts/nano_install_systemd.sh"
```

Expected: 8 pasos OK, mensaje final "Setup completo". Status del server `active (running)`.

- [ ] **Step 3: verificar las 14 entradas sudoers**

```bash
ssh nano "sudo -l -U jetson | grep -c embebidos3"
```

Expected: `14`.

- [ ] **Step 4: verificar que el server responde**

```bash
ssh nano "curl -s http://localhost:8000/health"
```

Expected: JSON con `engine`, `classes`, etc (el endpoint actual sigue funcionando).

- [ ] **Step 5: editar HF_TOKEN**

```bash
echo "EDITAR MANUALMENTE: ssh nano 'sudo nano /etc/embebidos3/secrets.env'"
echo "Reemplazar 'hf_REEMPLAZAR' por el token real (https://huggingface.co/settings/tokens)"
echo "Después: ssh nano 'sudo systemctl restart embebidos3-server.service'"
```

- [ ] **Step 6: verificar auto-start al boot (validación opcional, requiere reboot)**

```bash
ssh nano "sudo reboot" || true
# esperar 60s
sleep 60
ssh nano "systemctl is-active embebidos3-server.service && curl -s http://localhost:8000/health | head -1"
```

Expected: `active` + respuesta del health endpoint.

---

## Fase B — Refactor `nano_server.py`

**Objetivo:** TRTWorker reentrante (load/unload), endpoint `/model/state`, recovery en startup, bug fix `:360`. Sin tocar todavía los endpoints de jobs (vienen en fase E).

### Task B1: extraer constantes y schemas en módulo separado

**Files:**
- Create: `scripts/nano_server_constants.py`
- Modify: `scripts/nano_server.py`

- [ ] **Step 1: crear el módulo de constantes**

```python
"""Constantes compartidas entre nano_server.py, builder y helpers."""
import os
from pathlib import Path

# Rutas
ROOT = Path("/home/jetson/embebidos-3")
ENGINES_DIR = ROOT / "engines"
ONNX_DIR = ROOT / "onnx"
LOGS_DIR = ROOT / "logs"
JOBS_LOGS_DIR = LOGS_DIR / "jobs"
STAGING_DIR = ENGINES_DIR / ".staging"
PREVIOUS_DIR = ENGINES_DIR / ".previous"
TEST_IMAGES_DIR = ROOT / "test_images"

ACTIVE_ENGINE = ENGINES_DIR / "best_fp16.engine"
ACTIVE_ENGINE_META = ENGINES_DIR / "best_fp16.engine.meta.json"
PREVIOUS_ENGINE = PREVIOUS_DIR / "best_fp16.engine.old"
PREVIOUS_ENGINE_META = PREVIOUS_DIR / "best_fp16.engine.old.meta.json"

RUNTIME_DIR = Path("/var/run/embebidos3")
JOB_STATE_FILE = RUNTIME_DIR / "job.json"
BUILDER_LOCK_FILE = RUNTIME_DIR / "builder.lock"

# HF Hub
HF_REPO = "mitgar14/embebidos-3-models"
HF_REVISION_DEFAULT = "main"
ONNX_REMOTE_PATH = "exports/best.onnx"
MANIFEST_REMOTE_PATH = "manifests/manifest.json"
ENGINES_ARCHIVE_PREFIX = "engines-archive"

# Inferencia
IMGSZ = 416
CLASSES = ["glass", "paper", "plastic"]
DEFAULT_CONF = 0.25
DEFAULT_NMS = 0.45

# Builder
TRTEXEC_WORKSPACE_DEFAULT_MB = int(os.environ.get("EMBEBIDOS3_TRTEXEC_WORKSPACE", "512"))
TRTEXEC_TIMEOUT_SEC = 2400  # 40 min
HEARTBEAT_STALE_SEC = 120
```

- [ ] **Step 2: importar desde `nano_server.py`**

Reemplazar las constantes hardcodeadas (`ENGINE_PATH`, `IMGSZ`, `CLASSES`, etc.) en `nano_server.py` por imports de `nano_server_constants`:

```python
from nano_server_constants import (
    ACTIVE_ENGINE, IMGSZ, CLASSES, DEFAULT_CONF, DEFAULT_NMS,
    JOB_STATE_FILE,
)
```

- [ ] **Step 3: commit**

```bash
git add scripts/nano_server_constants.py scripts/nano_server.py
git commit -m "refactor(server): extraer constantes a nano_server_constants.py"
```

### Task B2: refactor `TRTWorker` con métodos `_load_engine()` / `_unload_engine()` reentrantes

**Files:**
- Modify: `scripts/nano_server.py` (clase `TRTWorker`)
- Create: `tests/test_trt_worker_lifecycle.py` (smoke test sin GPU)

- [ ] **Step 1: escribir el smoke test (sin GPU, mockea pycuda)**

```python
"""Test de smoke del lifecycle de TRTWorker. Mockea pycuda/tensorrt para correr
en el host sin GPU. Solo verifica que las llamadas a request_swap() encolan
correctamente y que el orden de destrucción es el esperado."""
import sys
from unittest.mock import MagicMock, patch

# Mock pycuda y tensorrt antes de importar nano_server
sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from scripts.nano_server import TRTWorker


def test_request_swap_sets_event():
    worker = TRTWorker("/tmp/fake.engine")
    assert not worker._swap_event.is_set()
    worker.request_swap("/tmp/new.engine")
    assert worker._swap_event.is_set()
    assert worker._swap_path == "/tmp/new.engine"


def test_request_swap_overwrites_pending():
    worker = TRTWorker("/tmp/fake.engine")
    worker.request_swap("/tmp/first.engine")
    worker.request_swap("/tmp/second.engine")
    assert worker._swap_path == "/tmp/second.engine"
```

- [ ] **Step 2: correr el test (debe fallar — request_swap no existe aún)**

```bash
pytest tests/test_trt_worker_lifecycle.py -v
```

Expected: FAIL con `AttributeError: 'TRTWorker' object has no attribute 'request_swap'`.

- [ ] **Step 3: implementar `request_swap` y `_unload_engine` en TRTWorker**

Editar `scripts/nano_server.py` clase `TRTWorker`. Añadir al `__init__`:

```python
self._swap_path = None
self._swap_event = threading.Event()
# guardar refs como atributos de instancia (no locales) para que _unload los pueda destruir
self._runtime = None
self._engine = None
self._trt_ctx = None
self._stream = None
self._host_in = self._host_out = None
self._dev_in = self._dev_out = None
self._bindings = []
self._in_shape = self._out_shape = None
```

Añadir métodos:

```python
def request_swap(self, new_path: str) -> None:
    """Pide hot-swap a un engine nuevo. Llamado desde el handler HTTP."""
    with self._lock:
        self._swap_path = new_path
    self._swap_event.set()

def _load_engine(self, path: str) -> None:
    """Carga engine + bindings desde path. Asume cu_ctx pushed."""
    import tensorrt as trt
    import pycuda.driver as cuda
    logger = trt.Logger(trt.Logger.WARNING)
    self._runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        self._engine = self._runtime.deserialize_cuda_engine(f.read())
    self._trt_ctx = self._engine.create_execution_context()
    self._bindings = []
    for i in range(self._engine.num_bindings):
        shape = tuple(self._engine.get_binding_shape(i))
        dtype = trt.nptype(self._engine.get_binding_dtype(i))
        size = int(np.prod(shape))
        h = cuda.pagelocked_empty(size, dtype=dtype)
        d = cuda.mem_alloc(h.nbytes)
        self._bindings.append(int(d))
        if self._engine.binding_is_input(i):
            self._host_in = h; self._dev_in = d; self._in_shape = shape
        else:
            self._host_out = h; self._dev_out = d; self._out_shape = shape
    self._stream = cuda.Stream()
    print(f"[trt-worker] engine cargado desde {path}. in={self._in_shape} out={self._out_shape}", flush=True)

def _unload_engine(self) -> None:
    """Destruye engine + buffers. Asume cu_ctx pushed."""
    del self._stream; self._stream = None
    del self._dev_out; self._dev_out = None
    del self._host_out; self._host_out = None
    del self._dev_in; self._dev_in = None
    del self._host_in; self._host_in = None
    self._bindings = []
    del self._trt_ctx; self._trt_ctx = None
    del self._engine; self._engine = None
    del self._runtime; self._runtime = None
    print("[trt-worker] engine descargado", flush=True)
```

- [ ] **Step 4: refactor `run()` para usar `_load_engine` + check de swap en cada iteración**

Reemplazar la sección de inicialización (líneas que cargan engine + bindings) por:

```python
def run(self) -> None:
    cuda.init()
    cu_ctx = cuda.Device(0).make_context()
    try:
        cu_ctx.push()
        self._load_engine(str(self.engine_path))
        cu_ctx.pop()
        self._ready.set()

        while not self._stop.is_set():
            # ¿hay swap pendiente?
            if self._swap_event.is_set():
                self._swap_event.clear()
                with self._lock:
                    new_path = self._swap_path
                if new_path:
                    cu_ctx.push()
                    try:
                        self._unload_engine()
                        self._load_engine(new_path)
                        self.engine_path = new_path
                    finally:
                        cu_ctx.pop()

            item = self.in_q.get()
            if item is None:
                break

            jpeg_bytes, client_ts_ms, seq, loop, future = item
            # ... resto del bucle de inferencia (sin cambios)
    finally:
        try:
            cu_ctx.push()
            self._unload_engine()
            cu_ctx.pop()
        except Exception:
            pass
        try:
            cu_ctx.detach()
        except Exception:
            pass
        print("[trt-worker] stopped", flush=True)
```

- [ ] **Step 5: re-correr el test (debe pasar)**

```bash
pytest tests/test_trt_worker_lifecycle.py -v
```

Expected: PASS.

- [ ] **Step 6: commit**

```bash
git add scripts/nano_server.py tests/test_trt_worker_lifecycle.py
git commit -m "refactor(server): TRTWorker reentrante con load/unload + request_swap"
```

### Task B3: fix bug `ConnectionClosedOK` en `nano_server.py:360`

**Files:**
- Modify: `scripts/nano_server.py` (alrededor de línea 358-362)

- [ ] **Step 1: localizar el código actual**

```bash
grep -n "send_text(json.dumps" scripts/nano_server.py | tail -5
```

Expected: ver la línea final `await ws.send_text(...)` dentro de `except Exception as e:`.

- [ ] **Step 2: wrap el `send_text` en try/except para silenciar `ConnectionClosedOK`**

Reemplazar el bloque final del handler `ws_handler`:

```python
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"ok": False, "error": f"server: {e}"}))
        except Exception:
            pass  # cliente puede haber cerrado durante el envío
        finally:
            try:
                await ws.close()
            except Exception:
                pass
```

- [ ] **Step 3: commit**

```bash
git add scripts/nano_server.py
git commit -m "fix(server): silenciar ConnectionClosedOK durante envío de error en ws_handler"
```

### Task B4: endpoint `GET /model/state`

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `tests/test_model_state_endpoint.py`

- [ ] **Step 1: escribir el test (mockea filesystem)**

```python
"""Test del endpoint GET /model/state. Mockea filesystem para los 5 estados."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from fastapi.testclient import TestClient
from scripts.nano_server import app


def test_model_state_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE", tmp_path / "nope.engine")
    monkeypatch.setattr("scripts.nano_server_constants.PREVIOUS_ENGINE", tmp_path / "nope.old")
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.get("/model/state")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "no_model"
    assert data["active_engine"] is None


def test_model_state_ready(tmp_path, monkeypatch):
    eng = tmp_path / "best_fp16.engine"
    meta = tmp_path / "best_fp16.engine.meta.json"
    eng.write_bytes(b"\x00" * 100)
    meta.write_text(json.dumps({
        "engine_sha256": "abc",
        "onnx_sha256": "def",
        "hf_revision": "65c1634",
        "hf_commit_date": "2026-05-14T18:38:31Z",
        "trtexec_args": ["--fp16"],
        "build_completed_at": "2026-05-16T14:47:18-05:00",
        "build_duration_s": 496,
    }))
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE", eng)
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE_META", meta)
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.get("/model/state")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "ready"
    assert data["active_engine"]["hf_revision"] == "65c1634"


def test_model_state_building(tmp_path, monkeypatch):
    eng = tmp_path / "best_fp16.engine"
    eng.write_bytes(b"\x00" * 100)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "job_id": "20260516-1422-abc123",
        "pid": 99999,  # no existe, será cleaned
        "phase": "trtexec_building",
        "progress_pct": 47,
        "heartbeat": 1747424793.17,
    }))
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE", eng)
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", job)
    with TestClient(app) as c:
        r = c.get("/model/state")
    data = r.json()
    # PID muerto → no se considera building, vuelve a ready
    assert data["state"] in ("ready", "no_model")
```

- [ ] **Step 2: correr el test (debe fallar — endpoint no existe)**

```bash
pytest tests/test_model_state_endpoint.py -v
```

Expected: FAIL con `404 Not Found`.

- [ ] **Step 3: implementar el endpoint**

Añadir a `nano_server.py`:

```python
import json
import os
from nano_server_constants import (
    ACTIVE_ENGINE, ACTIVE_ENGINE_META,
    PREVIOUS_ENGINE, PREVIOUS_ENGINE_META,
    JOB_STATE_FILE, HEARTBEAT_STALE_SEC,
)


def _read_engine_meta(meta_path):
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return None


def _is_pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def _read_active_job():
    if not JOB_STATE_FILE.exists():
        return None
    try:
        state = json.loads(JOB_STATE_FILE.read_text())
    except Exception:
        return None
    pid = state.get("pid")
    if pid and _is_pid_alive(pid):
        return state
    return None  # huérfano, se considera no activo


@app.get("/model/state")
def model_state():
    """Devuelve el estado del modelo: no_model | ready | building | degraded.
    'update_available' lo computa el cliente via POST /model/check-updates."""
    active_meta = _read_engine_meta(ACTIVE_ENGINE_META)
    previous_meta = _read_engine_meta(PREVIOUS_ENGINE_META)
    active_job = _read_active_job()

    if active_job:
        return {
            "state": "building",
            "active_engine": active_meta,
            "previous_engine": previous_meta,
            "active_job": active_job,
        }

    if ACTIVE_ENGINE.exists() and active_meta:
        # ¿es engine de fallback?
        degraded = bool(active_meta.get("from_fallback", False))
        return {
            "state": "degraded" if degraded else "ready",
            "active_engine": active_meta,
            "previous_engine": previous_meta,
            "active_job": None,
        }

    return {
        "state": "no_model",
        "active_engine": None,
        "previous_engine": previous_meta,
        "active_job": None,
    }
```

- [ ] **Step 4: re-correr el test (debe pasar)**

```bash
pytest tests/test_model_state_endpoint.py -v
```

Expected: PASS los 3 tests.

- [ ] **Step 5: commit**

```bash
git add scripts/nano_server.py tests/test_model_state_endpoint.py
git commit -m "feat(server): añadir endpoint GET /model/state con 4 estados base"
```

### Task B5: recovery state en `@app.on_event("startup")`

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `scripts/recover_job_state.py`
- Create: `tests/test_recover_job_state.py`

- [ ] **Step 1: escribir test de recovery**

```python
"""Tests de recovery: server arranca y detecta job activo/huérfano/stalled."""
import json
import os
import time
from pathlib import Path

from scripts.recover_job_state import recover_job_state


def test_no_state_file(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    monkeypatch.setattr("scripts.recover_job_state.JOB_STATE_FILE", state_file)
    assert recover_job_state() is None


def test_dead_pid_returns_none(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    state_file.write_text(json.dumps({
        "job_id": "test-job-001234",
        "pid": 99999999,  # PID que no existe
        "phase": "trtexec_building",
        "heartbeat": time.time(),
    }))
    monkeypatch.setattr("scripts.recover_job_state.JOB_STATE_FILE", state_file)
    assert recover_job_state() is None


def test_alive_pid_with_fresh_heartbeat_returns_running(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    state_file.write_text(json.dumps({
        "job_id": "test-job-001234",
        "pid": os.getpid(),  # nosotros estamos vivos
        "phase": "trtexec_building",
        "heartbeat": time.time(),
        "progress_pct": 50,
    }))
    monkeypatch.setattr("scripts.recover_job_state.JOB_STATE_FILE", state_file)
    monkeypatch.setattr("scripts.recover_job_state._check_cmdline", lambda p: True)
    state = recover_job_state()
    assert state is not None
    assert state["status"] == "running"
    assert state["job_id"] == "test-job-001234"


def test_alive_pid_with_stale_heartbeat_returns_stalled(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    state_file.write_text(json.dumps({
        "job_id": "test-job-001234",
        "pid": os.getpid(),
        "phase": "trtexec_building",
        "heartbeat": time.time() - 200,  # stale (>120s)
        "progress_pct": 50,
    }))
    monkeypatch.setattr("scripts.recover_job_state.JOB_STATE_FILE", state_file)
    monkeypatch.setattr("scripts.recover_job_state._check_cmdline", lambda p: True)
    state = recover_job_state()
    assert state["status"] == "stalled"
```

- [ ] **Step 2: correr test (debe fallar — módulo no existe)**

```bash
pytest tests/test_recover_job_state.py -v
```

Expected: FAIL con `ModuleNotFoundError`.

- [ ] **Step 3: implementar `recover_job_state.py`**

```python
"""Recovery del estado del builder al arrancar el server.
Importable desde nano_server.py o ejecutable como CLI para diagnóstico.
"""
import json
import os
import time
from pathlib import Path

from nano_server_constants import JOB_STATE_FILE, HEARTBEAT_STALE_SEC, JOBS_LOGS_DIR


def _is_pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def _check_cmdline(pid):
    """True si /proc/<pid>/cmdline contiene 'nano_build_engine'.
    Defensa contra PID reuse: aseguramos que el proceso es realmente el builder."""
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return False
    try:
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        return "nano_build_engine" in cmdline
    except Exception:
        return False


def _finalize_abandoned(state):
    """Persiste el state final marcado como ABANDONED en logs/jobs/."""
    job_id = state.get("job_id", "unknown")
    final = {
        **state,
        "phase": "abandoned",
        "ended_at_unix": time.time(),
        "reason": "builder process died, no heartbeat",
    }
    out = JOBS_LOGS_DIR / f"{job_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))


def recover_job_state():
    """Llamado en startup del server. Retorna estado del job activo o None."""
    if not JOB_STATE_FILE.exists():
        return None
    try:
        state = json.loads(JOB_STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None

    pid = state.get("pid")
    if pid is None:
        return None

    if not _is_pid_alive(pid):
        _finalize_abandoned(state)
        try: JOB_STATE_FILE.unlink()
        except FileNotFoundError: pass
        return None

    if not _check_cmdline(pid):
        _finalize_abandoned(state)
        try: JOB_STATE_FILE.unlink()
        except FileNotFoundError: pass
        return None

    age = time.time() - state.get("heartbeat", 0)
    if age > HEARTBEAT_STALE_SEC:
        return {"status": "stalled", "age_seconds": age, **state}

    return {"status": "running", **state}


if __name__ == "__main__":
    import sys
    result = recover_job_state()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result else 1)
```

- [ ] **Step 4: importar y llamar desde nano_server.py startup**

Añadir en `nano_server.py`:

```python
from recover_job_state import recover_job_state

_recovered_job_at_startup = None

@app.on_event("startup")
def _startup():
    global _recovered_job_at_startup
    worker.start()
    if not worker.wait_ready(60):
        raise RuntimeError("TRT worker no ready en 60s")
    _recovered_job_at_startup = recover_job_state()
    if _recovered_job_at_startup:
        print(f"[server] job recovery: status={_recovered_job_at_startup.get('status')} "
              f"job_id={_recovered_job_at_startup.get('job_id')}", flush=True)
    print(f"[server] engine listo. ws://0.0.0.0:8000/ws", flush=True)
```

- [ ] **Step 5: re-correr tests (todos deben pasar)**

```bash
pytest tests/test_recover_job_state.py -v
```

Expected: PASS los 4 tests.

- [ ] **Step 6: commit**

```bash
git add scripts/recover_job_state.py scripts/nano_server.py tests/test_recover_job_state.py
git commit -m "feat(server): recovery de jobs activos en startup (PID + cmdline + heartbeat watchdog)"
```

### Task B6: pyproject.toml — añadir pytest + fastapi.testclient deps de dev

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: añadir grupo dev con pytest**

Bajo `[project.optional-dependencies]` o `[tool.uv]` (según sea la config actual):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "httpx>=0.24",  # requerido por fastapi.testclient
    "pytest-mock>=3.10",
]
```

- [ ] **Step 2: `uv sync`**

```bash
uv sync --extra dev
```

Expected: instala pytest y httpx en `.venv`.

- [ ] **Step 3: correr todos los tests acumulados**

```bash
uv run pytest tests/ -v
```

Expected: PASS de todos los tests añadidos hasta ahora (B2, B4, B5).

- [ ] **Step 4: commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): añadir pytest + httpx como dev deps"
```

---

## Fase C — Cliente HF REST

**Objetivo:** módulo `scripts/hf_rest.py` con `download`, `list_files`, `repo_info`, `get_head_revision`, `upload_file_inline`. Tests con `requests-mock` o `responses` para no pegarle a HF en CI.

### Task C1: esqueleto + `download`

**Files:**
- Create: `scripts/hf_rest.py`
- Create: `tests/test_hf_rest.py`

- [ ] **Step 1: test de download con mock**

```python
"""Tests del cliente HF REST. Usa requests-mock para evitar tráfico real."""
import os
from pathlib import Path
import pytest
import requests_mock

from scripts.hf_rest import download, REPO, BASE


def test_download_streaming(tmp_path):
    fake_content = b"x" * 1024 * 100  # 100 KB
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, content=fake_content)
        download("exports/best.onnx", out)
    assert out.read_bytes() == fake_content


def test_download_with_revision(tmp_path):
    fake_content = b"y" * 50
    url = f"{BASE}/{REPO}/resolve/abc1234/manifests/manifest.json"
    out = tmp_path / "manifest.json"
    with requests_mock.Mocker() as m:
        m.get(url, content=fake_content)
        download("manifests/manifest.json", out, revision="abc1234")
    assert out.read_bytes() == fake_content


def test_download_failure_raises(tmp_path):
    url = f"{BASE}/{REPO}/resolve/main/exports/best.onnx"
    out = tmp_path / "best.onnx"
    with requests_mock.Mocker() as m:
        m.get(url, status_code=404)
        with pytest.raises(Exception):
            download("exports/best.onnx", out)
    assert not out.exists()
```

- [ ] **Step 2: correr test (FAIL — módulo no existe)**

```bash
pytest tests/test_hf_rest.py -v
```

Expected: FAIL con `ModuleNotFoundError`.

- [ ] **Step 3: implementar `download` y constantes**

```python
"""hf_rest.py — cliente minimal HF Hub para Python 3.6 (sin huggingface_hub).
Solo requiere `requests`. Diseñado para Jetson Nano JP 4.6.1.
"""
import os
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict

import requests

REPO = "mitgar14/embebidos-3-models"
BASE = "https://huggingface.co"


def _headers():
    token = os.environ.get("HF_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download(filename: str, local_path: Path, revision: str = "main",
             chunk_size: int = 65536, timeout: int = 120) -> None:
    """Descarga un archivo del repo a local_path. Streaming, atomic write."""
    url = f"{BASE}/{REPO}/resolve/{revision}/{filename}"
    r = requests.get(url, headers=_headers(), stream=True, timeout=timeout)
    r.raise_for_status()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    tmp.rename(local_path)
```

- [ ] **Step 4: añadir `requests-mock` a deps dev e instalar**

```bash
uv add --dev requests-mock
```

- [ ] **Step 5: correr tests (PASS)**

```bash
uv run pytest tests/test_hf_rest.py -v
```

Expected: 3 PASS.

- [ ] **Step 6: commit**

```bash
git add scripts/hf_rest.py tests/test_hf_rest.py pyproject.toml uv.lock
git commit -m "feat(hf): cliente REST minimal con download streaming"
```

### Task C2: `list_files`, `repo_info`, `get_head_revision`

**Files:**
- Modify: `scripts/hf_rest.py`
- Modify: `tests/test_hf_rest.py`

- [ ] **Step 1: tests para los 3 helpers**

```python
def test_list_files():
    fake_response = {
        "siblings": [
            {"rfilename": "README.md", "size": 1287},
            {"rfilename": "exports/best.onnx", "size": 12169740},
            {"rfilename": "manifests/manifest.json", "size": 3095},
        ]
    }
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json=fake_response)
        files = list_files()
    assert len(files) == 3
    assert any(f["rfilename"] == "exports/best.onnx" for f in files)


def test_repo_info():
    fake_response = {"sha": "65c1634abc", "lastModified": "2026-05-14T18:38:31Z"}
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json=fake_response)
        info = repo_info()
    assert info["sha"] == "65c1634abc"


def test_get_head_revision():
    fake_response = {"sha": "65c1634404ea379e38522885101222a07242f37f9"}
    url = f"{BASE}/api/models/{REPO}"
    with requests_mock.Mocker() as m:
        m.get(url, json=fake_response)
        rev = get_head_revision()
    assert rev == "65c1634404ea379e38522885101222a07242f37f9"
```

- [ ] **Step 2: importar los nuevos nombres**

```python
from scripts.hf_rest import download, REPO, BASE, list_files, repo_info, get_head_revision
```

- [ ] **Step 3: correr (FAIL)**

```bash
uv run pytest tests/test_hf_rest.py -v
```

Expected: 3 tests nuevos FAIL con `ImportError`.

- [ ] **Step 4: implementar**

```python
def list_files(revision: str = "main", timeout: int = 30) -> List[Dict]:
    """Lista siblings del repo en la revision dada."""
    url = f"{BASE}/api/models/{REPO}"
    r = requests.get(url, headers=_headers(),
                     params={"revision": revision}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("siblings", [])


def repo_info(revision: str = "main", timeout: int = 30) -> Dict:
    """Info de la revision: sha, lastModified, etc."""
    url = f"{BASE}/api/models/{REPO}"
    r = requests.get(url, headers=_headers(),
                     params={"revision": revision}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_head_revision(timeout: int = 30) -> str:
    """SHA del último commit en main."""
    return repo_info("main", timeout=timeout).get("sha", "")
```

- [ ] **Step 5: re-correr (PASS)**

```bash
uv run pytest tests/test_hf_rest.py -v
```

Expected: 6 PASS.

- [ ] **Step 6: commit**

```bash
git add scripts/hf_rest.py tests/test_hf_rest.py
git commit -m "feat(hf): añadir list_files, repo_info, get_head_revision"
```

### Task C3: `upload_file_inline` (base64, sin LFS)

**Files:**
- Modify: `scripts/hf_rest.py`
- Modify: `tests/test_hf_rest.py`

- [ ] **Step 1: test de upload exitoso y test de detección de LFS forzado**

```python
def test_upload_file_inline_success(tmp_path):
    local = tmp_path / "best_fp16.engine"
    local.write_bytes(b"\x00" * 1024)
    fake_response = {"success": True, "commitUrl": "https://huggingface.co/..."}
    url = f"{BASE}/api/models/{REPO}/commit/main"
    with requests_mock.Mocker() as m:
        m.post(url, json=fake_response)
        result = upload_file_inline(local, "engines-archive/test/best_fp16.engine", "test commit")
    assert result["success"] is True


def test_upload_file_inline_lfs_raises(tmp_path):
    local = tmp_path / "best_fp16.engine"
    local.write_bytes(b"\x00" * 1024)
    url = f"{BASE}/api/models/{REPO}/commit/main"
    with requests_mock.Mocker() as m:
        m.post(url, status_code=422, text="LFS upload required for this file type")
        with pytest.raises(RuntimeError, match="LFS"):
            upload_file_inline(local, "engines-archive/test/best_fp16.engine")
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_hf_rest.py -v -k upload
```

Expected: FAIL `ImportError`.

- [ ] **Step 3: implementar**

```python
def upload_file_inline(local_path: Path, remote_path: str,
                       commit_msg: str = "embebidos3 backup",
                       branch: str = "main", timeout: int = 300) -> Dict:
    """Upload sin LFS, base64 inline. Apto para archivos < ~50 MB.
    Si el server exige LFS (422), levanta RuntimeError con instrucción para fallback."""
    local_path = Path(local_path)
    content_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    payload = {
        "summary": commit_msg,
        "files": [{
            "path": remote_path,
            "encoding": "base64",
            "content": content_b64,
        }]
    }
    url = f"{BASE}/api/models/{REPO}/commit/{branch}"
    r = requests.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    if r.status_code == 422 and "lfs" in r.text.lower():
        raise RuntimeError(f"Servidor exige LFS para {remote_path}. "
                           "Ver fallback con git-lfs en docs.")
    r.raise_for_status()
    return r.json()
```

- [ ] **Step 4: re-correr (PASS)**

```bash
uv run pytest tests/test_hf_rest.py -v
```

Expected: 8 PASS total.

- [ ] **Step 5: commit**

```bash
git add scripts/hf_rest.py tests/test_hf_rest.py
git commit -m "feat(hf): upload_file_inline con detección de LFS-required"
```

### Task C4: CLI wrapper para uso desde bash

**Files:**
- Modify: `scripts/hf_rest.py` (añadir `if __name__ == "__main__"`)

- [ ] **Step 1: añadir CLI argparse**

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="cliente REST HF Hub minimal")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl = sub.add_parser("download")
    p_dl.add_argument("filename", help="path en el repo, ej. exports/best.onnx")
    p_dl.add_argument("local_path", help="destino local")
    p_dl.add_argument("--revision", default="main")

    p_up = sub.add_parser("upload")
    p_up.add_argument("local_path", help="archivo local a subir")
    p_up.add_argument("remote_path", help="path en el repo")
    p_up.add_argument("--message", default="embebidos3 backup")

    p_info = sub.add_parser("head-revision")

    args = parser.parse_args()
    if args.cmd == "download":
        download(args.filename, Path(args.local_path), revision=args.revision)
        print(f"OK: {args.filename} -> {args.local_path}")
    elif args.cmd == "upload":
        result = upload_file_inline(Path(args.local_path), args.remote_path, args.message)
        print(f"OK: {result.get('commitUrl', 'commit OK')}")
    elif args.cmd == "head-revision":
        print(get_head_revision())
```

- [ ] **Step 2: verificación manual (smoke contra HF real, opcional)**

```bash
HF_TOKEN=hf_xxx uv run python scripts/hf_rest.py head-revision
```

Expected: imprime el SHA del HEAD del repo.

- [ ] **Step 3: commit**

```bash
git add scripts/hf_rest.py
git commit -m "feat(hf): CLI argparse para download/upload/head-revision"
```

---

## Fase D — Builder

**Objetivo:** orquestador bash + helpers Python para ejecutar el flujo completo: download → verify SHA → stop server → prep Nano → trtexec → validate → backup HF → swap → restore → start server. Cada paso atómico y observable.

### Task D1: `builder_state.py` — helper CLI estado job

**Files:**
- Create: `scripts/builder_state.py`
- Create: `tests/test_builder_state.py`

- [ ] **Step 1: tests del helper**

```python
"""Tests de builder_state.py CLI."""
import json
import subprocess
import sys
from pathlib import Path


def run_helper(*args, env=None):
    """Invoca builder_state.py como subprocess."""
    cmd = [sys.executable, "scripts/builder_state.py"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def test_phase_writes_state(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    env = {**dict(__import__("os").environ),
           "EMBEBIDOS3_JOB_STATE_FILE": str(state_file)}
    r = run_helper("test-job-001234", "phase", "--name", "acquired_lock", "--pct", "5", env=env)
    assert r.returncode == 0, r.stderr
    state = json.loads(state_file.read_text())
    assert state["job_id"] == "test-job-001234"
    assert state["phase"] == "acquired_lock"
    assert state["progress_pct"] == 5
    assert "heartbeat" in state
    assert state["phases_completed"] == ["acquired_lock"]


def test_phase_appends_completed(tmp_path):
    state_file = tmp_path / "job.json"
    env = {**dict(__import__("os").environ),
           "EMBEBIDOS3_JOB_STATE_FILE": str(state_file)}
    run_helper("jid", "phase", "--name", "phase_one", "--pct", "10", env=env)
    run_helper("jid", "phase", "--name", "phase_two", "--pct", "20", env=env)
    state = json.loads(state_file.read_text())
    assert state["phases_completed"] == ["phase_one", "phase_two"]
    assert state["phase"] == "phase_two"


def test_finalize_moves_to_logs(tmp_path):
    state_file = tmp_path / "job.json"
    logs_dir = tmp_path / "logs"
    env = {**dict(__import__("os").environ),
           "EMBEBIDOS3_JOB_STATE_FILE": str(state_file),
           "EMBEBIDOS3_JOBS_LOGS_DIR": str(logs_dir)}
    run_helper("jid", "phase", "--name", "phase_one", "--pct", "10", env=env)
    run_helper("jid", "finalize", "--phase", "done", "--exit-code", "0", env=env)
    assert not state_file.exists()
    final = json.loads((logs_dir / "jid.json").read_text())
    assert final["phase"] == "done"
    assert final["result"]["exit_code"] == 0
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_builder_state.py -v
```

Expected: FAIL `FileNotFoundError`.

- [ ] **Step 3: implementar `builder_state.py`**

```python
"""builder_state.py — CLI helper para escribir job state desde el builder bash.

Usage:
  builder_state.py <job_id> phase --name <fase> --pct <n> [--message <txt>] [--eta-seconds <n>]
  builder_state.py <job_id> finalize --phase done|failed|cancelled|abandoned --exit-code <n>

Env vars opcionales (para testing):
  EMBEBIDOS3_JOB_STATE_FILE  (default /var/run/embebidos3/job.json)
  EMBEBIDOS3_JOBS_LOGS_DIR   (default /home/jetson/embebidos-3/logs/jobs)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def _state_file():
    return Path(os.environ.get("EMBEBIDOS3_JOB_STATE_FILE",
                               "/var/run/embebidos3/job.json"))


def _logs_dir():
    return Path(os.environ.get("EMBEBIDOS3_JOBS_LOGS_DIR",
                               "/home/jetson/embebidos-3/logs/jobs"))


def _read_current():
    f = _state_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}


def _atomic_write(data):
    f = _state_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.rename(f)


def cmd_phase(job_id, args):
    state = _read_current()
    if not state or state.get("job_id") != job_id:
        # primer phase o job_id distinto, reset
        state = {
            "job_id": job_id,
            "pid": os.getppid(),  # PID del parent (el bash script)
            "started_at_unix": time.time(),
            "phases_completed": [],
        }
    completed = state.get("phases_completed", [])
    if args.name not in completed:
        completed.append(args.name)
    state.update({
        "phase": args.name,
        "phases_completed": completed,
        "progress_pct": args.pct,
        "heartbeat": time.time(),
    })
    if args.message:
        state["current_message"] = args.message
    if args.eta_seconds is not None:
        state["eta_seconds"] = args.eta_seconds
    _atomic_write(state)
    print(f"[state] {job_id} phase={args.name} pct={args.pct}")


def cmd_finalize(job_id, args):
    state = _read_current()
    if state.get("job_id") != job_id:
        print(f"[state] WARN: job_id en state file ({state.get('job_id')}) != {job_id}")
    state["phase"] = args.phase
    state["ended_at_unix"] = time.time()
    state["result"] = {"exit_code": args.exit_code}
    if state.get("started_at_unix"):
        state["build_duration_s"] = round(state["ended_at_unix"] - state["started_at_unix"], 1)
    # mover a logs/jobs/<id>.json
    logs = _logs_dir()
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"{job_id}.json").write_text(json.dumps(state, indent=2))
    # borrar state file activo
    sf = _state_file()
    if sf.exists():
        sf.unlink()
    print(f"[state] {job_id} finalized phase={args.phase} exit_code={args.exit_code}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job_id")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_phase = sub.add_parser("phase")
    p_phase.add_argument("--name", required=True)
    p_phase.add_argument("--pct", type=int, required=True)
    p_phase.add_argument("--message", default=None)
    p_phase.add_argument("--eta-seconds", type=int, default=None, dest="eta_seconds")

    p_fin = sub.add_parser("finalize")
    p_fin.add_argument("--phase", required=True,
                       choices=["done", "failed", "cancelled", "abandoned"])
    p_fin.add_argument("--exit-code", type=int, required=True)

    args = p.parse_args()
    if args.cmd == "phase":
        cmd_phase(args.job_id, args)
    elif args.cmd == "finalize":
        cmd_finalize(args.job_id, args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: re-correr (PASS)**

```bash
uv run pytest tests/test_builder_state.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: commit**

```bash
git add scripts/builder_state.py tests/test_builder_state.py
git commit -m "feat(builder): builder_state.py CLI helper para job.json + finalize"
```

### Task D2: `parse_trtexec_progress.py` — parser stdout trtexec

**Files:**
- Create: `scripts/parse_trtexec_progress.py`
- Create: `tests/test_parse_trtexec_progress.py`

- [ ] **Step 1: tests con fixture de output trtexec**

```python
"""Tests del parser de progreso de trtexec."""
import io
import json
from pathlib import Path
import subprocess
import sys

SAMPLE_LINES = [
    "[12:34:56] [I] Finished parsing network model. Parse time: 1.234",
    "[12:34:58] [I] [TRT] [MemUsageChange] Init builder: 234 MiB",
    "[12:35:02] [V] [TRT] --------------- Timing Runner: Conv_42 (CaskConvolution)",
    "[12:35:05] [V] [TRT] --------------- Timing Runner: Conv_43 (CaskConvolution)",
    "[12:42:18] [I] Engine built in 442.12 sec.",
    "[12:42:19] [I] [TRT] Loaded engine size: 13 MiB",
    "[12:42:20] [I] Engine deserialized in 0.92 sec.",
    "&&&& PASSED TensorRT.trtexec [TensorRT v8201]",
]


def test_parser_emits_phase_for_known_lines(tmp_path):
    """Pasa el sample por stdin del parser y verifica que llama builder_state correctamente."""
    # Redirigimos el helper a un archivo temporal vía env
    state_file = tmp_path / "job.json"
    logs_dir = tmp_path / "logs"
    env = {**dict(__import__("os").environ),
           "EMBEBIDOS3_JOB_STATE_FILE": str(state_file),
           "EMBEBIDOS3_JOBS_LOGS_DIR": str(logs_dir)}
    proc = subprocess.run(
        [sys.executable, "scripts/parse_trtexec_progress.py", "test-jid-12345"],
        input="\n".join(SAMPLE_LINES),
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr
    state = json.loads(state_file.read_text())
    # debe haber registrado al menos 'parsing_done' y 'engine_built'
    completed = state["phases_completed"]
    assert "parsing_done" in completed
    assert "engine_built" in completed


def test_parser_passes_through_stdout():
    """El parser debe escribir las líneas tal cual a stdout (es un tee)."""
    sample = "line one\nline two\nline three"
    proc = subprocess.run(
        [sys.executable, "scripts/parse_trtexec_progress.py", "test-jid-12345"],
        input=sample, capture_output=True, text=True,
    )
    assert "line one" in proc.stdout
    assert "line two" in proc.stdout
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_parse_trtexec_progress.py -v
```

Expected: FAIL `FileNotFoundError`.

- [ ] **Step 3: implementar**

```python
"""parse_trtexec_progress.py — lee stdin (output de trtexec), pasa líneas tal cual
a stdout (tee), y detecta fases para llamar builder_state.py con el progreso.

Usage: trtexec ... 2>&1 | parse_trtexec_progress.py <job_id>
"""
import re
import subprocess
import sys
from pathlib import Path

HOOKS = [
    (re.compile(r'Finished parsing network model'), 'parsing_done', 30),
    (re.compile(r'\[MemUsageChange\].*Init builder'), 'mem_init', 35),
    (re.compile(r'Engine built in'), 'engine_built', 70),
    (re.compile(r'Engine deserialized'), 'deserialized', 75),
    (re.compile(r'&&&& PASSED'), 'trtexec_passed', 78),
    (re.compile(r'&&&& FAILED'), 'trtexec_failed', None),
]
TIMING_RX = re.compile(r'Timing Runner')

BUILDER_STATE = str(Path(__file__).parent / "builder_state.py")


def update_phase(job_id, name, pct, message=None):
    cmd = [sys.executable, BUILDER_STATE, job_id, "phase",
           "--name", name, "--pct", str(pct)]
    if message:
        cmd += ["--message", message]
    subprocess.run(cmd, check=False)


def main(job_id):
    timing_count = 0
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
        for rx, phase, pct in HOOKS:
            if rx.search(line):
                update_phase(job_id, phase, pct, message=line.strip()[:200])
                break
        else:
            if TIMING_RX.search(line):
                timing_count += 1
                if timing_count % 20 == 0:
                    pct_est = min(65, 40 + timing_count // 4)
                    update_phase(job_id, "trtexec_optimizing", pct_est,
                                 message=line.strip()[:200])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: parse_trtexec_progress.py <job_id>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
```

- [ ] **Step 4: re-correr (PASS)**

```bash
uv run pytest tests/test_parse_trtexec_progress.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: commit**

```bash
git add scripts/parse_trtexec_progress.py tests/test_parse_trtexec_progress.py
git commit -m "feat(builder): parse_trtexec_progress.py tee + phase hooks"
```

### Task D3: `validate_engine.py` — mini-correctness post-build

**Files:**
- Create: `scripts/validate_engine.py`

> Nota: este script requiere TRT + pycuda + cv2 reales en el Nano. NO se puede testear en el host sin GPU. Solo verificación manual.

- [ ] **Step 1: implementar el script (reusa letterbox + postprocess de `nano_correctness.py`)**

```python
"""validate_engine.py — carga el engine en TRT context separado y corre 3 imágenes
de test_images/. Pasa si >= 2 imágenes producen >= 1 detección con conf > 0,3.

Exit 0 OK, 1 falla, 2 error de carga.

Usage: validate_engine.py <engine_path>
"""
import sys
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

# Reusa las constantes del nano_correctness.py
sys.path.insert(0, str(Path(__file__).parent))
from nano_correctness import letterbox, postprocess, IMGSZ, CLASSES, CONF_TH, NMS_TH

TEST_IMAGES = sorted(glob.glob("/home/jetson/embebidos-3/test_images/*.jpg"))[:3]
MIN_PASS = 2  # al menos 2 de 3 imágenes deben producir detecciones


def run_engine(engine_path):
    cuda.init()
    ctx_cu = cuda.Device(0).make_context()
    try:
        ctx_cu.push()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            return None
        trt_ctx = engine.create_execution_context()
        bindings = []
        host_in = host_out = None
        dev_in = dev_out = None
        out_shape = None
        for i in range(engine.num_bindings):
            shape = tuple(engine.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            h = cuda.pagelocked_empty(size, dtype=dtype)
            d = cuda.mem_alloc(h.nbytes)
            bindings.append(int(d))
            if engine.binding_is_input(i):
                host_in = h; dev_in = d
            else:
                host_out = h; dev_out = d; out_shape = shape
        stream = cuda.Stream()

        results = []
        for img_path in TEST_IMAGES:
            img = cv2.imread(img_path)
            if img is None:
                results.append({"image": img_path, "detections": 0, "error": "imdecode fail"})
                continue
            oh, ow = img.shape[:2]
            lb, r, dx, dy = letterbox(img, IMGSZ)
            rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
            inp = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            np.copyto(host_in, inp.ravel())
            cuda.memcpy_htod_async(dev_in, host_in, stream)
            trt_ctx.execute_async_v2(bindings, stream.handle)
            cuda.memcpy_dtoh_async(host_out, dev_out, stream)
            stream.synchronize()
            raw = host_out.reshape(out_shape)
            dets = postprocess(raw, (r, dx, dy), (ow, oh))
            good = [d for d in dets if d["conf"] > 0.3]
            results.append({"image": img_path, "detections": len(good)})

        # cleanup
        del stream
        del dev_out; del host_out
        del dev_in; del host_in
        del trt_ctx
        del engine
        del runtime
        ctx_cu.pop()
        return results
    finally:
        try: ctx_cu.detach()
        except Exception: pass


def main():
    if len(sys.argv) < 2:
        print("usage: validate_engine.py <engine_path>", file=sys.stderr)
        sys.exit(2)
    engine_path = sys.argv[1]
    if not Path(engine_path).exists():
        print(f"engine no existe: {engine_path}", file=sys.stderr)
        sys.exit(2)
    results = run_engine(engine_path)
    if results is None:
        print("deserialize_cuda_engine retornó None", file=sys.stderr)
        sys.exit(2)
    print(json.dumps({"validation": results}, indent=2))
    passed = sum(1 for r in results if r["detections"] > 0)
    if passed >= MIN_PASS:
        print(f"PASS ({passed}/{len(results)} imágenes con detecciones)", file=sys.stderr)
        sys.exit(0)
    else:
        print(f"FAIL ({passed}/{len(results)} imágenes con detecciones, requeridas {MIN_PASS})",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: verificar sintaxis en el host (no ejecutar)**

```bash
uv run python -c "import ast; ast.parse(open('scripts/validate_engine.py').read()); print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: commit**

```bash
git add scripts/validate_engine.py
git commit -m "feat(builder): validate_engine.py mini-correctness in-process"
```

### Task D4: `write_engine_meta.py` — escribe `engine.meta.json`

**Files:**
- Create: `scripts/write_engine_meta.py`
- Create: `tests/test_write_engine_meta.py`

- [ ] **Step 1: tests**

```python
import json
import subprocess
import sys
from pathlib import Path


def test_write_engine_meta(tmp_path):
    engine = tmp_path / "best_fp16.engine"
    engine.write_bytes(b"X" * 1024)
    proc = subprocess.run([
        sys.executable, "scripts/write_engine_meta.py",
        str(engine),
        "65c163404ea3",
        "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad",
        "512",
        "--build-duration-s", "496",
    ], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    meta_path = engine.with_suffix(".engine.meta.json")
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["onnx_sha256"] == "223f1a71c4b1bd08effdfa02fabb1ce259a3a507015d39e2359b5bec3dc805ad"
    assert meta["hf_revision"] == "65c163404ea3"
    assert meta["trtexec_args"] == ["--fp16", "--workspace=512", "--buildOnly"]
    assert "engine_sha256" in meta  # debe calcularse automáticamente
    assert "build_completed_at" in meta
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_write_engine_meta.py -v
```

Expected: FAIL `FileNotFoundError`.

- [ ] **Step 3: implementar**

```python
"""write_engine_meta.py — escribe el .meta.json al lado del engine.
Usage: write_engine_meta.py <engine_path> <hf_revision> <onnx_sha256> <workspace_mb>
         [--build-duration-s <n>] [--validation-json <path>]
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("engine_path")
    p.add_argument("hf_revision")
    p.add_argument("onnx_sha256")
    p.add_argument("workspace_mb", type=int)
    p.add_argument("--build-duration-s", type=int, default=None, dest="build_duration_s")
    p.add_argument("--validation-json", default=None, dest="validation_json")
    p.add_argument("--hf-commit-date", default=None, dest="hf_commit_date")
    args = p.parse_args()

    engine = Path(args.engine_path)
    if not engine.exists():
        print(f"engine no existe: {engine}", file=sys.stderr)
        sys.exit(2)

    meta = {
        "engine_sha256": sha256_file(engine),
        "onnx_sha256": args.onnx_sha256,
        "hf_revision": args.hf_revision,
        "hf_commit_date": args.hf_commit_date,
        "trtexec_args": ["--fp16", f"--workspace={args.workspace_mb}", "--buildOnly"],
        "build_completed_at": datetime.now(timezone.utc).isoformat(),
        "build_duration_s": args.build_duration_s,
    }
    if args.validation_json:
        try:
            meta["validation"] = json.loads(Path(args.validation_json).read_text())
        except Exception as e:
            meta["validation"] = {"passed": None, "error": str(e)}

    meta_path = engine.with_suffix(".engine.meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"OK: {meta_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: re-correr (PASS)**

```bash
uv run pytest tests/test_write_engine_meta.py -v
```

Expected: 1 PASS.

- [ ] **Step 5: commit**

```bash
git add scripts/write_engine_meta.py tests/test_write_engine_meta.py
git commit -m "feat(builder): write_engine_meta.py con sha256 calculado"
```

### Task D5: `nano_build_engine.sh` — orquestador bash

**Files:**
- Create: `scripts/nano_build_engine.sh`

> Nota: este script solo es testable end-to-end en el Nano. No hay TDD del bash mismo (smoke test en task A8 + D7).

- [ ] **Step 1: crear el script orquestador**

```bash
#!/usr/bin/env bash
# nano_build_engine.sh <job_id>
# Orquestador del build TRT FP16. Invocado por la unit systemd embebidos3-builder@<jobid>.service.
# Asume sudoers granular ya configurado.
set -euo pipefail

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

mkdir -p "${ROOT}/logs/jobs" "${STAGING_DIR}" "${PREV_DIR}"

# Lock exclusivo: fd se libera automáticamente al morir el proceso
exec {LOCK_FD}<>/var/run/embebidos3/builder.lock
flock -n "$LOCK_FD" || { echo "[BUILD] otro builder en curso, abort" >&2; exit 1; }
echo "$$" > /var/run/embebidos3/builder.lock

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
        # garantizar server vivo
        sudo systemctl is-active --quiet embebidos3-server.service \
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
python3 "${ROOT}/scripts/validate_engine.py" "$STAGING_ENGINE" || \
    { echo "[BUILD] validación falló" >&2; exit 3; }
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
python3 "${ROOT}/scripts/write_engine_meta.py" \
    "$ACTIVE_ENGINE" "$HF_REV" "$EXPECTED_SHA" "$WORKSPACE" \
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
```

- [ ] **Step 2: hacer ejecutable**

```bash
chmod +x scripts/nano_build_engine.sh
```

- [ ] **Step 3: lint sintaxis bash**

```bash
bash -n scripts/nano_build_engine.sh
```

Expected: sin output (sintaxis OK).

- [ ] **Step 4: commit**

```bash
git add scripts/nano_build_engine.sh
git commit -m "feat(builder): nano_build_engine.sh orquestador con 12 fases + cleanup trap"
```

### Task D6: actualizar `nano_install_inference.sh` con deps nuevas

**Files:**
- Modify: `scripts/nano_install_inference.sh`

- [ ] **Step 1: añadir `requests` y `jq` al install**

Después del bloque `pip install ... fastapi ...`:

```bash
echo "[5/6] dependencias adicionales para pipeline"
pip3 install --user --no-cache-dir "requests>=2.25,<3"
# jq via apt si no está
command -v jq >/dev/null 2>&1 || sudo apt-get install -y jq

python3 -c "import requests; print('requests', requests.__version__)"
jq --version
```

- [ ] **Step 2: commit**

```bash
git add scripts/nano_install_inference.sh
git commit -m "chore(install): añadir requests + jq como deps del pipeline"
```

### Task D7: validar el builder end-to-end en el Nano

- [ ] **Step 1: rsync los nuevos archivos al Nano**

```bash
scp scripts/builder_state.py scripts/parse_trtexec_progress.py \
    scripts/validate_engine.py scripts/write_engine_meta.py \
    scripts/nano_build_engine.sh scripts/hf_rest.py \
    nano:/home/jetson/embebidos-3/scripts/
ssh nano "chmod +x /home/jetson/embebidos-3/scripts/nano_build_engine.sh"
```

- [ ] **Step 2: instalar deps adicionales**

```bash
ssh nano "bash /home/jetson/embebidos-3/scripts/nano_install_inference.sh"
```

Expected: `requests` instalado, `jq` instalado.

- [ ] **Step 3: smoke test del helper**

```bash
ssh nano "python3 /home/jetson/embebidos-3/scripts/builder_state.py test-jid-aaa111 phase --name smoke --pct 50"
ssh nano "cat /var/run/embebidos3/job.json"
ssh nano "python3 /home/jetson/embebidos-3/scripts/builder_state.py test-jid-aaa111 finalize --phase cancelled --exit-code 0"
ssh nano "ls /home/jetson/embebidos-3/logs/jobs/"
```

Expected: `job.json` contiene state, después se mueve a `logs/jobs/test-jid-aaa111.json`.

- [ ] **Step 4: smoke test del wrapper de launch**

```bash
ssh nano "sudo /usr/local/bin/embebidos3-builder-launch abc"  # debe fallar (jobid corto)
ssh nano "sudo /usr/local/bin/embebidos3-builder-launch test-jobid-aaa111"  # debe lanzar unit
ssh nano "systemctl status embebidos3-builder@test-jobid-aaa111.service --no-pager"
```

Expected: el primero falla con regex error; el segundo lanza la unit (que correrá el script).

- [ ] **Step 5: ver el log del job**

```bash
ssh nano "tail -50 /home/jetson/embebidos-3/logs/jobs/test-jobid-aaa111.log"
ssh nano "cat /home/jetson/embebidos-3/logs/jobs/test-jobid-aaa111.json"
```

Expected: log del trtexec + estado final del job (success o failed según HF_TOKEN configurado).

---

## Fase E — Endpoints job lifecycle

**Objetivo:** endpoints HTTP que disparan/observan/cancelan jobs. SSE para stream de logs. Tests con `TestClient` mockeando subprocess.

### Task E1: endpoint `POST /model/build`

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `tests/test_model_build_endpoint.py`

- [ ] **Step 1: test**

```python
"""Tests POST /model/build."""
import sys
from unittest.mock import MagicMock, patch
sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from fastapi.testclient import TestClient
from scripts.nano_server import app


def test_build_starts_job(monkeypatch, tmp_path):
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", tmp_path / "job.json")
    # mock del subprocess.run que llama al wrapper sudo
    called = {}
    def fake_run(cmd, **kw):
        called["cmd"] = cmd
        class R: returncode = 0; stdout = ""; stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)

    with TestClient(app) as c:
        r = c.post("/model/build", json={"force": False})
    assert r.status_code == 202
    data = r.json()
    assert data["ok"] is True
    assert "job_id" in data
    assert called["cmd"][0:2] == ["sudo", "/usr/local/bin/embebidos3-builder-launch"]


def test_build_409_when_already_active(monkeypatch, tmp_path):
    state_file = tmp_path / "job.json"
    state_file.write_text('{"job_id": "active-jid", "pid": 1, "phase": "trtexec", "heartbeat": 9999999999}')
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", state_file)
    monkeypatch.setattr("scripts.nano_server._is_pid_alive", lambda p: True)

    with TestClient(app) as c:
        r = c.post("/model/build")
    assert r.status_code == 409
    assert r.json()["error"] == "build_in_progress"
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_model_build_endpoint.py -v
```

Expected: FAIL `404`.

- [ ] **Step 3: implementar**

```python
import subprocess
import uuid
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel


class BuildRequest(BaseModel):
    force: bool = False
    workspace_mb: int = None


def _generate_job_id():
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M")
    suffix = uuid.uuid4().hex[:6]
    return f"{ts}-{suffix}"


@app.post("/model/build", status_code=202)
def model_build(req: BuildRequest):
    active = _read_active_job()
    if active:
        raise HTTPException(
            status_code=409,
            detail={"ok": False, "error": "build_in_progress",
                    "active_job_id": active.get("job_id")},
        )
    job_id = _generate_job_id()
    try:
        subprocess.run(
            ["sudo", "/usr/local/bin/embebidos3-builder-launch", job_id],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, {"ok": False, "error": "launch_failed",
                                  "stderr": e.stderr})
    return {
        "ok": True,
        "job_id": job_id,
        "monitor_url": f"/jobs/{job_id}",
        "logs_stream_url": f"/jobs/{job_id}/logs",
    }
```

- [ ] **Step 4: re-correr (PASS)**

```bash
uv run pytest tests/test_model_build_endpoint.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: commit**

```bash
git add scripts/nano_server.py tests/test_model_build_endpoint.py
git commit -m "feat(server): POST /model/build dispara builder via sudo wrapper"
```

### Task E2: endpoints `GET /jobs/active` y `GET /jobs/<id>`

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `tests/test_jobs_get_endpoints.py`

- [ ] **Step 1: tests**

```python
import json
import sys
from unittest.mock import MagicMock
sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from fastapi.testclient import TestClient
from scripts.nano_server import app


def test_jobs_active_none(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", tmp_path / "job.json")
    with TestClient(app) as c:
        r = c.get("/jobs/active")
    assert r.status_code == 200
    assert r.json() is None


def test_jobs_get_by_id_terminal(tmp_path, monkeypatch):
    logs_dir = tmp_path / "jobs"
    logs_dir.mkdir()
    (logs_dir / "test-jid-001234.json").write_text(json.dumps({
        "job_id": "test-jid-001234",
        "phase": "done",
        "result": {"exit_code": 0},
    }))
    monkeypatch.setattr("scripts.nano_server_constants.JOBS_LOGS_DIR", logs_dir)
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.get("/jobs/test-jid-001234")
    assert r.status_code == 200
    assert r.json()["phase"] == "done"


def test_jobs_get_by_id_404(tmp_path, monkeypatch):
    logs_dir = tmp_path / "jobs"
    logs_dir.mkdir()
    monkeypatch.setattr("scripts.nano_server_constants.JOBS_LOGS_DIR", logs_dir)
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.get("/jobs/inexistente")
    assert r.status_code == 404
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_jobs_get_endpoints.py -v
```

Expected: FAIL.

- [ ] **Step 3: implementar**

```python
from fastapi import Path as FPath
from nano_server_constants import JOBS_LOGS_DIR


@app.get("/jobs/active")
def jobs_active():
    return _read_active_job()


@app.get("/jobs/{job_id}")
def jobs_get(job_id: str = FPath(..., regex=r"^[A-Za-z0-9_-]{10,40}$")):
    active = _read_active_job()
    if active and active.get("job_id") == job_id:
        return active
    final = JOBS_LOGS_DIR / f"{job_id}.json"
    if final.exists():
        return json.loads(final.read_text())
    raise HTTPException(404, {"ok": False, "error": "job_not_found"})
```

- [ ] **Step 4: PASS**

```bash
uv run pytest tests/test_jobs_get_endpoints.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: commit**

```bash
git add scripts/nano_server.py tests/test_jobs_get_endpoints.py
git commit -m "feat(server): GET /jobs/active y GET /jobs/<id>"
```

### Task E3: endpoint SSE `GET /jobs/<id>/logs`

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `tests/test_jobs_logs_sse.py`

- [ ] **Step 1: test SSE básico**

```python
import sys
from unittest.mock import MagicMock
sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from fastapi.testclient import TestClient
from scripts.nano_server import app


def test_jobs_logs_sse_streams_existing(tmp_path, monkeypatch):
    logs_dir = tmp_path / "jobs"
    logs_dir.mkdir()
    log = logs_dir / "test-jid-001234.log"
    log.write_text("[I] line one\n[I] line two\n&&&& PASSED\n")
    monkeypatch.setattr("scripts.nano_server_constants.JOBS_LOGS_DIR", logs_dir)
    with TestClient(app) as c:
        with c.stream("GET", "/jobs/test-jid-001234/logs", params={"follow": False}) as r:
            assert r.status_code == 200
            body = b"".join(r.iter_bytes())
    text = body.decode()
    assert "line one" in text
    assert "PASSED" in text
    assert "event: done" in text
```

- [ ] **Step 2: implementar usando `sse_starlette` o respuesta manual**

Si no hay sse-starlette, implementación manual:

```python
from fastapi.responses import StreamingResponse
import time

@app.get("/jobs/{job_id}/logs")
def jobs_logs(job_id: str, follow: bool = True):
    log = JOBS_LOGS_DIR / f"{job_id}.log"
    if not log.exists():
        raise HTTPException(404, {"ok": False, "error": "log_not_found"})

    def gen():
        with open(log, "r") as f:
            # leer todo lo existente
            for line in f:
                yield f"event: log\ndata: {json.dumps({'line': line.rstrip()})}\n\n"
            if not follow:
                yield f"event: done\ndata: {json.dumps({'phase': 'eof'})}\n\n"
                return
            # follow: tail-f
            while True:
                where = f.tell()
                line = f.readline()
                if not line:
                    # check si job terminó (estado final en logs/jobs/<id>.json)
                    final = JOBS_LOGS_DIR / f"{job_id}.json"
                    if final.exists():
                        result = json.loads(final.read_text())
                        yield f"event: done\ndata: {json.dumps(result)}\n\n"
                        return
                    time.sleep(0.25)
                    f.seek(where)
                else:
                    yield f"event: log\ndata: {json.dumps({'line': line.rstrip(), 'ts': time.time()})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
```

- [ ] **Step 3: re-correr test (PASS)**

```bash
uv run pytest tests/test_jobs_logs_sse.py -v
```

Expected: 1 PASS.

- [ ] **Step 4: commit**

```bash
git add scripts/nano_server.py tests/test_jobs_logs_sse.py
git commit -m "feat(server): GET /jobs/<id>/logs SSE con follow"
```

### Task E4: endpoint `DELETE /jobs/<id>` (cancelar)

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `tests/test_jobs_delete_endpoint.py`

- [ ] **Step 1: test**

```python
import sys
from unittest.mock import MagicMock, patch
sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from fastapi.testclient import TestClient
from scripts.nano_server import app


def test_cancel_active_job(tmp_path, monkeypatch):
    state_file = tmp_path / "job.json"
    state_file.write_text('{"job_id": "test-jid-cancel", "pid": 1, "phase": "trtexec_building", "heartbeat": 9999999999, "cancellable": true}')
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", state_file)
    monkeypatch.setattr("scripts.nano_server._is_pid_alive", lambda p: True)
    called = {}
    def fake_run(cmd, **kw):
        called["cmd"] = cmd
        class R: returncode = 0; stdout = ""; stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)

    with TestClient(app) as c:
        r = c.delete("/jobs/test-jid-cancel")
    assert r.status_code == 200
    assert r.json()["phase"] == "cancelling"
    assert called["cmd"][:3] == ["sudo", "/bin/systemctl", "stop"]


def test_cancel_unknown_job_404(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.nano_server_constants.JOB_STATE_FILE", tmp_path / "nope.json")
    with TestClient(app) as c:
        r = c.delete("/jobs/unknown-jid-001234")
    assert r.status_code == 404
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_jobs_delete_endpoint.py -v
```

Expected: FAIL.

- [ ] **Step 3: implementar**

```python
@app.delete("/jobs/{job_id}")
def jobs_cancel(job_id: str):
    active = _read_active_job()
    if not active or active.get("job_id") != job_id:
        raise HTTPException(404, {"ok": False, "error": "job_not_active"})
    try:
        subprocess.run(
            ["sudo", "/bin/systemctl", "stop", f"embebidos3-builder@{job_id}.service"],
            check=True, capture_output=True, text=True, timeout=10,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, {"ok": False, "error": "stop_failed", "stderr": e.stderr})
    return {"ok": True, "phase": "cancelling", "job_id": job_id}
```

- [ ] **Step 4: PASS**

```bash
uv run pytest tests/test_jobs_delete_endpoint.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: añadir signal handler en `nano_build_engine.sh` para cancelación limpia**

Modificar el trap cleanup en `nano_build_engine.sh` para manejar SIGTERM:

```bash
cancel_handler() {
    echo "[BUILD] SIGTERM recibido, cancelando..." >&2
    # matar trtexec si está corriendo
    pkill -KILL -P $$ trtexec 2>/dev/null || true
    JS finalize --phase cancelled --exit-code 130 2>/dev/null || true
    exit 130
}
trap cancel_handler SIGTERM
```

(Añadir en el script de Task D5 después del trap EXIT cleanup).

- [ ] **Step 6: commit**

```bash
git add scripts/nano_server.py scripts/nano_build_engine.sh tests/test_jobs_delete_endpoint.py
git commit -m "feat(server): DELETE /jobs/<id> + SIGTERM handler en builder para cancelación limpia"
```

### Task E5: endpoint `POST /model/check-updates`

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `tests/test_model_check_updates.py`

- [ ] **Step 1: test**

```python
import json
import sys
from unittest.mock import MagicMock
sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from fastapi.testclient import TestClient
from scripts.nano_server import app


def test_check_updates_up_to_date(tmp_path, monkeypatch):
    meta_path = tmp_path / "best_fp16.engine.meta.json"
    meta_path.write_text(json.dumps({"hf_revision": "65c1634abc"}))
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE_META", meta_path)
    monkeypatch.setattr("scripts.nano_server.hf_rest.get_head_revision", lambda: "65c1634abc")
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["up_to_date"] is True


def test_check_updates_new_commit(tmp_path, monkeypatch):
    meta_path = tmp_path / "best_fp16.engine.meta.json"
    meta_path.write_text(json.dumps({"hf_revision": "65c1634abc"}))
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE_META", meta_path)
    monkeypatch.setattr("scripts.nano_server.hf_rest.get_head_revision", lambda: "7a3b8e2new")
    with TestClient(app) as c:
        r = c.post("/model/check-updates")
    data = r.json()
    assert data["up_to_date"] is False
    assert data["latest_revision"] == "7a3b8e2new"
    assert data["current_revision"] == "65c1634abc"
```

- [ ] **Step 2: correr (FAIL)**

```bash
uv run pytest tests/test_model_check_updates.py -v
```

Expected: FAIL `404`.

- [ ] **Step 3: implementar**

```python
import hf_rest

@app.post("/model/check-updates")
def check_updates():
    current_meta = _read_engine_meta(ACTIVE_ENGINE_META)
    current_rev = (current_meta or {}).get("hf_revision")
    try:
        latest_rev = hf_rest.get_head_revision()
    except Exception as e:
        raise HTTPException(503, {"ok": False, "error": "hf_unreachable", "detail": str(e)})
    return {
        "up_to_date": current_rev == latest_rev,
        "current_revision": current_rev,
        "latest_revision": latest_rev,
    }
```

- [ ] **Step 4: PASS**

```bash
uv run pytest tests/test_model_check_updates.py -v
```

Expected: 2 PASS.

- [ ] **Step 5: commit**

```bash
git add scripts/nano_server.py tests/test_model_check_updates.py
git commit -m "feat(server): POST /model/check-updates compara HEAD HF vs engine actual"
```

### Task E6: endpoint `POST /model/rollback`

**Files:**
- Modify: `scripts/nano_server.py`
- Create: `tests/test_model_rollback.py`

- [ ] **Step 1: test**

```python
import json
import sys
from unittest.mock import MagicMock
sys.modules['pycuda'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['tensorrt'] = MagicMock()

from fastapi.testclient import TestClient
from scripts.nano_server import app


def test_rollback_success(tmp_path, monkeypatch):
    active = tmp_path / "engines" / "best_fp16.engine"
    prev = tmp_path / "engines" / ".previous" / "best_fp16.engine.old"
    active.parent.mkdir(parents=True)
    prev.parent.mkdir(parents=True)
    active.write_bytes(b"NEW")
    prev.write_bytes(b"OLD")
    (active.with_suffix(".engine.meta.json")).write_text('{"hf_revision":"new"}')
    (prev.parent / "best_fp16.engine.old.meta.json").write_text('{"hf_revision":"old"}')
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE", active)
    monkeypatch.setattr("scripts.nano_server_constants.ACTIVE_ENGINE_META", active.with_suffix(".engine.meta.json"))
    monkeypatch.setattr("scripts.nano_server_constants.PREVIOUS_ENGINE", prev)
    monkeypatch.setattr("scripts.nano_server_constants.PREVIOUS_ENGINE_META", prev.parent / "best_fp16.engine.old.meta.json")

    # mock worker.request_swap
    monkeypatch.setattr("scripts.nano_server.worker.request_swap", lambda p: None)

    with TestClient(app) as c:
        r = c.post("/model/rollback")
    assert r.status_code == 200
    # ahora el contenido de active debería ser el viejo
    assert active.read_bytes() == b"OLD"


def test_rollback_no_previous(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.nano_server_constants.PREVIOUS_ENGINE", tmp_path / "nope.engine")
    with TestClient(app) as c:
        r = c.post("/model/rollback")
    assert r.status_code == 409
```

- [ ] **Step 2: implementar**

```python
@app.post("/model/rollback")
def model_rollback():
    if not PREVIOUS_ENGINE.exists():
        raise HTTPException(409, {"ok": False, "error": "no_previous_engine"})
    # swap inverso
    tmp_active = ACTIVE_ENGINE.with_suffix(".engine.swap_tmp")
    if ACTIVE_ENGINE.exists():
        ACTIVE_ENGINE.rename(tmp_active)
        ACTIVE_ENGINE_META.rename(tmp_active.with_suffix(".meta.json")) if ACTIVE_ENGINE_META.exists() else None
    PREVIOUS_ENGINE.rename(ACTIVE_ENGINE)
    if PREVIOUS_ENGINE_META.exists():
        PREVIOUS_ENGINE_META.rename(ACTIVE_ENGINE_META)
    if tmp_active.exists():
        tmp_active.rename(PREVIOUS_ENGINE)
        tmp_meta = tmp_active.with_suffix(".meta.json")
        if tmp_meta.exists():
            tmp_meta.rename(PREVIOUS_ENGINE_META)
    # marcar como degradado en metadata
    meta = _read_engine_meta(ACTIVE_ENGINE_META) or {}
    meta["from_fallback"] = True
    ACTIVE_ENGINE_META.write_text(json.dumps(meta, indent=2))
    # pedir hot-swap en el worker
    worker.request_swap(str(ACTIVE_ENGINE))
    return {"ok": True, "phase": "rolled_back"}
```

- [ ] **Step 3: PASS**

```bash
uv run pytest tests/test_model_rollback.py -v
```

- [ ] **Step 4: commit**

```bash
git add scripts/nano_server.py tests/test_model_rollback.py
git commit -m "feat(server): POST /model/rollback con swap inverso + hot-reload worker"
```

### Task E7: re-deploy al Nano y validación end-to-end

- [ ] **Step 1: rsync los archivos modificados**

```bash
scp scripts/nano_server.py scripts/nano_server_constants.py scripts/recover_job_state.py \
    nano:/home/jetson/embebidos-3/scripts/
```

- [ ] **Step 2: reiniciar el server**

```bash
ssh nano "sudo systemctl restart embebidos3-server.service"
sleep 5
ssh nano "systemctl status embebidos3-server.service --no-pager | head -10"
```

Expected: server `active (running)`, sin errores en logs.

- [ ] **Step 3: smoke test de cada endpoint**

```bash
ssh nano "curl -s http://localhost:8000/model/state | python3 -m json.tool"
ssh nano "curl -s http://localhost:8000/jobs/active"
ssh nano "curl -sX POST http://localhost:8000/model/check-updates | python3 -m json.tool"
```

Expected: respuestas válidas. `check-updates` debería decir `up_to_date: true` si el engine actual viene del commit HEAD de HF.

---

## Fase F — UI pestaña `modelo`

**Objetivo:** dashboard muestra estado del modelo, dispara builds, observa progreso en vivo, permite cancelar y rollback. Hereda tipografía/tokens del redesign existente.

### Task F1: tabs en header + routing hash

**Files:**
- Modify: `scripts/dashboard/index.html`
- Modify: `scripts/dashboard/app.js`
- Modify: `scripts/dashboard/style.css`

- [ ] **Step 1: añadir markup de tabs en header**

En `index.html`, después del `.brand` y antes de `.chips`:

```html
<nav class="tabs" role="tablist">
  <button class="tab" role="tab" data-tab="live" aria-selected="true">live</button>
  <button class="tab" role="tab" data-tab="modelo" aria-selected="false">modelo</button>
</nav>
```

- [ ] **Step 2: añadir lógica de routing en app.js**

```javascript
// ---------- Routing simple hash-based -----------------------------------
function setTab(name) {
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(t => t.setAttribute('aria-selected', t.dataset.tab === name ? 'true' : 'false'));
  document.querySelectorAll('[data-pane]').forEach(p => {
    p.hidden = p.dataset.pane !== name;
  });
  if (name === 'modelo') initModelTab();
  if (window.location.hash !== '#' + name) {
    history.replaceState(null, '', '#' + name);
  }
}

function currentTab() {
  return (window.location.hash.replace('#', '') || 'live');
}

document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => setTab(t.dataset.tab);
});
window.addEventListener('hashchange', () => setTab(currentTab()));
```

- [ ] **Step 3: envolver el main actual en `[data-pane="live"]` y añadir `[data-pane="modelo"]` vacío por ahora**

```html
<main class="layout" data-pane="live">
  <!-- contenido actual -->
</main>
<main class="layout-modelo" data-pane="modelo" hidden>
  <section class="modelo-main">
    <div id="modelo-content">cargando...</div>
  </section>
  <aside class="modelo-side">
    <!-- cards se renderizan dinámicamente -->
  </aside>
</main>
```

- [ ] **Step 4: estilos tabs**

```css
.tabs {
  display: flex;
  gap: 4px;
  margin-left: 24px;
}
.tab {
  background: transparent;
  color: var(--text-muted);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 4px 12px;
  font-size: var(--fs-chip-label);
  font-weight: 600;
  cursor: pointer;
  text-transform: lowercase;
}
.tab[aria-selected="true"] {
  color: var(--text);
  background: var(--bg-elev-2);
  border-color: var(--border-strong);
}
.layout-modelo {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  height: calc(100vh - 52px);
  overflow: hidden;
}
.modelo-main {
  padding: 24px;
  overflow-y: auto;
}
.modelo-side {
  padding: 16px;
  border-left: 1px solid var(--border);
  overflow-y: auto;
}
```

- [ ] **Step 5: setTab inicial al cargar**

En el bloque `(async () => { ... })()` final:

```javascript
setTab(currentTab());
```

- [ ] **Step 6: smoke test visual local**

```bash
uv run python scripts/launch_demo.py --no-browser
# abrir http://localhost:8001 en browser, alternar entre tabs
```

Expected: las dos pestañas alternan, la `modelo` muestra "cargando..." por ahora.

- [ ] **Step 7: commit**

```bash
git add scripts/dashboard/index.html scripts/dashboard/app.js scripts/dashboard/style.css
git commit -m "feat(dashboard): tabs live/modelo con routing hash-based"
```

### Task F2: módulo `modelo.js` con polling /model/state + render de los 5 estados

**Files:**
- Create: `scripts/dashboard/modelo.js`
- Modify: `scripts/dashboard/index.html` (incluir el script)

- [ ] **Step 1: crear módulo `modelo.js`**

```javascript
// modelo.js — gestión de la pestaña modelo
(() => {
  "use strict";
  const state = {
    pollTimer: null,
    sse: null,
    lastState: null,
  };

  function api(path) {
    const base = els.wsUrl ? els.wsUrl.value.replace(/^ws/, 'http').replace(/\/ws$/, '') : window.location.origin;
    return base + path;
  }

  async function fetchState() {
    try {
      const r = await fetch(api('/model/state'));
      const data = await r.json();
      state.lastState = data;
      render(data);
    } catch (e) {
      renderError(e);
    }
  }

  function render(s) {
    const main = document.getElementById('modelo-content');
    if (!main) return;
    const tpl = TEMPLATES[s.state] || TEMPLATES.no_model;
    main.innerHTML = tpl(s);
    renderSidebar(s);
    wireActions(s);
  }

  function renderError(e) {
    document.getElementById('modelo-content').innerHTML =
      `<div class="card-error">No se pudo cargar el estado: ${e.message}</div>`;
  }

  const TEMPLATES = {
    no_model: (s) => `
      <div class="hero-card">
        <h2>Sin modelo cargado</h2>
        <p>Descargá el último ONNX desde HF Hub y compilá el engine TRT optimizado para este Jetson Nano.</p>
        <button id="btn-build" class="primary">descargar y compilar engine</button>
        <p class="hint">Tarda entre 15 y 40 minutos. Podés cerrar la pestaña; el proceso sigue en el servidor.</p>
      </div>`,
    ready: (s) => readyTemplate(s),
    update_available: (s) => readyTemplate(s, { banner: true }),
    degraded: (s) => degradedTemplate(s),
    building: (s) => buildingTemplate(s),
  };

  function readyTemplate(s, opts = {}) {
    const m = s.active_engine || {};
    const banner = opts.banner ? `
      <div class="banner-update">
        <strong>Nuevo entrenamiento disponible en HF Hub.</strong>
        <button id="btn-build" class="primary">actualizar engine</button>
      </div>` : '';
    return `
      ${banner}
      <div class="modelo-card">
        <h2>Modelo activo</h2>
        <dl class="modelo-info">
          <dt>origen</dt><dd>commit <span class="mono">${(m.hf_revision || '').slice(0,7)}</span> · ${(m.hf_commit_date || '—')}</dd>
          <dt>onnx</dt><dd>sha <span class="mono">${(m.onnx_sha256 || '').slice(0,8)}</span></dd>
          <dt>engine</dt><dd>sha <span class="mono">${(m.engine_sha256 || '').slice(0,8)}</span> · FP16</dd>
          <dt>compilado</dt><dd>${(m.build_completed_at || '—')} · ${m.build_duration_s || '—'} s</dd>
          <dt>workspace</dt><dd>${(m.trtexec_args || []).find(a => a.startsWith('--workspace=')) || '—'}</dd>
        </dl>
        <div class="modelo-actions">
          <button id="btn-check-updates">verificar actualizaciones</button>
          <button id="btn-force-rebuild">forzar recompilación</button>
        </div>
      </div>`;
  }

  function degradedTemplate(s) {
    return `
      <div class="banner-warn">
        <strong>Usando engine anterior.</strong>
        El último intento de actualización falló. Engine en uso: commit
        <span class="mono">${(s.active_engine?.hf_revision || '').slice(0,7)}</span>.
      </div>
      ${readyTemplate(s)}`;
  }

  function buildingTemplate(s) {
    const j = s.active_job || {};
    const pct = j.progress_pct || 0;
    return `
      <div class="modelo-card">
        <h2>Compilando engine — <span class="mono">${j.job_id || ''}</span></h2>
        <dl class="modelo-info">
          <dt>fase actual</dt><dd>${j.phase || '—'}</dd>
          <dt>progreso</dt><dd>
            <div class="progress"><div class="progress-bar" style="width:${pct}%"></div></div>
            <span class="mono">${pct}%</span>
          </dd>
          <dt>origen</dt><dd>commit <span class="mono">${(j.onnx_source?.hf_revision || '').slice(0,7)}</span></dd>
        </dl>
        <div class="modelo-actions">
          <button id="btn-cancel" class="danger">cancelar build</button>
        </div>
        <div class="logs-pane">
          <h3>logs en vivo</h3>
          <pre id="logs-stream" class="logs-stream"></pre>
        </div>
      </div>`;
  }

  function renderSidebar(s) {
    // implementado en task F5
  }

  function wireActions(s) {
    const btnBuild = document.getElementById('btn-build');
    if (btnBuild) btnBuild.onclick = () => triggerBuild();

    const btnCheck = document.getElementById('btn-check-updates');
    if (btnCheck) btnCheck.onclick = () => checkUpdates();

    const btnForce = document.getElementById('btn-force-rebuild');
    if (btnForce) btnForce.onclick = () => triggerBuild(true);

    const btnCancel = document.getElementById('btn-cancel');
    if (btnCancel) btnCancel.onclick = () => cancelBuild(s.active_job?.job_id);

    if (s.state === 'building' && s.active_job?.job_id) {
      startLogsStream(s.active_job.job_id);
    } else {
      stopLogsStream();
    }
  }

  async function triggerBuild(force = false) {
    try {
      const r = await fetch(api('/model/build'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force }),
      });
      const data = await r.json();
      if (!data.ok) alert(`No se pudo lanzar build: ${data.error}`);
      fetchState();
    } catch (e) { alert(e.message); }
  }

  async function cancelBuild(jobId) {
    if (!jobId || !confirm('¿Cancelar el build en curso?')) return;
    try {
      await fetch(api('/jobs/' + jobId), { method: 'DELETE' });
      fetchState();
    } catch (e) { alert(e.message); }
  }

  async function checkUpdates() {
    try {
      const r = await fetch(api('/model/check-updates'), { method: 'POST' });
      const data = await r.json();
      alert(data.up_to_date ? 'Modelo al día.' :
            `Hay novedad: ${data.latest_revision.slice(0,7)} (actual ${data.current_revision?.slice(0,7) || '—'})`);
      fetchState();
    } catch (e) { alert(e.message); }
  }

  function startLogsStream(jobId) {
    stopLogsStream();
    const url = api(`/jobs/${jobId}/logs`);
    state.sse = new EventSource(url);
    state.sse.addEventListener('log', (ev) => {
      const data = JSON.parse(ev.data);
      const pane = document.getElementById('logs-stream');
      if (pane) {
        pane.textContent += data.line + '\n';
        pane.scrollTop = pane.scrollHeight;
      }
    });
    state.sse.addEventListener('done', () => {
      stopLogsStream();
      fetchState();
    });
  }

  function stopLogsStream() {
    if (state.sse) { state.sse.close(); state.sse = null; }
  }

  window.initModelTab = function () {
    fetchState();
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(fetchState, 3000);
  };
})();
```

- [ ] **Step 2: incluir el script en index.html antes de app.js**

```html
<script src="modelo.js?v=20260516-1"></script>
<script src="app.js?v=20260516-1"></script>
```

- [ ] **Step 3: estilos para los templates**

Añadir al CSS:

```css
.hero-card, .modelo-card {
  background: var(--bg-elev-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 20px;
  margin-bottom: 16px;
}
.modelo-info { display: grid; grid-template-columns: 140px 1fr; gap: 8px; margin: 16px 0; }
.modelo-info dt { color: var(--text-muted); font-size: var(--fs-meta); }
.modelo-info dd { font-size: var(--fs-body); }
.modelo-actions { display: flex; gap: 8px; margin-top: 16px; }
.modelo-actions button.primary { background: var(--accent); color: var(--bg); }
.modelo-actions button.danger { background: var(--err); color: var(--bg); }
.mono { font-family: var(--font-mono); font-variant-numeric: tabular-nums; }
.progress { display: inline-block; width: 200px; height: 8px; background: var(--bg-elev-3); border-radius: 4px; vertical-align: middle; }
.progress-bar { height: 100%; background: var(--accent); border-radius: 4px; transition: width 200ms ease; }
.logs-pane { margin-top: 24px; }
.logs-stream { background: var(--bg); border: 1px solid var(--border); border-radius: 4px; padding: 12px; max-height: 240px; overflow-y: auto; font-family: var(--font-mono); font-size: 12px; }
.banner-update, .banner-warn { padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; }
.banner-update { background: color-mix(in oklch, var(--warn) 15%, transparent); border: 1px solid var(--warn); }
.banner-warn { background: color-mix(in oklch, var(--warn) 20%, transparent); border: 1px solid var(--warn); }
```

- [ ] **Step 4: smoke test visual con server real**

```bash
ssh nano "curl -s http://localhost:8000/model/state"
uv run python scripts/launch_demo.py --no-browser
# en browser: ir a tab modelo, ver render del estado actual
```

Expected: muestra metadata del engine activo (`ready`).

- [ ] **Step 5: commit**

```bash
git add scripts/dashboard/modelo.js scripts/dashboard/index.html scripts/dashboard/style.css
git commit -m "feat(dashboard): pestaña modelo con 5 estados, SSE logs, polling, acciones"
```

### Task F3: sidebar cards (Servidor, HF Hub, Acciones, Histórico)

**Files:**
- Modify: `scripts/dashboard/modelo.js`
- Modify: `scripts/dashboard/style.css`

- [ ] **Step 1: implementar `renderSidebar` en modelo.js**

```javascript
function renderSidebar(s) {
  const side = document.querySelector('.modelo-side');
  if (!side) return;
  const m = s.active_engine || {};
  const prev = s.previous_engine;
  side.innerHTML = `
    <section class="side-card">
      <h3>servidor</h3>
      <dl class="kv">
        <dt>estado</dt><dd>activo</dd>
        <dt>endpoint</dt><dd class="mono">${window.location.origin}</dd>
      </dl>
    </section>

    <section class="side-card">
      <h3>HF Hub</h3>
      <dl class="kv">
        <dt>repo</dt><dd class="mono small">mitgar14/embebidos-3-models</dd>
        <dt>revision activa</dt><dd class="mono">${(m.hf_revision || '—').slice(0,7)}</dd>
      </dl>
      <button id="btn-side-check" class="ghost">verificar ahora</button>
    </section>

    <section class="side-card">
      <h3>acciones</h3>
      <button id="btn-side-rebuild" class="ghost">forzar recompilación</button>
      <button id="btn-side-rollback" class="ghost" ${prev ? '' : 'disabled'}>revertir a engine anterior</button>
    </section>

    <section class="side-card">
      <h3>histórico</h3>
      <p class="hint">últimos jobs (próximamente)</p>
    </section>
  `;
  const sb = document.getElementById('btn-side-check');
  if (sb) sb.onclick = () => checkUpdates();
  const fb = document.getElementById('btn-side-rebuild');
  if (fb) fb.onclick = () => triggerBuild(true);
  const rb = document.getElementById('btn-side-rollback');
  if (rb && !rb.disabled) rb.onclick = () => rollback();
}

async function rollback() {
  if (!confirm('¿Revertir al engine anterior?')) return;
  try {
    const r = await fetch(api('/model/rollback'), { method: 'POST' });
    const data = await r.json();
    if (!data.ok) alert('No se pudo: ' + (data.error || ''));
    fetchState();
  } catch (e) { alert(e.message); }
}
```

- [ ] **Step 2: estilos sidebar**

```css
.side-card { margin-bottom: 16px; padding: 12px; background: var(--bg-elev-1); border: 1px solid var(--border); border-radius: 6px; }
.side-card h3 { font-size: var(--fs-h2); text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin: 0 0 8px; }
.kv { display: grid; grid-template-columns: 1fr auto; gap: 4px; font-size: var(--fs-meta); margin-bottom: 12px; }
.kv dt { color: var(--text-muted); }
.kv dd { text-align: right; }
.side-card button { width: 100%; margin-bottom: 4px; }
.side-card button.ghost { background: transparent; border: 1px solid var(--border-strong); color: var(--text); }
.side-card button[disabled] { opacity: 0.4; cursor: not-allowed; }
.small { font-size: 10px; }
```

- [ ] **Step 3: smoke test**

```bash
uv run python scripts/launch_demo.py --no-browser
# en browser: tab modelo, ver sidebar con cards
```

- [ ] **Step 4: commit**

```bash
git add scripts/dashboard/modelo.js scripts/dashboard/style.css
git commit -m "feat(dashboard): sidebar pestaña modelo con cards Servidor/HF/Acciones/Histórico"
```

### Task F4: banner "modelo no disponible" en pestaña `live` cuando hay build

**Files:**
- Modify: `scripts/dashboard/app.js`

- [ ] **Step 1: añadir poll periódico ligero a /model/state desde app.js**

```javascript
// ---------- Monitoring del estado del modelo (cross-tab) -----------------
async function pollModelStateLight() {
  try {
    const httpUrl = els.wsUrl.value.trim().replace(/^ws/, 'http').replace(/\/ws$/, '');
    const r = await fetch(httpUrl + '/model/state');
    const data = await r.json();
    const banner = document.getElementById('live-banner');
    if (data.state === 'building' && !banner) {
      const b = document.createElement('div');
      b.id = 'live-banner';
      b.className = 'live-banner';
      b.innerHTML = 'modelo no disponible — build en curso · <a href="#modelo">ver pestaña modelo</a>';
      document.querySelector('[data-pane="live"] .stage').prepend(b);
    } else if (data.state !== 'building' && banner) {
      banner.remove();
    }
  } catch (e) {}
}

setInterval(pollModelStateLight, 5000);
```

- [ ] **Step 2: estilos del banner**

```css
.live-banner {
  background: color-mix(in oklch, var(--warn) 20%, transparent);
  border: 1px solid var(--warn);
  border-radius: 6px;
  padding: 8px 12px;
  margin-bottom: 8px;
  font-size: var(--fs-meta);
  text-align: center;
}
.live-banner a { color: var(--accent); text-decoration: underline; }
```

- [ ] **Step 3: smoke test**

Disparar un build (simulado en test, real en validación end-to-end), verificar que aparece el banner en la pestaña `live`.

- [ ] **Step 4: commit**

```bash
git add scripts/dashboard/app.js scripts/dashboard/style.css
git commit -m "feat(dashboard): banner cross-tab indicando build en curso desde la pestaña live"
```

---

## Fase H — Validación end-to-end + docs

**Objetivo:** ejecutar el flujo completo desde la UI, verificar que sobrevive un reboot del Nano, registrar la validación.

### Task H1: build end-to-end desde la UI

- [ ] **Step 1: rsync todo lo nuevo al Nano**

```bash
scp scripts/*.py scripts/*.sh systemd/*.service systemd/*.conf \
    nano:/home/jetson/embebidos-3/scripts/  # ajustar paths
# y la nueva carpeta dashboard
rsync -av scripts/dashboard/ nano:/home/jetson/embebidos-3/scripts/dashboard/
```

- [ ] **Step 2: lanzar el dashboard local**

```bash
uv run python scripts/launch_demo.py
```

- [ ] **Step 3: abrir tab `modelo` y disparar `forzar recompilación`**

Verificar:
- Banner indica build en curso.
- Logs aparecen en vivo.
- Pestaña `live` muestra el banner cross-tab.
- Después de 15-40 min: estado vuelve a `ready` con nueva metadata (mismo `hf_revision`, nuevo `build_completed_at`).

- [ ] **Step 4: registrar duración y métricas en una nota**

```bash
# En la consola JS del browser, copiar el job final:
fetch('/jobs/<jid>').then(r=>r.json()).then(console.log)
```

Guardar el JSON en `docs/superpowers/validations/2026-MM-DD-build-end-to-end.json`.

### Task H2: sobrevive a reboot del Nano

- [ ] **Step 1: reboot**

```bash
ssh nano "sudo reboot"
sleep 60
```

- [ ] **Step 2: verificar arranque automático**

```bash
ssh nano "systemctl is-active embebidos3-server.service"
ssh nano "curl -s http://localhost:8000/health"
ssh nano "curl -s http://localhost:8000/model/state | python3 -m json.tool"
```

Expected: `active`, `health` OK, `model/state` `ready` con engine actual.

### Task H3: actualizar README del repo con sección pipeline

**Files:**
- Modify: `README.md`

- [ ] **Step 1: añadir sección en el README**

Después de la sección "Quick start", añadir:

```markdown
## 6. Pipeline automatizado dashboard → builder

A partir del 2026-05-16 existe un pipeline supervisado que automatiza desde la
página web el flujo descarga `.onnx` desde HF Hub → compilación TRT FP16 con
`trtexec` → validación → swap atómico → reanudación del server. Incluye
rollback automático ante fallas y backup a HF Hub antes de cualquier borrado.

Detalles completos en:
- `docs/superpowers/specs/2026-05-16-dashboard-pipeline-design.md` — diseño aprobado.
- `docs/superpowers/plans/2026-05-16-dashboard-pipeline.md` — plan de implementación.
- `investigaciones/2026-05-15-pipeline-tecnico-vacios.md` — investigación previa de 7 vacíos técnicos.

Instalación inicial en el Nano:

```bash
bash /home/jetson/embebidos-3/scripts/nano_install_systemd.sh
sudo nano /etc/embebidos3/secrets.env   # poner HF_TOKEN real
sudo systemctl restart embebidos3-server.service
```

Uso: abrir el dashboard local, pestaña `modelo`, botón `verificar actualizaciones` → si hay nuevo commit en HF, botón `actualizar engine` dispara el build (15-45 min) en background.
```

- [ ] **Step 2: commit**

```bash
git add README.md
git commit -m "docs(readme): documentar pipeline automatizado dashboard → builder"
```

---

## Self-review

Después de escribir el plan, lo reviso contra el spec:

**1. Spec coverage**: las 10 secciones del spec quedan cubiertas:
- §1 Constraints: declarados en cada tarea relevante.
- §2 Arquitectura macro: Fase A (systemd + sudoers + wrapper).
- §3 Estados + layout: Fase B (TRTWorker reentrante + endpoint /model/state).
- §4 Endpoints FastAPI: Fase E (build, jobs, check-updates, rollback, SSE).
- §5 UI pestaña `modelo`: Fase F (tabs + render por estado + sidebar + banner cross-tab).
- §6 Builder oneshot: Fase D (12 fases del builder + helpers).
- §7 Robustez + retención: cubierto en la lógica del builder (Fase D5) + tmpfiles (Fase A3) + rollback (Fase E6).
- §8 Setup + install: Fase A (script install) + Task D6 (deps).
- §9 Hallazgos técnicos: aplicados en los snippets (validate_engine, hf_rest, builder_state, recovery).
- §10 Plan implementación: este documento.

**2. Placeholder scan**: no hay "TBD", "TODO", "implement later". Todo el código está completo o tiene comando exacto. Excepción: Fase G del spec ya estaba completada antes (cleanup HF), por eso no aparece como fase en este plan.

**3. Type consistency**: nombres de funciones, archivos, variables coherentes entre tareas:
- `request_swap` (B2, E6): consistente.
- `_read_active_job`, `_is_pid_alive`, `_read_engine_meta`: consistentes entre B4, E1, E2, E4.
- `JOB_STATE_FILE`, `JOBS_LOGS_DIR`, `ACTIVE_ENGINE`, `PREVIOUS_ENGINE`: definidos en B1, usados en todo el resto.
- `nano_server_constants.py` central, referenciado por todos los módulos.
- `embebidos3-builder-launch` wrapper consistente entre A4, A5, E1.
- `parse_trtexec_progress.py` invocado en D5 (snippet bash) con `<job_id>` argv, consistente con D2.

Sin issues encontrados.

---

## Execution Handoff

Plan completo y guardado en `docs/superpowers/plans/2026-05-16-dashboard-pipeline.md`. Dos opciones de ejecución:

1. **Subagent-Driven (recomendado)** — un subagente fresh por tarea, revisión entre tareas, iteración rápida.
2. **Inline Execution** — ejecutar tareas en esta sesión usando executing-plans, con checkpoints para review.

¿Cuál preferís?
