# Infraestructura de training — Vast.ai + `uv` dual venv + HF Hub + auto-destroy

**Proyecto:** `embebidos-3` (clasificador de residuos Jetson Nano B01, entrega 2026-05-26).
**Dominio:** infraestructura de ejecución del training. Cómo correr los notebooks `.ipynb` en Vast.ai con tolerancia a cierre de pestaña, dos virtualenvs `uv` aislados, persistencia automática en Hugging Face Hub y auto-destroy de la instancia.
**Documentos hermanos:** [`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md) (D3, D5, D6, D9, D10, D11) · [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) · [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) · [`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md).
**Fecha de cierre:** 2026-05-12.

---

## 1. Resumen ejecutivo

Cuatro decisiones operativas vinculantes de Ronda 5 cubren los cuatro problemas que surgieron al ejecutar training largo en cloud:

| # | Problema | Decisión |
|---|----------|----------|
| **D9** | Cómo ejecutar `.ipynb` toleranto cierre de pestaña sin perder output | Mantener `.ipynb` interactivo + ejecutar con `jupyter nbconvert --execute --inplace` dentro de `tmux` (o `papermill` equivalente). Reemplaza D4 (jupytext). |
| **D10** | Deps de Track A (TF 2.15 + protobuf 3.20.3) y Track B (torch + numpy<2) son irreconciliables | Dos `uv venv` separados (`/opt/venv/tracka` + `/opt/venv/trackb`), kernels `ipykernel` distintos. |
| **D5 + D6** | Cómo persistir checkpoints + logs sin perder nada si el container muere | `CommitScheduler(every=5)` + `upload_folder(run_as_future=True)` al repo HF Hub privado `mitgar14/embebidos-3-models` (D3). TensorBoard hosted en HF (Track A) + W&B nativo (Track B). |
| **D11** | Cómo destruir la instancia Vast.ai al terminar el training sin `--auto-stop` ni `idle shutdown` (GAPs confirmados) | Cron watchdog interno + última celda del notebook que crea `/workspace/embebidos-3/.training_done` y dispara `vastai destroy instance ${VAST_CONTAINERLABEL#C.}`. Reemplaza D7 (`trap EXIT`). |

**Patrón ganador end-to-end:**

1. Usuario lanza `vastai create instance <OFFER_ID> --image ... --jupyter --jupyter-lab --jupyter-dir /workspace --disk 30 --env '-e VAST_API_KEY=... -e HF_TOKEN=... -e WANDB_API_KEY=... -p 8080:8080' --onstart-cmd 'bash bootstrap.sh'`.
2. `bootstrap.sh` crea dos `uv venv`, registra kernels, instala `vastai` CLI, configura JupyterLab para "kernel-survives-disconnect" y registra cron watchdog.
3. Usuario abre JupyterLab → terminal → `tmux new -s training` → `jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=trackb <notebook>` → `Ctrl+B, D` para detach → cierra pestaña.
4. Notebook corre 2–3 h. Outputs se guardan al `.ipynb` en disco; HF Hub recibe checkpoints cada 5 min.
5. Última celda crea `.training_done` → cron watchdog detecta en < 60 s → `vastai destroy instance` → instancia desaparece.

---

## 2. Por qué mantener `.ipynb` interactivo (D9 reemplaza D4)

### 2.1 La decisión renegociada

R4 había definido D4: convertir los `.ipynb` a `.py` con `jupytext --to py:percent` para ejecutar headless con `python train_track_a.py`. R5 renegocia esta decisión a petición explícita del usuario (cita verbatim del HANDOFF §2): *"Necesito negociar [...] correrlo como notebook [...] que su ejecución no se afecte porque cerré la pestaña [...]"*.

### 2.2 Comparativa de cinco estrategias evaluadas

| Estrategia | Output post-disconnect | Requiere pestaña abierta | Pros | Contras |
|------------|------------------------|---------------------------|------|---------|
| JupyterLab + kernel detached "puro" desde UI | Solo lo guardado en `.ipynb` (autosave 120 s) | No | UI familiar, interactividad | **Output entre autosaves NO recuperable** |
| `papermill input.ipynb output.ipynb` | Sí, en `output.ipynb` completo | No | Output completo garantizado | Sin UI durante ejecución; archivo separado |
| `jupyter execute notebook.ipynb` | Sí, in-place | No | Simple, sin deps extra | Menos manejo de errores que papermill |
| **`nbconvert --execute --to notebook --inplace`** | **Sí, in-place** | **No** | **Estándar, distribuido con Jupyter** | Inicialización lenta |
| `nohup jupyter execute ... &` | Sí (background) | No | Sencillo | Sin reintentos ni logs estructurados |

**Conclusión:** `nbconvert --execute --inplace` dentro de `tmux` es el **único patrón** que combina:

1. Preservación del `.ipynb` con outputs incrustados navegable desde JupyterLab al reconectar.
2. Supervivencia del kernel al cierre de pestaña (tmux mantiene el proceso vivo).
3. Recuperación completa del output al volver horas después.

`jupytext --to py:percent` no satisface (1) porque el `.py` resultante no captura outputs en el `.ipynb` original (queda solo en stdout del proceso). Si se quiere ver el output al reconectar, hay que redirigir a archivo y abrir desde otra herramienta — fricción innecesaria.

### 2.3 Limitación crítica de "kernel detached puro"

> **Al reconectar a JupyterLab tras una desconexión, el output de celdas ya ejecutadas NO se recupera automáticamente del stream en vivo. Solo el output guardado en el `.ipynb` es visible.**

El kernel Python es un proceso independiente del WebSocket. Al cerrar el navegador, el proceso continúa; lo que se pierde es solo el stream IOPub. Sin `nbconvert`, los outputs emitidos entre el último autosave (cada 120 s) y la desconexión se pierden completamente.

Por eso `tmux + nbconvert --execute --inplace` es obligatorio: `nbconvert` escribe directamente al `.ipynb` en disco, no depende de IOPub stream.

---

## 3. Patrón `tmux + nbconvert --execute --inplace` paso a paso

### 3.1 Config crítica de JupyterLab (`/root/.jupyter/jupyter_server_config.py`)

```python
# /root/.jupyter/jupyter_server_config.py
# Aplica al lanzar JupyterLab desde Vast.ai (D9 — kernel-survives-disconnect)

# 1. Kernel sobrevive cierre de pestaña / disconnect
c.MappingKernelManager.cull_idle_timeout = 0       # 0 deshabilita el culler completamente
c.MappingKernelManager.cull_busy = False           # no cull si está busy
c.MappingKernelManager.cull_connected = False      # no cull si hay clients conectados
c.ServerApp.shutdown_no_activity_timeout = 0       # server no se apaga por inactividad

# 2. Sin truncado de output verboso (training loops loggean cada batch)
c.ZMQChannelsWebsocketConnection.iopub_msg_rate_limit = 0          # default 1000 msg/s
c.ZMQChannelsWebsocketConnection.iopub_data_rate_limit = 10_000_000  # 10 MB/s (default 1 MB/s)
```

**Justificación:**

- *"To disable kernel culling entirely, set `cull_idle_timeout` to 0 or lower. This ensures kernels survive browser disconnects."* — DeepWiki sobre `jupyter-server/jupyter_server`.
- *"Kernel culling is initialized lazily when the first kernel starts, so setting `cull_idle_timeout=0` prevents the culler from ever starting."* — Idem.
- *"These [rate limit settings] were deprecated in `ServerApp` in favor of configuring them directly on `ZMQChannelsWebsocketConnection`."* — Idem.

Sin estos ajustes, un training que loggea métricas cada batch genera `[stdout truncated]` en la celda y el kernel puede ser asesinado por el culler aunque el proceso siga vivo.

### 3.2 Comando canónico de ejecución (Track B ejemplo)

```bash
# Dentro del terminal de JupyterLab (o vía SSH)

# 1. Crear sesión tmux detached
tmux new-session -d -s training

# 2. Enviar comando de ejecución
tmux send-keys -t training '
  source /opt/venv/trackb/bin/activate
  cd /workspace/embebidos-3
  jupyter nbconvert --to notebook --execute --inplace \
    notebooks/train_track_b_yolov8.ipynb \
    --ExecutePreprocessor.timeout=10800 \
    --ExecutePreprocessor.kernel_name=trackb \
    2>&1 | tee /workspace/embebidos-3/logs/train_track_b.log
  touch /workspace/embebidos-3/.training_done
' Enter

# 3. Verificar que el proceso sigue vivo
tmux ls
# Output esperado: training: 1 windows (created ...) [80x24]

# 4. (Opcional) inspeccionar progreso
tmux attach-session -t training
# Ctrl+B, D para detach sin matar el proceso
```

### 3.3 Flags importantes de `nbconvert`

| Flag | Valor | Razón |
|------|-------|-------|
| `--to notebook` | (fijo) | Mantener formato `.ipynb` |
| `--execute` | (fijo) | Ejecutar todas las celdas |
| `--inplace` | (fijo) | Escribir outputs al mismo archivo `.ipynb` |
| `--ExecutePreprocessor.timeout` | `10800` | 3 h por celda. Default 30 s (rompe para training). |
| `--ExecutePreprocessor.kernel_name` | `tracka` o `trackb` | Forzar kernel custom (sino usa kernel por default, falla con TF/torch específico) |
| `--allow-errors` | (NO usar) | Default `false` aborta al primer error: queremos eso |

**Alternativa con `papermill`:** sintaxis similar pero con archivo separado y parametrización inyectable. `papermill input.ipynb output.ipynb -p PARAM_NAME PARAM_VALUE`. Útil si se quiere ejecutar múltiples variantes. Para nuestro caso (un run por track) `nbconvert` es suficiente y no añade deps extra.

### 3.4 Trampas conocidas y mitigaciones

| # | Trampa | Mitigación |
|---|--------|-----------|
| T1 | `"Notebook is too large to be saved"` cuando el output acumulado supera ~25 MB | `%%capture` en celdas de instalación; `IPython.display.clear_output(wait=True)` antes de loops verbosos; log verboso a archivo con `tqdm.write()` en vez de `print()`; mostrar solo métricas resumidas con `tqdm` |
| T2 | Autosave conflictivo si se abre el mismo `.ipynb` en dos pestañas | Una sola sesión activa por notebook (en general no se abrirá mientras está corriendo `nbconvert`) |
| T3 | WebSocket ping/pong timeout: el proxy de Vast.ai puede cerrar conexiones inactivas | `jupyter lab --ServerApp.tornado_settings='{"websocket_ping_interval": 30000}'` o `--ping-interval 30` al lanzar |
| T4 | Buffer overflow del frontend con `iopub_data_rate_limit=0` | El navegador se ralentiza con miles de líneas. Log verboso a archivo; mostrar pocos prints |
| T5 | `nbconvert` falla silenciosamente si kernel no existe | Pre-validar con `jupyter kernelspec list` antes; especificar `--ExecutePreprocessor.kernel_name=<tracka\|trackb>` explícito (issue R9 del HANDOFF) |
| T6 | `tmux` no preinstalado en imagen vastai | `apt-get install -y tmux` en bootstrap |
| T7 | `nbconvert` no respeta `cull_idle_timeout=0` si la config no se aplicó antes de lanzar JupyterLab | Reiniciar JupyterLab tras escribir `jupyter_server_config.py`: `supervisorctl restart jupyter` |

---

## 4. `uv` dual venv (D10)

### 4.1 Por qué dos virtualenvs separados

Las dependencias son intrínsecamente incompatibles:

- **Track A** requiere TF 2.15 + `protobuf==3.20.3` + Pillow 10.4 + numpy 1.26 + grpcio-tools 1.64.1.
- **Track B** requiere PyTorch 2.1+cu121 + Ultralytics 8.4.46 + `numpy<2.0` (deps ultralytics) + onnxslim.

El solver de `pip`/`uv` no encuentra una resolución única porque `protobuf 3.20.3` (pin Track A) entra en conflicto con deps transitivas de `torch 2.1+cu121` que arrastran `protobuf>=4`. Y `numpy<2.0` de Ultralytics colisiona con TF 2.15 que pide `numpy==1.26.4` específicamente.

### 4.2 Tres opciones evaluadas

| Estrategia | Lockfile | Aislamiento | Complejidad | Recomendado |
|------------|----------|-------------|-------------|-------------|
| **Opción 1**: monorepo único con `[project.optional-dependencies]` + `[tool.uv] conflicts` + `[tool.uv.sources]` por extra | Único, compartido | Mismo venv, mismo proceso | Alta | ❌ Overhead para proyecto académico; deps incompatibles fundamentalmente |
| **Opción 2**: dos `pyproject.toml` separados invocados con `uv sync --project <archivo>` | Independiente | Venvs separados | Media | ❌ **GAP:** `--project` acepta directorios, no archivos. No soportado |
| **Opción 3**: dos `uv venv` separados con `uv pip install` | Sin lockfile (deps explícitas en bootstrap) | Procesos completamente aislados | Baja | ✅ **ELEGIDA (D10)** |

### 4.3 Capacidades de uv (referencia)

- **Workspaces** (`[tool.uv.workspace]`): inspirado en Cargo, `uv.lock` compartido — [docs.astral.sh/uv/concepts/projects/workspaces/](https://docs.astral.sh/uv/concepts/projects/workspaces/).
- **Dependency groups** (`[dependency-groups]`): PEP 735, soportado desde uv 0.5+ — [docs.astral.sh/uv/concepts/projects/dependencies/](https://docs.astral.sh/uv/concepts/projects/dependencies/).
- **Per-package index** con `[tool.uv.sources]` + `[[tool.uv.index]] explicit = true`:
  > *"An index can be marked as `explicit = true` to ensure it's only used for packages explicitly pinned to it in `tool.uv.sources`."* — [docs.astral.sh/uv/guides/integration/pytorch/](https://docs.astral.sh/uv/guides/integration/pytorch/).
- **Conflicts entre extras**: declaración explícita de incompatibilidad:
  > *"This tells `uv` to resolve them separately, preventing both from being installed simultaneously."* — DeepWiki `astral-sh/uv`.
- **`uv pip install` con `--index-url`** (caso PyTorch):
  > *"To use the same workflow with uv, replace `pip3` with `uv pip`: `$ uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`"* — [docs.astral.sh/uv/guides/integration/pytorch/](https://docs.astral.sh/uv/guides/integration/pytorch/).

### 4.4 Aislamiento de runtime entre TF 2.15 y torch 2.1+cu121

**No hay conflicto cuando se ejecutan en kernels separados.** JupyterLab abre un proceso Python independiente por cada kernel. `libcudart`, `libcudnn`, `libnvinfer` se cargan independientemente por cada proceso. El JupyterLab server en sí no carga ninguna de estas librerías — solo gestiona los kernels vía ZeroMQ.

Esto incluye:

- `libcudart.so.10.2` (TF 2.5 path, irrelevante aquí; en Vast.ai usamos CUDA 12.4) y `libcudart.so.12.4` (4090 driver).
- `libcudnn.so.8.9` (CUDA 12.4) — single version, ambos tracks lo usan.
- `libnvinfer.so.10` (CUDA 12.4 TRT 10) — para Track B en validación local (aunque la validación canónica de D13 usa Docker NGC con TRT 8.2.1).

### 4.5 Comandos canónicos de creación de venvs

```bash
# ============== Track A — venv /opt/venv/tracka ==============
uv venv /opt/venv/tracka --python 3.10
source /opt/venv/tracka/bin/activate

# Stack core (sin --no-deps, deja que uv resuelva)
uv pip install tensorflow==2.15.0 \
              tf-models-official==2.15.0 \
              "tensorflow-model-optimization>=0.7.5,<0.8.0" \
              "numpy==1.26.4" \
              "protobuf==3.20.3" \
              "Pillow==10.4.0" \
              "opencv-python-headless==4.10.0.84" \
              "pycocotools==2.0.7" \
              "lvis==0.5.3" \
              "tensorflow-addons==0.23.0" \
              "tensorflow-text==2.15.0" \
              grpcio-tools==1.64.1 \
              huggingface_hub ipykernel

# Registrar kernel
python -m ipykernel install --user --name tracka \
  --display-name "Track A (TF 2.15)"

deactivate

# ============== Track B — venv /opt/venv/trackb ==============
uv venv /opt/venv/trackb --python 3.10
source /opt/venv/trackb/bin/activate

# Defensa contra NumPy 2.x ANTES de ultralytics (issue #22346)
uv pip install "numpy<2.0"

# PyTorch CUDA 12.1 vía index URL
uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Ultralytics + ONNX stack + Roboflow
uv pip install "ultralytics>=8.4.46,<8.5" \
              "onnxslim>=0.1.82" \
              "onnx>=1.16,<1.18" \
              "onnxruntime>=1.18,<1.21" \
              "roboflow>=1.3.6,<1.4" \
              wandb huggingface_hub ipykernel

# Registrar kernel
python -m ipykernel install --user --name trackb \
  --display-name "Track B (YOLOv8)"

deactivate

# ============== Validación de kernels ==============
jupyter kernelspec list
# Esperado:
#   tracka     ~/.local/share/jupyter/kernels/tracka
#   trackb     ~/.local/share/jupyter/kernels/trackb
```

### 4.6 `kernel.json` resultante (referencia)

Tras `python -m ipykernel install --user --name trackb`, el archivo `~/.local/share/jupyter/kernels/trackb/kernel.json` contiene:

```json
{
 "argv": [
  "/opt/venv/trackb/bin/python",
  "-Xfrozen_modules=off",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Track B (YOLOv8)",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```

El campo `argv[0]` debe apuntar a **`/opt/venv/trackb/bin/python`**, no a `/usr/bin/python3`. Si apunta al sistema, el kernel cargará deps del sistema (no del venv) y romperá con `ModuleNotFoundError: ultralytics`.

### 4.7 Variante con detección automática de backend CUDA (no usada aquí)

`uv` soporta `UV_TORCH_BACKEND=cu121` para autodetección:

```bash
UV_TORCH_BACKEND=cu121 uv pip install torch==2.1.0 torchvision==0.16.0
```

No la usamos porque conocemos el backend de antemano (`cu121` para 4090 con driver CUDA 12.4 compatible).

### 4.8 Por qué Python 3.10 (no 3.11 o 3.12)

- **TF 2.15.0 cp310** existe en PyPI: `tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (2023-11-14).
- **TF 2.15.0 cp311 y cp312 NO existen**: TF 2.15 fue el último con wheels cp310 antes de saltar a cp312 en TF 2.16.
- **Ultralytics 8.4.46** soporta Python ≥ 3.8 hasta 3.12; 3.10 está en el sweet spot maduro.
- **ipykernel** sin issues conocidos en Py 3.10.

El container `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` ya viene con Python 3.10 base. **Coincidencia perfecta.**

---

## 5. HF Hub persistence patterns (D3, D5, D6)

### 5.1 Estructura del repo `mitgar14/embebidos-3-models` (D3)

Repo **privado** ya creado en HF Hub con la siguiente estructura:

```
mitgar14/embebidos-3-models/
├── README.md                  # generado en setup
├── track_a/
│   ├── runs/                  # logs intermedios y tfevents
│   │   └── .gitkeep
│   ├── checkpoints/           # ckpt-N (TF OD API)
│   │   └── .gitkeep
│   ├── exports/               # detect_int8.tflite, manifest.json, pipeline.config
│   │   └── .gitkeep
│   └── logs/                  # bootstrap.log, train_track_a.log
│       └── .gitkeep
└── track_b/
    ├── runs/                  # logs W&B mirror si necesario
    │   └── .gitkeep
    ├── checkpoints/           # best.pt, last.pt
    │   └── .gitkeep
    ├── exports/               # best.onnx, manifest.json
    │   └── .gitkeep
    └── logs/                  # bootstrap.log, train_track_b.log
        └── .gitkeep
```

**Free tier 2026:** 100 GB privado total, sin límite de repos privados, 500 GB max por archivo, sin cap de bandwidth. Xet storage por default desde mayo 2025 (dedup chunks). Artefactos totales del proyecto ~250–450 MB.

### 5.2 `CommitScheduler` cada 5 min (D5)

`CommitScheduler` es el patrón canónico para push automático periódico sin bloquear el kernel:

```python
from huggingface_hub import CommitScheduler
from pathlib import Path

# Track A
checkpoints_dir = Path("/workspace/embebidos-3/track_a/checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

scheduler_a = CommitScheduler(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path=str(checkpoints_dir),
    path_in_repo="track_a/checkpoints",
    every=5,                # minutos
    private=True,
    squash_history=False,   # mantener historial para rollback
    token=None,             # usa ~/.cache/huggingface/token
)

# Lo mismo para logs/tfevents
scheduler_a_logs = CommitScheduler(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_a/logs",
    path_in_repo="track_a/logs",
    every=5,
    private=True,
)
```

**Cómo funciona internamente:**

- Inicia un thread daemon que cada `every` minutos hace `snapshot` de `folder_path`.
- Calcula diff contra el último commit en `path_in_repo` del repo.
- Si hay cambios, hace `commit` con mensaje autogenerado (`Update from CommitScheduler [...]`).
- Si no hay cambios, no hace commit (no contamina historial).
- Resistente a errores de red: reintenta en el próximo intervalo.

### 5.3 `upload_folder(run_as_future=True)` para checkpoints finales

Al terminar el training, push asíncrono de exports + manifests:

```python
from huggingface_hub import HfApi
api = HfApi()

# Push asíncrono — no bloquea
future = api.upload_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_a/exports",
    path_in_repo="track_a/exports",
    commit_message="Track A — final exports (model.tflite + manifest)",
    run_as_future=True,
)

# ... otras tareas ...

# Bloquear solo cuando se necesite
result = future.result(timeout=600)  # 10 min max
print(f"Upload completo: {result}")
```

**`run_as_future=True`:** devuelve un `concurrent.futures.Future`. Permite ejecutar otras tareas en paralelo mientras el upload corre en background.

### 5.4 `upload_large_folder` para casos resumibles (si fuera necesario)

Patrón resumible para artefactos grandes (no aplica a embebidos-3 con ~50 MB total por track, pero documentado por completitud):

```python
api.upload_large_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_b/exports",
    path_in_repo="track_b/exports",
    private=True,
    multi_commits=True,                # chunks separados
    multi_commits_verbose=True,
    create_pr=False,
)
```

Reanuda automáticamente si la red se cae a mitad de la subida.

### 5.5 TensorBoard hosted en HF Hub (D6 Track A)

HF Hub detecta automáticamente archivos `tfevents.*` y monta una instancia de TensorBoard gratis. Patrón:

```python
from huggingface_hub import HFSummaryWriter

writer = HFSummaryWriter(
    repo_id="mitgar14/embebidos-3-models",
    logdir="/workspace/embebidos-3/track_a/logs",
    commit_every=5,         # minutos
    repo_private=True,
)

# Uso idéntico a torch.utils.tensorboard.SummaryWriter
writer.add_scalar("train/loss", loss, step)
writer.add_scalar("train/mAP", map_value, step)
writer.add_image("samples/predictions", img, step)
```

URL de TensorBoard hosted: `https://huggingface.co/mitgar14/embebidos-3-models/tensorboard`. La detección y boot del board son automáticos al primer commit con `tfevents`.

### 5.6 W&B nativo (D6 Track B)

Ultralytics tiene integración nativa con Weights & Biases:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=416,
    batch=32,
    project="embebidos-3",
    name="track_b_yolov8n",
    wandb=True,             # ← magia: crea W&B run, loggea métricas, sube samples
)
```

**Variables requeridas:** `WANDB_API_KEY` (pasada al container vía `--env`).

**Free tier W&B 2026:** sin cap conocido en runs para proyectos personales. Confirmado en [`wandb.ai/site/pricing`](https://wandb.ai/site/pricing).

### 5.7 Snippet completo Track A (Bootstrap CommitScheduler + HFSummaryWriter)

```python
import os
from pathlib import Path
from huggingface_hub import CommitScheduler, HFSummaryWriter

REPO_ID = "mitgar14/embebidos-3-models"
ROOT = Path("/workspace/embebidos-3")

# Asegurar directorios
for sub in ["track_a/checkpoints", "track_a/exports", "track_a/logs"]:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

# Scheduler para checkpoints (cada 5 min)
ckpt_scheduler = CommitScheduler(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=str(ROOT / "track_a/checkpoints"),
    path_in_repo="track_a/checkpoints",
    every=5,
    private=True,
)

# Scheduler para logs / tfevents (cada 5 min)
logs_scheduler = CommitScheduler(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=str(ROOT / "track_a/logs"),
    path_in_repo="track_a/logs",
    every=5,
    private=True,
)

# TensorBoard writer
tb_writer = HFSummaryWriter(
    repo_id=REPO_ID,
    logdir=str(ROOT / "track_a/logs"),
    commit_every=5,
    repo_private=True,
)

print(f"✅ HF Hub persistence activa para {REPO_ID}")
print(f"   Checkpoints: cada 5 min desde {ROOT / 'track_a/checkpoints'}")
print(f"   Logs/TB:     cada 5 min desde {ROOT / 'track_a/logs'}")
```

---

## 6. Auto-destroy: cron watchdog + última celda (D11)

### 6.1 GAPs confirmados de Vast.ai (motivación de D11)

| GAP | Evidencia |
|-----|-----------|
| **`vastai create instance` NO tiene `--auto-stop`, `--max-runtime`, `--idle-timeout`** | Revisión de [`vast-ai/vast-python`](https://github.com/vast-ai/vast-python) source `vast.py` y de [docs.vast.ai/cli/reference/create-instance](https://docs.vast.ai/cli/reference/create-instance) |
| **Vast.ai NO tiene "Idle Shutdown" automático** basado en utilización de GPU | La doc no menciona ningún mecanismo de shutdown por inactividad. Vast.ai cobra por segundo de instancia activa pero no detiene automáticamente |
| **Flags `--end_date`, `--day`, `--hour` pertenecen a `add_scheduled_job`** (jobs programados), no a `create instance` | Verificación de subcomandos |

Quote de la doc oficial: *"Every offer has a maximum rental duration. When you rent an instance, the offer end date at the time of rental becomes your rental end date, the date your instance will run until."* — [docs.vast.ai/guides/reference/faq/instances](https://docs.vast.ai/guides/reference/faq/instances).

### 6.2 Patrón canónico de auto-destroy desde el container

La FAQ oficial documenta el patrón:

> *"A special instance API key is pre-installed. Install the CLI and use it: `pip install vastai` / `vastai stop instance $CONTAINER_ID`"*
> — [docs.vast.ai/guides/reference/faq/instances](https://docs.vast.ai/guides/reference/faq/instances)

El `$CONTAINER_ID` está disponible como env var `VAST_CONTAINERLABEL`, formato `C.<id>` (extraer con `${VAST_CONTAINERLABEL#C.}`).

### 6.3 Por qué NO `trap EXIT` (D7 reemplazada)

`trap EXIT` requiere un proceso bash longevo. Al ejecutar el notebook como kernel detached con `nbconvert --execute --inplace` dentro de `tmux`, el bash inicial (entrypoint del container) muere temprano y el `trap` se dispara antes de que termine el training. El proceso del kernel sobrevive a `trap`, pero el shutdown nunca se dispara.

### 6.4 Patrón de tres componentes

#### Componente 1 — Cron watchdog instalado en bootstrap

```bash
# Registrar cron job que cada minuto verifica el archivo señal
echo "* * * * * test -f /workspace/embebidos-3/.training_done && \
  vastai destroy instance \${VAST_CONTAINERLABEL#C.} 2>&1 | \
  tee -a /workspace/embebidos-3/logs/watchdog.log" | crontab -

# Verificar
crontab -l
```

**Frecuencia 1 min** es suficiente para nuestro caso (latencia tope 60 s). Para latencia menor, podría usar systemd timer con `OnUnitActiveSec=10s`, pero overhead innecesario.

#### Componente 2 — Última celda del notebook

```python
# Última celda — auto-destroy señaling + plan B inmediato
import os
import subprocess
import time
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()

# 1. Upload artefactos finales (espera a que termine, evitar pérdida)
print("[finalize] Subiendo exports a HF Hub...")
api.upload_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_b/exports",
    path_in_repo="track_b/exports",
    commit_message="Track B — final exports",
    run_as_future=False,  # bloqueante: necesitamos confirmación antes de destroy
)
print("[finalize] Upload completado.")

# 2. Marcar como completo (cron watchdog detecta en < 60 s)
Path("/workspace/embebidos-3/.training_done").touch()
print(f"[finalize] {time.strftime('%H:%M:%S')} Training done. Watchdog disparará destroy.")

# 3. Plan B inmediato (si watchdog falla): destruir desde el notebook
container_id = os.environ.get("VAST_CONTAINERLABEL", "").lstrip("C.")
if container_id:
    print(f"[finalize] Plan B: vastai destroy instance {container_id}")
    subprocess.run(["vastai", "destroy", "instance", container_id], check=False)
else:
    print("[finalize] VAST_CONTAINERLABEL no presente; depende del watchdog")
```

#### Componente 3 — Variable `$VAST_CONTAINERLABEL`

Disponible automáticamente como env var del container (inyectada por Vast.ai al crear instancia). Formato:

```
VAST_CONTAINERLABEL=C.12345678
```

Extracción: `${VAST_CONTAINERLABEL#C.}` → `12345678`.

### 6.5 API key dentro del container

Precedencia en `vast.py` (verificada en código fuente):

```
--api-key <KEY>  >  $VAST_API_KEY (env var)  >  ~/.config/vastai/vast_api_key (archivo)
```

**Inyección al crear instancia (Opción A — env var):**

```bash
vastai create instance <OFFER_ID> \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --env '-e VAST_API_KEY=<key> -e HF_TOKEN=<token> -e WANDB_API_KEY=<key> -e ROBOFLOW_API_KEY=<key> -p 8080:8080' \
  --jupyter --jupyter-lab \
  --jupyter-dir /workspace \
  --disk 30 \
  --onstart-cmd 'bash /workspace/embebidos-3/scripts/bootstrap.sh'
```

**Seguridad:** la API key queda visible en logs del container y en variables de entorno del usuario root. Para minimizar exposición, **Opción B — archivo:**

```bash
# En onstart, antes de cualquier vastai destroy:
pip install vastai
vastai set api-key "$VAST_API_KEY"
unset VAST_API_KEY
# Ahora el key vive en ~/.config/vastai/vast_api_key con permisos 600
```

### 6.6 HF Hub webhook → AWS Lambda → vastai destroy (descartado)

**No vale la pena para proyecto académico.** La cadena HF webhook → AWS Lambda/GitHub Action → `vastai destroy` añade tres puntos de fallo:

1. HF Hub webhook puede fallar / atrasarse.
2. Lambda cold start ~3 s.
3. Permisos cross-account.

Para ahorrar US$0,05 en un saldo de US$1,72 no compensa. El watchdog interno es suficiente.

---

## 7. Script `bootstrap.sh` completo

Archivo final: `scripts/bootstrap.sh`. **Crítico:** debe pushearse con line endings `LF`, no `CRLF` (ver gotcha #28 de `compatibilidad-stack-cloud-jetson.md`).

```bash
#!/usr/bin/env bash
set -euo pipefail

# scripts/bootstrap.sh — embebidos-3 Vast.ai entrypoint
# Invocado por --onstart-cmd al crear instancia Vast.ai.

LOG=/workspace/embebidos-3/logs/bootstrap.log
mkdir -p /workspace/embebidos-3/logs
exec > >(tee -a "$LOG") 2>&1

echo "==> [$(date -Iseconds)] Bootstrap iniciado"

# ============ 1. Variables requeridas ============
: "${HF_TOKEN:?Variable HF_TOKEN requerida (pasar con --env -e HF_TOKEN=...)}"
: "${WANDB_API_KEY:?Variable WANDB_API_KEY requerida}"
: "${VAST_API_KEY:?Variable VAST_API_KEY requerida}"
: "${ROBOFLOW_API_KEY:?Variable ROBOFLOW_API_KEY requerida}"

# ============ 2. Sistema y utilidades ============
apt-get update -qq
apt-get install -y -qq tmux cron curl git ca-certificates

# ============ 3. Clonar repo (si no está montado) ============
if [ ! -d /workspace/embebidos-3/.git ]; then
  cd /workspace
  # Token GitHub (PAT con scope repo) inyectado para repo privado
  git clone "https://${GITHUB_TOKEN:-}@github.com/mitgar14/embebidos-3.git" embebidos-3 || \
    git clone https://github.com/mitgar14/embebidos-3.git embebidos-3
fi
cd /workspace/embebidos-3

# ============ 4. Instalar uv ============
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ============ 5. venv Track A ============
echo "==> Creando /opt/venv/tracka..."
uv venv /opt/venv/tracka --python 3.10
# shellcheck disable=SC1091
source /opt/venv/tracka/bin/activate
uv pip install --quiet \
  tensorflow==2.15.0 \
  tf-models-official==2.15.0 \
  "tensorflow-model-optimization>=0.7.5,<0.8.0" \
  "numpy==1.26.4" \
  "protobuf==3.20.3" \
  "Pillow==10.4.0" \
  "opencv-python-headless==4.10.0.84" \
  "pycocotools==2.0.7" \
  "lvis==0.5.3" \
  "tensorflow-addons==0.23.0" \
  "tensorflow-text==2.15.0" \
  grpcio-tools==1.64.1 \
  huggingface_hub ipykernel

python -m ipykernel install --user --name tracka --display-name "Track A (TF 2.15)"

# Clonar TF Models pinned SHA
if [ ! -d /workspace/tf_models ]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/tensorflow/models.git /workspace/tf_models
  cd /workspace/tf_models
  git checkout 9cafa3d150
  test -d research/object_detection || { echo "ERROR: research/ no existe"; exit 1; }
fi

# Compilar protos (grpcio-tools)
cd /workspace/tf_models/research
python -m grpc_tools.protoc \
  --python_out=. \
  --proto_path=. \
  object_detection/protos/*.proto

# Instalar OD API sin deps
cp object_detection/packages/tf2/setup.py .
pip install --quiet --no-deps -e .

# Re-pin defensivo
pip install --quiet --force-reinstall --no-deps "Pillow==10.4.0" "protobuf==3.20.3"

deactivate
cd /workspace/embebidos-3

# ============ 6. venv Track B ============
echo "==> Creando /opt/venv/trackb..."
uv venv /opt/venv/trackb --python 3.10
# shellcheck disable=SC1091
source /opt/venv/trackb/bin/activate

# Defensa NumPy ANTES de ultralytics
uv pip install --quiet "numpy<2.0"

# PyTorch CUDA 12.1
uv pip install --quiet torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install --quiet \
  "ultralytics>=8.4.46,<8.5" \
  "onnxslim>=0.1.82" \
  "onnx>=1.16,<1.18" \
  "onnxruntime>=1.18,<1.21" \
  "roboflow>=1.3.6,<1.4" \
  wandb huggingface_hub ipykernel

python -m ipykernel install --user --name trackb --display-name "Track B (YOLOv8)"
deactivate

# ============ 7. Vast.ai CLI + cron watchdog ============
pip install --quiet vastai

# Guardar API key en archivo (más seguro que env var persistente)
vastai set api-key "$VAST_API_KEY"

# Registrar cron watchdog (cada 1 min)
service cron start || cron
echo "* * * * * test -f /workspace/embebidos-3/.training_done && \
  vastai destroy instance \${VAST_CONTAINERLABEL#C.} 2>&1 | \
  tee -a /workspace/embebidos-3/logs/watchdog.log" | crontab -

crontab -l

# ============ 8. Config JupyterLab persistente ============
mkdir -p /root/.jupyter
cat > /root/.jupyter/jupyter_server_config.py <<'PYEOF'
c.MappingKernelManager.cull_idle_timeout = 0
c.MappingKernelManager.cull_busy = False
c.MappingKernelManager.cull_connected = False
c.ServerApp.shutdown_no_activity_timeout = 0
c.ZMQChannelsWebsocketConnection.iopub_msg_rate_limit = 0
c.ZMQChannelsWebsocketConnection.iopub_data_rate_limit = 10_000_000
PYEOF

# ============ 9. Variables env permanentes (para sesiones SSH/tmux) ============
{
  echo "export HF_TOKEN='${HF_TOKEN}'"
  echo "export WANDB_API_KEY='${WANDB_API_KEY}'"
  echo "export ROBOFLOW_API_KEY='${ROBOFLOW_API_KEY}'"
  # NO exportar VAST_API_KEY (ya está en ~/.config/vastai/vast_api_key)
} >> /root/.bashrc

# Autenticar HF CLI (escribe ~/.cache/huggingface/token)
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# ============ 10. Reiniciar JupyterLab ============
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart jupyter || echo "==> JupyterLab no controlado por supervisor; reinicia desde la UI"
else
  echo "==> supervisorctl no disponible; reinicia JupyterLab desde la UI de Vast.ai para aplicar config"
fi

echo "==> [$(date -Iseconds)] Bootstrap completo."
echo "    Kernels registrados: tracka (TF 2.15), trackb (YOLOv8)."
echo "    HF Hub repo: mitgar14/embebidos-3-models."
echo "    Watchdog cron activo (frecuencia 1 min)."
echo "    Próximo paso: tmux new -s training; jupyter nbconvert --execute --inplace ..."
```

**Patrón de uso del usuario** (flujo end-to-end):

1. **Local:** `git push` con el notebook actualizado al repo `mitgar14/embebidos-3`.
2. **CLI Vast.ai:**
   ```bash
   vastai create instance <OFFER_ID> \
     --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
     --env '-e VAST_API_KEY=... -e HF_TOKEN=... -e WANDB_API_KEY=... -e ROBOFLOW_API_KEY=... -e GITHUB_TOKEN=... -p 8080:8080' \
     --jupyter --jupyter-lab \
     --jupyter-dir /workspace \
     --disk 30 \
     --onstart-cmd 'bash -c "curl -sSL https://raw.githubusercontent.com/mitgar14/embebidos-3/main/scripts/bootstrap.sh | bash"'
   ```
3. **Browser:** abrir la URL de JupyterLab que entrega Vast.ai. Confirmar que en `Kernel > Change Kernel` aparecen `Track A (TF 2.15)` y `Track B (YOLOv8)`.
4. **Terminal en JupyterLab:**
   ```bash
   tmux new -s training
   jupyter nbconvert --to notebook --execute --inplace \
     notebooks/train_track_b_yolov8.ipynb \
     --ExecutePreprocessor.timeout=10800 \
     --ExecutePreprocessor.kernel_name=trackb
   # Ctrl+B, D para detach
   ```
5. **Usuario cierra pestaña.** El proceso `nbconvert` sigue ejecutando en `tmux`. El kernel `trackb` sobrevive porque `cull_idle_timeout=0`.
6. **2 horas después:** usuario reabre la URL de JupyterLab. Abre `train_track_b_yolov8.ipynb`. Output completo está guardado en disco.
7. **Última celda** crea `/workspace/embebidos-3/.training_done`. Cron watchdog detecta en < 60 s. `vastai destroy` dispara. Instancia desaparece.

---

## 8. Comandos `vastai create instance` (referencia)

### 8.1 Flags relevantes

| Flag CLI | Campo API | Descripción |
|----------|-----------|-------------|
| `--image <tag>` | `image_uuid` | Tag Docker (`vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`) |
| `--jupyter` | `runtype: jupyter` | Activa launch mode jupyter (abre puerto 8080 + 22 SSH) |
| `--jupyter-lab` | `use_jupyter_lab: true` | Lanza JupyterLab en vez de Notebook clásico |
| `--jupyter-dir <path>` | `jupyter_dir: <path>` | Directorio raíz del server (default `/workspace`) |
| `--direct` | — | Conexión HTTPS directa (no proxy). Requiere instalar certificado TLS local |
| `--env '<flags>'` | `env: {...}` | Variables de entorno + port forwarding (e.g., `-e KEY=val -p PORT:PORT`) |
| `--disk <GB>` | `disk` | Tamaño del volumen (default 10 GB; usamos 30 para datasets) |
| `--onstart-cmd <cmd>` | `onstart` | Comando a ejecutar al arrancar el container |
| `--ssh` | `run_with_ssh: true` | Habilita SSH (cuidado con quirk `d5f717eb` si imagen no incluye `sshd`) |

### 8.2 Crear instancia (template completo)

```bash
# Buscar offer barato con RTX 4090 (CLI Vast.ai)
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 dph_total<=0.50 reliability>0.95 inet_down>=200' \
  --order 'dph_total'

# Crear instancia (reemplazar <OFFER_ID> con el del search)
vastai create instance <OFFER_ID> \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --env '-e VAST_API_KEY=<VAST_KEY> -e HF_TOKEN=<HF_KEY> -e WANDB_API_KEY=<WANDB_KEY> -e ROBOFLOW_API_KEY=<RF_KEY> -e GITHUB_TOKEN=<GH_PAT> -p 8080:8080' \
  --jupyter --jupyter-lab \
  --jupyter-dir /workspace \
  --disk 30 \
  --onstart-cmd 'bash -c "curl -sSL https://raw.githubusercontent.com/mitgar14/embebidos-3/main/scripts/bootstrap.sh | bash"'

# Ver instancias activas
vastai show instances

# Conectar (printa URL de JupyterLab)
vastai show instance <ID>

# Destruir manualmente (si watchdog falla)
vastai destroy instance <ID>
```

### 8.3 GitHub token para clone de repo privado

Para que `git clone https://github.com/mitgar14/embebidos-3.git` funcione desde el container:

1. Generar **fine-grained PAT** en GitHub con scope `repo:read` (solo lectura).
2. Inyectar como `GITHUB_TOKEN` vía `--env`.
3. Bootstrap usa `https://${GITHUB_TOKEN}@github.com/mitgar14/embebidos-3.git`.

**Alternativa:** deploy key SSH del repo, pero requiere config `~/.ssh/known_hosts` y agente en el container. PAT es más simple para uso único.

---

## 9. Configuración local previa (Windows)

### 9.1 Generar PAT GitHub (una sola vez)

1. https://github.com/settings/tokens?type=beta
2. Generate new token → scope `Contents: read` para `mitgar14/embebidos-3`.
3. Guardar como `GITHUB_TOKEN` en `.env` local (gitignored).

### 9.2 Token HF Hub

Ya autenticado: `hf auth whoami` → `user: mitgar14` (verificado 2026-05-12).
Token en `~/.cache/huggingface/token` localmente. Para Vast.ai, leer y exportar:

```powershell
# PowerShell Windows
$HF_TOKEN = Get-Content "$env:USERPROFILE\.cache\huggingface\token"
# Inyectar en --env -e HF_TOKEN=$HF_TOKEN al crear instancia
```

### 9.3 `.env` local recomendado (gitignored)

```bash
# .env (NO commitear)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
VAST_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Verificar `.gitignore` incluye `.env*` (HANDOFF §4 confirma que sí).

### 9.4 `.gitattributes` para line endings (obligatorio)

Para evitar el bug CRLF documentado (mnemon `27b66a6b`):

```gitattributes
# .gitattributes
*.sh   text eol=lf
*.py   text eol=lf
*.ipynb text eol=lf
*.json text eol=lf
*.yaml text eol=lf
*.yml  text eol=lf
*.md   text eol=lf
```

Validar tras commit con `git check-attr -a scripts/bootstrap.sh` (esperar `text: set`, `eol: lf`).

### 9.5 Vast.ai CLI local

```powershell
# PowerShell Windows
pip install vastai
vastai set api-key "$env:VAST_API_KEY"
vastai search offers 'gpu_name=RTX_4090' --order 'dph_total' | Select-Object -First 10
```

Path del binario en Windows (memoria mnemon `b13050ac`): `C:\Users\mitgar14\AppData\Roaming\Python\Python312\Scripts\vastai.exe`.

---

## 10. Fuentes consultadas

| # | Título | URL | Tipo |
|---|--------|-----|------|
| 1 | Jupyter — Vast.ai Documentation | https://docs.vast.ai/guides/instances/connect/jupyter | Doc oficial |
| 2 | vastai create instance — Vast.ai CLI | https://docs.vast.ai/cli/reference/create-instance | Doc oficial |
| 3 | Docker Execution Environment — Vast.ai | https://docs.vast.ai/documentation/instances/docker-environment | Doc oficial |
| 4 | Jupyter Server full config | https://jupyter-server.readthedocs.io/en/latest/other/full-config.html | Doc oficial |
| 5 | DeepWiki jupyter-server/jupyter_server | https://deepwiki.com/jupyter-server/jupyter_server | Doc generada |
| 6 | Using uv with Jupyter | https://docs.astral.sh/uv/guides/integration/jupyter/ | Doc oficial |
| 7 | Using uv workspaces | https://docs.astral.sh/uv/concepts/projects/workspaces/ | Doc oficial |
| 8 | Using uv with PyTorch | https://docs.astral.sh/uv/guides/integration/pytorch/ | Doc oficial |
| 9 | uv Managing dependencies | https://docs.astral.sh/uv/concepts/projects/dependencies/ | Doc oficial |
| 10 | bluss/pyproject-local-kernel | https://github.com/bluss/pyproject-local-kernel | Repo |
| 11 | DeepWiki astral-sh/uv | https://deepwiki.com/astral-sh/uv | Doc generada |
| 12 | HuggingFace Hub `_tensorboard_logger.py` source | https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_tensorboard_logger.py | Código fuente |
| 13 | HuggingFace Hub CLI upload.py source | https://github.com/huggingface/huggingface_hub/blob/0b55fb46/src/huggingface_hub/cli/upload.py | Código fuente |
| 14 | HF Hub Uploading models docs | https://huggingface.co/docs/hub/en/models-uploading | Doc oficial |
| 15 | HF Hub TensorBoard docs | https://huggingface.co/docs/hub/tensorboard | Doc oficial |
| 16 | HF Hub `HFSummaryWriter` reference | https://huggingface.co/docs/huggingface_hub/main/package_reference/tensorboard | Doc oficial |
| 17 | HF Hub Upload files guide | https://huggingface.co/docs/huggingface_hub/guides/upload | Doc oficial |
| 18 | HF Hub Storage Buckets blog | https://huggingface.co/blog/storage-buckets | Blog oficial |
| 19 | papermill 2.7.0 documentation | https://papermill.readthedocs.io/ | Doc oficial |
| 20 | nteract/papermill repo | https://github.com/nteract/papermill | Repo |
| 21 | Jupytext CLI docs | https://jupytext.readthedocs.io/en/stable/using-cli.html | Doc oficial |
| 22 | mwouts/papermill_jupytext | https://github.com/mwouts/papermill_jupytext | Repo |
| 23 | Google Cloud "Deep Learning VMs + Jupyter + Papermill" blog | https://cloud.google.com/blog/products/ai-machine-learning/let-deep-learning-vms-and-jupyter-notebooks-to-burn-the-midnight-oil-for-you-robust-and-automated-training-with-papermill | Blog Google |
| 24 | vast-ai/base-image Dockerfile + README | https://github.com/vast-ai/base-image/blob/main/Dockerfile | Código fuente |
| 25 | vast-ai/vast-python (vastai CLI) | https://github.com/vast-ai/vast-python | Código fuente |
| 26 | hub.docker.com vastai/base-image tags | https://hub.docker.com/r/vastai/base-image/tags | Doc oficial |
| 27 | docs.vast.ai Jupyter & SSH FAQ | https://docs.vast.ai/documentation/reference/faq/jupyter-ssh | Doc oficial |
| 28 | docs.vast.ai Choosing a Template | https://docs.vast.ai/guides/instances/choosing/templates | Doc oficial |
| 29 | docs.vast.ai Technical FAQ (Docker) | https://docs.vast.ai/documentation/reference/faq/technical | Doc oficial |
| 30 | Wandb pricing free tier | https://wandb.ai/site/pricing | Doc oficial |
| 31 | Ultralytics docs Kaggle integration (W&B) | https://docs.ultralytics.com/integrations/kaggle/ | Doc oficial |
| 32 | YouTube "Simplest Way to run Jupyter Notebooks on Cheap Cloud GPUs" (Thunder Compute) | https://www.youtube.com/watch?v=XprbTJYTc6M | Video |
| 33 | YouTube ArjanCodes "Share Code Between Python Apps" (uv workspaces) | https://www.youtube.com/watch?v=N_ypJwV8Q8I | Video |
| 34 | YouTube "Vast.ai Quickstart Guide 2025 Update" | https://www.youtube.com/watch?v=GxCLo1vYrbY | Video |
| 35 | docs.vast.ai FAQ instances (auto-destroy) | https://docs.vast.ai/guides/reference/faq/instances | Doc oficial |

---

## 11. Cross-references

- **[`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md)** — D3, D5, D6, D9, D10, D11 detallados.
- **[`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md)** — Stack TF/PyTorch pins (Track A §6, Track B §7); gotchas Vast.ai (#25, #26, #27, #28).
- **[`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md)** — Gates pre-deploy a ejecutar antes del auto-destroy (D12, D13).
- **[`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md)** — Roboflow `DATASET_DIRECTORY` env var en bootstrap.
- **[`HANDOFF-implementacion-vastai-hf.md`](HANDOFF-implementacion-vastai-hf.md)** — Tarea #4' (`bootstrap.sh`) usa la plantilla de §7.

---

**Fin del documento.** Cualquier cambio a D3, D5, D6, D9, D10 o D11 requiere nueva ronda `/investiga`.
