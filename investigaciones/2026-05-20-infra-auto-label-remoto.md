# Investigación — Infraestructura para auto-labeling remoto compartido

**Dominio:** `embebidos-3` — orquestación de jobs de auto-labeling (Autodistill + Grounding DINO) entre mitgar14 (Windows 11 sin GPU), Nicolás (Windows 11 + WSL2 + RTX 3060 6 GB) y Claude Code (agente).
**Proyecto:** clasificador 3 clases YOLOv8n para banda transportadora — demo 2026-05-26 (T-6 días).
**Ronda 1 inaugural** — dominio nuevo. Ortogonal a la investigación previa de modelo (`2026-05-15-mejoras-modelo-deteccion-plasticos.md`, sobre recall, SAHI, augmentation).

---

## Contexto operacional

**Hardware disponible:**
- mitgar14: Windows 11, sin GPU. Cliente del flujo.
- Nicolás: Windows 11 + WSL2 instalado, RTX 3060 Laptop **6 GB VRAM** (driver 595.97, CUDA 13.2 reportado). Miniconda + Ollama ya corriendo. NO presencial 100% del tiempo.
- Vast.ai: créditos ya disponibles. Patrón validado (`bootstrap.sh` + `train_track_b_yolov8.ipynb` con nbconvert + tmux + CommitScheduler + auto-destroy).
- Jetson Nano: target de deploy final, no relevante para labeling.

**Constraints negativas (descartadas explícitamente):**
- Escritorio remoto invasivo (AnyDesk/RustDesk/Parsec) — viola privacidad de Nicolás.
- Modal/RunPod/Replicate/Colab Pro — usuario quiere consolidar en Vast.ai.

**Requisitos:**
1. mitgar14 dispara y monitorea jobs sin estar presente físicamente.
2. Claude Code controla jobs vía HTTP/SSH/API sin intervención humana.
3. Nicolás conserva control y privacidad de su PC.
4. Costo cero o muy bajo.
5. Setup reproducible.

---

## Resumen ejecutivo — Veredicto

**PRIMARIO: Vast.ai on-demand con notebook `autolabel_vastai.ipynb` análogo al de training.**

**PLAN B: Patrón FastAPI + systemd-in-WSL2 + Tailscale Serve en la RTX 3060 de Nicolás** (si créditos Vast.ai se agotan o queremos infraestructura permanente post-demo).

### Razones de la decisión

1. **Costo despreciable con créditos ya cargados.** Una RTX 3060 12 GB en Vast.ai cuesta $0.07-0.15/hr (gpuperhour.com, gpufinder.dev, may 2026). 30 min ≈ $0.035-0.075. Para 5-10 iteraciones antes del demo: $0.35-0.70 total → consumido del crédito existente.
2. **Autonomía total para mitgar14 y Claude.** Vast.ai expone Python SDK (`pip install vastai`) y REST API. Claude provisiona, ejecuta, recoge artifacts, destruye, sin tocar la PC de Nicolás. mitgar14 sólo necesita comprobar que el HF Hub recibió los labels.
3. **Cero fricción de coordinación.** No hay que pedirle permiso a Nicolás, ni esperar a que esté disponible, ni preocuparse de que su PC esté competing por VRAM con Ollama/juegos. La RTX 3060 6 GB queda muy justa para Grounding DINO base (~4-5 GB) + procesos Windows.
4. **Reusa código y conocimiento que ya tienen.** `bootstrap.sh`, patrón nbconvert headless, CommitScheduler para artifacts (descartado para job corto), auto-destroy via `CONTAINER_API_KEY` (variable inyectada por Vast.ai con scope limitado a esa única instancia — no necesita exponer el API key principal).
5. **Vast.ai skill oficial para Claude Code** (lanzada mayo 2026): `npx skills add vast-ai/vast-cli`. Reduce aún más la fricción de automatización.

### Cuándo activar el Plan B (Nicolás)

- Vast.ai está caído o sin ofertas RTX 3060/T4 disponibles en el momento.
- Créditos se agotaron (probable después de 200+ jobs de 30 min — irrelevante para T-6 días).
- Decisión post-demo de mover labeling a infraestructura permanente "casera".

---

## Ronda 1 — 2026-05-20 (profundidad media)

### Track A — Implementaciones técnicas

#### A1. Vast.ai workflow para job corto (research-web)

**Costo verificado (mayo 2026):**
| GPU | Modo | $/hr | $/30min |
|---|---|---|---|
| RTX 3060 12 GB | on-demand | $0.07-0.15 | $0.035-0.075 |
| RTX 3060 12 GB | bid (interruptible) | $0.09 | $0.045 |
| RTX 4090 | on-demand | $0.30-0.50 | $0.15-0.25 |
| V100 16 GB (fallback si no hay 3060) | on-demand | $0.18-0.20 | $0.09-0.10 |

Fuentes: `https://gpuperhour.com/rent/rtx-3060` (feb 2026), `https://gpufinder.dev/providers/vast` (may 2026).

**Cold start:** docs oficiales reportan "generally under 5 minutes with recommended templates" (`https://docs.vast.ai/api-reference/creating-instances-with-api`). Realista 3-4 min con imagen PyTorch ya cacheada en el host.

**Recomendación on-demand vs bid:** para job de 30 min con artifacts que no queremos perder a mitad de ejecución, **on-demand**. El diferencial ($0.03 vs $0.045) es despreciable y eliminamos riesgo de outbid. Las instancias bid son válidas para training de varias horas con checkpoints frecuentes (caso del notebook training actual con CommitScheduler), no para inference cortos.

**Storage al detener:** docs oficial: "Storage charges continue even when instances are stopped. Delete instances completely to cease storage billing." Patrón correcto para jobs ephemeral: **destroy, no stop**.

**Autonomía de Claude — Python SDK:**

```python
from vastai import VastAI
vast = VastAI()  # Lee ~/.config/vastai/vast_api_key

# 1. Buscar oferta
offers = vast.search_offers(
    query="gpu_name=RTX_3060 gpu_ram>=10 reliability>0.95 rentable=true",
    order="dph_total+",   # mas barato primero
    limit="3",
)

# 2. Crear instancia con onstart auto-destroy
result = vast.create_instance(
    id=offers[0]["id"],
    image="pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime",
    disk=20,
    onstart_cmd=(
        "pip install -q autodistill autodistill-grounding-dino "
        "supervision huggingface_hub && "
        "python /workspace/autolabel.py && "
        "vastai destroy instance $CONTAINER_API_KEY"
    ),
    ssh=True,
    direct=True,
)
# 3. Poll status hasta running, hacer copy() de artifacts, ya esta auto-destroy
```

`CONTAINER_API_KEY` es inyectado automáticamente por Vast.ai en el contenedor con permisos `start/stop/destroy` solo sobre esa instancia — no expone tu API key principal (fuente: `https://docs.vast.ai/sdk/python/quickstart`).

**Runtype recomendado:** `runtype: "ssh_direct"` para job headless (no necesitamos Jupyter). El `onstart` corre después del entrypoint Vast inicializa SSH.

**Gotcha verificado:** `env` field debe ser JSON dict, NO Docker flag string. Correcto: `{"VAR": "val"}`. Wrong: `"-e VAR=val"`. Las env vars no son visibles en SSH a menos que `onstart` ejecute `env >> /etc/environment`.

#### A2. Notebook `autolabel_vastai.ipynb` propuesto (research-web + research-code)

Celdas análogas al `train_track_b_yolov8.ipynb`:

1. **Env setup** — `pip install autodistill autodistill-grounding-dino supervision huggingface_hub`.
2. **Config** — ontology dict, conf threshold (0.25 base, 0.35 para `paper`), destino HF Hub (`mitgar14/embebidos3-labels`).
3. **Pre-flight** — `nvidia-smi`, verificar VRAM >= 8 GB, validar token HF, validar input dir.
4. **Descargar imágenes** — `hf_hub_download` desde un repo dataset privado donde mitgar14 subió las 43 fotos, o `git clone` de un repo público.
5. **Auto-label** — `GroundingDINO(ontology=...).label(input_folder=..., output_folder=..., extension=".jpg")`. Output: `labeled/train/{images,labels}/*.{jpg,txt}` + `data.yaml`.
6. **Gate de calidad** — verificar `len(labels) >= n_images * 0.85` (al menos 85% de imágenes con detecciones). Si no, abortar sin destruir para inspección manual.
7. **Upload** — `huggingface_hub.upload_large_folder` con `labeled/` al repo dataset.
8. **Auto-destroy** — `vastai destroy instance $CONTAINER_API_KEY`.

**Diferencias con notebook training:**
- Sin `CommitScheduler` (job dura 30 min, no horas).
- Sin gates ONNX/Polygraphy.
- Auto-destroy condicional al gate de calidad (no incondicional como training).
- Sin freeze del backbone ni hiperparámetros augmentation (es inference puro).

#### A3. Plan B — Patrón FastAPI + systemd-WSL2 + Tailscale (research-code)

Setup mínimo viable si decidimos usar la RTX 3060 de Nicolás en lugar de Vast.ai.

**Setup en WSL2 Ubuntu 22.04 (PC Nicolás):**

```bash
# 1. Habilitar systemd (OBLIGATORIO para Tailscale, SSH persistente, services)
echo -e "[boot]\nsystemd=true" | sudo tee /etc/wsl.conf
# Desde PowerShell host: wsl --shutdown && wsl

# 2. CUDA: ya funciona via passthrough Windows. Verificar:
/usr/lib/wsl/lib/nvidia-smi   # ruta absoluta, no esta en PATH default

# 3. Conda env aislado para autolabel
conda create -n autolabel python=3.10 -y
conda activate autolabel
pip install fastapi uvicorn autodistill autodistill-grounding-dino \
            supervision huggingface_hub python-multipart

# 4. Service systemd unit
sudo tee /etc/systemd/system/embebidos3-label.service > /dev/null <<EOF
[Unit]
Description=embebidos3 Autolabeling API
After=network.target
[Service]
User=nico
WorkingDirectory=/home/nico/embebidos3-label
ExecStart=/home/nico/miniconda3/envs/autolabel/bin/uvicorn main:app --host 0.0.0.0 --port 8765
Restart=on-failure
RestartSec=5
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PATH=/home/nico/miniconda3/envs/autolabel/bin:/usr/lib/wsl/lib:/usr/bin:/bin
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now embebidos3-label

# 5. Tailscale en Windows host (NO en WSL2 — el doble Tailscale rompe MTU)
# Descargar tailscale.exe en Windows, login Nicolas
# Desde WSL2:
tailscale serve --bg 8765   # expone localhost:8765 dentro del tailnet
```

**FastAPI mínima (`main.py`) — análoga a `scripts/server/`:**

```python
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import subprocess, json, uuid, os, signal, asyncio
from pathlib import Path

app = FastAPI()
JOBS: dict = {}
JOBS_FILE = Path("/run/embebidos3-label/jobs.json")
JOBS_FILE.parent.mkdir(exist_ok=True)

@app.post("/autolabel/job")
async def start_job(input_url: str, ontology: dict, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    log_path = Path(f"/tmp/label-{job_id}.log")
    proc = subprocess.Popen(
        ["python", "-m", "autolabel_worker", input_url, json.dumps(ontology), job_id],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
    )
    JOBS[job_id] = {"pid": proc.pid, "log": str(log_path), "state": "running"}
    JOBS_FILE.write_text(json.dumps(JOBS))
    return {"job_id": job_id}

@app.get("/jobs/{job_id}/state")
def get_state(job_id: str):
    return JOBS.get(job_id, {"error": "not_found"})

@app.get("/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    log_path = JOBS[job_id]["log"]
    async def gen():
        with open(log_path) as f:
            while True:
                line = f.readline()
                if line: yield f"data: {line}\n\n"
                else: await asyncio.sleep(0.5)
    return StreamingResponse(gen(), media_type="text/event-stream")

@app.delete("/jobs/{job_id}")
def cancel(job_id: str):
    os.kill(JOBS[job_id]["pid"], signal.SIGTERM)
    JOBS[job_id]["state"] = "cancelled"
    return {"ok": True}
```

**Acceso desde mitgar14 / Claude (mismo tailnet):**

```python
import requests
BASE = "https://nicolas-pc.tailnet-name.ts.net:8765"
r = requests.post(f"{BASE}/autolabel/job", json={
    "input_url": "https://huggingface.co/datasets/mitgar14/embebidos3-raw/resolve/main/batch1.zip",
    "ontology": {
        "plastic bottle or plastic container": "plastic",
        "paper or cardboard": "paper",
        "glass bottle or glass jar": "glass",
    }
})
job_id = r.json()["job_id"]
# Poll state, stream logs SSE, GET artifact, mismo patron del dashboard embebidos3.
```

**Privacidad para Nicolás:** él controla el servicio (`systemctl status/stop/restart`), ve los jobs corriendo, puede leer logs con `journalctl -u embebidos3-label`. Claude solo accede a los endpoints HTTP. Tailscale ACL puede restringir qué dispositivos del tailnet pueden hablar con `:8765`.

#### A4. Por qué NO los otros patrones evaluados

| Patrón | Razón de descarte |
|---|---|
| **SSH + tmux + script bash** | Bajo control granular para Claude (comandos compuestos), gotchas frágiles (WSL2 idle mata tmux si `autoStop=true` en `.wslconfig`), bajo cumplimiento de privacidad (Nicolás ve todas las SSH sessions pero no qué hace Claude). |
| **JupyterLab Server + REST API** | Protocolo Jupyter de kernels requiere WebSocket además de REST — no hay endpoint `POST /api/execute-notebook` simple (Jupyter Community Forum, sep 2023). Implementación de `tfoldi/jupyterapi_nbrunner` (3 estrellas) es la referencia más simple pero añade complejidad. |
| **GitHub Actions self-hosted runner Windows** | Setup +30-40 min sobre Patrón C (Zenn, abr 2026). El runner necesita registrarse manualmente con token, y `gh workflow run` es asíncrono — perdemos SSE logs en vivo. |
| **NSSM servicio Windows nativo** | Capa de indirección extra (`wsl.exe python main.py`), complica forwarding GPU y env vars. systemd-in-WSL2 es netamente superior. |

---

### Track B — Búsqueda ampliada (discover.py + Exa)

#### B1. CUDA passthrough WSL2 — funciona out-of-the-box en Windows 11

Docs oficiales Microsoft (`https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl`) y NVIDIA confirman que el driver Windows proyecta `libcuda.so` dentro de WSL2 en `/usr/lib/wsl/lib/`. **Nunca instalar drivers NVIDIA dentro de WSL2** — usa el del host. Issue verificado `microsoft/WSL#11589` (may 2024): si `nvidia-smi` muestra "CUDA Version: ERR!", actualizar driver Windows del host.

**Gotcha SSH no-interactivo:** sesiones SSH no-interactivas (`ssh host CMD`) NO cargan `.bashrc` → `/usr/lib/wsl/lib` no está en PATH, `nvidia-smi` falla con "command not found". Solución: ruta absoluta `/usr/lib/wsl/lib/nvidia-smi` o `ssh host 'bash -lc "nvidia-smi"'` (login shell forzado). Mismo patrón que el insight de memoria del proyecto sobre SSH a Jetson Nano (`/home/jetson/.local/bin/uv`).

#### B2. Tailscale + WSL2 — gotcha del doble Tailscale

Docs oficiales Tailscale (`https://tailscale.com/kb/1295/install-windows-wsl2`, nov 2025): **NO correr Tailscale simultáneamente en Windows host y dentro de WSL2** — el tráfico cifrado dentro de tráfico cifrado causa fragmentation y bugs MTU. Issues confirmados:
- `tailscale/tailscale#4140` — SSH tunnel breaks on flood due to MTU size
- `tailscale/tailscale#4833` — Cannot receive large packets inside wsl2 due to MTU
- `tailscale/tailscale#6189` — Tailscale SSH not working on newest WSL2 from MS Store

**Patrón correcto:** Tailscale solo en Windows host. WSL2 hereda la conectividad via NAT del host. Para exponer un puerto WSL2 al tailnet, usar `tailscale serve --bg PORT` desde dentro de WSL2 (que apunta a un proceso local en `0.0.0.0:PORT`).

Precedente exacto del patrón propuesto: post oficial Tailscale "Remote machine learning on Windows with Docker and WSL2 from anywhere" (abr 2024, `https://tailscale.com/blog/remote-gpus-docker-wsl2-immich`) — usuario expone GPU casera (Windows 11 + WSL2 + Docker) al tailnet para Immich machine learning. Caso de uso casi idéntico al nuestro.

#### B3. Vast.ai cli/SDK actualizado mayo 2026

Repo oficial: `https://github.com/vast-ai/vast-python` (191 stars, Python SDK + CLI unificados). Mayo 2026 product update (`https://vast.ai/article/may-2026-product-update`) menciona la skill oficial para Claude Code/Codex via `npx skills add vast-ai/vast-cli`.

Endpoint clave para auto-destroy: `vastai destroy instance $CONTAINER_API_KEY` dentro del onstart_cmd. La variable de entorno la inyecta Vast.ai automáticamente con scope limitado (start/stop/destroy sobre esa instancia única). No requiere API key principal.

#### B4. Anti-patterns documentados de "share home GPU"

Patrones recurrentes en Reddit r/homelab, r/LocalLLaMA, r/MachineLearning (2023-2025):
- **OOM silencioso** cuando 2 procesos compiten por VRAM sin `CUDA_VISIBLE_DEVICES` explícito.
- **Imposibilidad de cancelar job ajeno** sin SSH + `kill` — el "dueño" queda atrapado.
- **"Pedir permiso" via WhatsApp/Discord** antes de cada job → latencia de coordinación de horas, mata iteración.
- **VRAM justa**: RTX 3060 6 GB de Nicolás deja ~1-2 GB libres una vez Grounding DINO base (~4-5 GB) está cargado, lo que entra en conflicto con cualquier proceso GPU de Windows (overlay drivers, OBS, juegos).

Mitigación si vamos por Plan B: `CUDA_VISIBLE_DEVICES=0` explícito en systemd Environment, dashboard FastAPI con `DELETE /jobs/{id}` accesible a Nicolás también, cola serial (no concurrente) en el endpoint.

---

## Plan de ejecución (próximos pasos)

1. **Re-descargar v1-B desde Roboflow** antes de cancelar workspace (Task #8).
2. **Crear `notebooks/autolabel_vastai.ipynb`** análogo al de training (Task #4).
3. **Subir las 43 fotos** a un HF Dataset repo (`mitgar14/embebidos3-raw-batches`) — fuente de input para el notebook.
4. **Probar el flujo end-to-end** con las 43: provision → auto-label → upload labels → auto-destroy. Validar gate calidad >85%.
5. **Revisar 10-15 labels visualmente** (overlay en imagen). Si el recall de `paper` es bajo, subir conf threshold sólo para esa clase o re-prompt la ontology.
6. **Fusionar con dataset v1-B** y mandar a re-training en Vast.ai con hiperparámetros augmentation reforzados (degrees=20, hsv_v=0.6, mixup=0.15, copy_paste=0.3, close_mosaic=15, + Albumentations RandomShadow/CLAHE/RandomBrightnessContrast).
7. **Re-deploy via dashboard embebidos3** (`POST /model/build`).

**No-go decisions:**
- NO doble augmentation Roboflow + Ultralytics (negative constraint ronda anterior).
- NO usar best.pt como pre-labeler (refuerza overfit hacia plastic).
- NO doble Tailscale Windows host + WSL2.
- NO usar `dynamic=True` en re-export TRT (Maxwell inestable).

---

---

## Ronda 2 — 2026-05-20 (ejecución end-to-end Vast.ai)

Smoke test completo del flujo Vast.ai con `notebooks/autolabel_vastai.ipynb`. **Resultado: 100% recall (37/37 imágenes), 256 bboxes, $0.057/hr × ~20 min ≈ $0.02 total.**

### Instancia y costo

- ID: `37177982` (RTX 3060 12 GB, Polonia, host 78246, driver 535.154.05, CUDA 12.2).
- Imagen: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`, disk 20 GB, runtype `ssh_direct`.
- Costo unitario: `$0.0597/hr` total (gpu $0.056 + disk $0.0037).
- Duración real medida: ~20 min (deps install ~3 min, model load ~1.5 min, label 37 imgs ~50s, upload ~30s, decisión manual ~14 min para diagnosticar bug del gate).
- Destruida tras finalizar (`vastai destroy instance 37177982`).

### Bugs encontrados y corregidos en el notebook

| Bug | Síntoma | Fix |
|-----|---------|-----|
| `GroundingDINO(model_type=...)` | `TypeError: unexpected kwarg 'model_type'` | API actual de `autodistill-grounding-dino` (≥0.1.x): NO acepta `model_type`. Usar `box_threshold` + `text_threshold` en constructor (heurística: `text_threshold = box_threshold * 0.7`). |
| `base.label(conf=CONF)` | `TypeError: unexpected kwarg 'conf'` | El threshold se pasa al constructor, no a `.label()`. |
| Gate cuenta solo `train/labels` | Reporta 78.4% cuando recall real es 100% | Autodistill hace split automático train/valid. Gate debe iterar AMBOS: `(train/labels/*.txt + valid/labels/*.txt)`. |
| Deps transitivas no declaradas | `ImportError: No module named 'sklearn'`, `'roboflow'`, `AutoTokenizer` (transformers 5+ rompe) | Pin explícito en `REQ` antes de autodistill: `opencv-python-headless`, `transformers<5`, `scikit-learn`, `roboflow`, `timm>=1.0`, `accelerate`. |
| `hf` CLI (huggingface-cli deprecated) | `huggingface-cli` deprecation warning | Migrar a `hf download`/`hf upload` y pasar `--token` explícito (env var no se hereda en subshell SSH). |
| `vastai execute` 400 | "Execute command only avail on stopped instances" | Para instancias `running` usar SSH directo, no `vastai execute`. |
| `vastai destroy` interactive prompt | Stuck en confirmación `[y/N]` | `echo "y" \| vastai destroy instance ID`. |

### Distribución de bboxes generados

- **plastic**: 65 (25%)
- **paper**: 143 (56%)
- **glass**: 48 (19%)

**Comparación crítica con el dataset original v1-B (donde el modelo overfitteó plastic 5.3×):** la batch1 invierte la dominancia (paper 2.2× sobre plastic). Esto es exactamente lo que necesita el modelo para des-sesgar. La intuición de la ronda anterior (`2026-05-15`) sobre re-equilibrar dataset queda validada por la composición de las fotos físicas del 2026-05-15.

### Privacidad/handoff confirmados

- Nicolás NO fue tocado en este flujo. Plan B (`scripts/labeling/`) queda como respaldo activo pero sin activación.
- mitgar14 controla 100% el flujo desde Windows + `vastai` CLI (`uv add vastai` ya en `pyproject.toml`).
- Claude controla SSH directo (puerto 13822) + HF Hub API + Vast.ai SDK sin tocar la PC de Nicolás.

### Artifacts publicados

- HF dataset privado `mitgar14/embebidos3-raw-batches`:
  - `batch1/WIN_20260515_*.jpg` (37 fotos, 4.36 MB)
  - `.notebook/autolabel_vastai.ipynb` (versión corregida con los 5 fixes anteriores)
- HF dataset privado `mitgar14/embebidos3-labels`:
  - `batch1/data.yaml`
  - `batch1/train/{images,labels}/` (29 pairs)
  - `batch1/valid/{images,labels}/` (8 pairs)
- Local: `datasets/waste-3class-batch1-auto/batch1/` (5.5 MB, copia pull desde HF para próxima fase de merge con v1-B).

### Aprendizajes para futuras rondas

1. **El gate de calidad debe alinearse con la lógica de Autodistill** (auto-split train/valid). Cualquier downstream tool que cuente imágenes etiquetadas debe iterar ambos splits.
2. **`autodistill-grounding-dino` tiene API frágil**: las versiones recientes movieron `box_threshold`/`text_threshold` del `.label()` al constructor. Verificar cada actualización menor.
3. **El pin `transformers<5`** es probable que se requiera por varios meses más hasta que autodistill se actualice. Documentar en el README del notebook.
4. **Coste real << esperado**: con créditos Vast.ai existentes, 10 iteraciones de 30 min cada una son ~$0.30 — completamente despreciable para la fase de iteración previa al demo.

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-20 | media | Infra compartida para auto-labeling: Vast.ai vs RTX 3060 Nicolás. Veredicto: Vast.ai primario, Plan B FastAPI+systemd-WSL2+Tailscale. |
| 2 | 2026-05-20 | media | Ejecución end-to-end Vast.ai. 5 bugs encontrados + corregidos. 100% recall (37/37), 256 bboxes, distribución paper-dominante (re-equilibra v1-B). $0.02 total. |

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | Vast.ai SDK Hello World | https://docs.vast.ai/sdk/python/quickstart | Doc oficial | 1 |
| 2 | Vast.ai Creating Instances with API | https://docs.vast.ai/api-reference/creating-instances-with-api | Doc oficial | 1 |
| 3 | Vast.ai May 2026 Product Update | https://vast.ai/article/may-2026-product-update | Blog oficial | 1 |
| 4 | Vast.ai Pricing | https://docs.vast.ai/documentation/instances/pricing | Doc oficial | 1 |
| 5 | Vast.ai On-Demand vs Interruptible | https://vast.ai/article/Rental-Types | Blog oficial | 1 |
| 6 | RTX 3060 Price on Vast.ai | https://gpuperhour.com/rent/rtx-3060 | Agregador | 1 |
| 7 | Vast GPU Cloud Pricing | https://gpufinder.dev/providers/vast | Agregador | 1 |
| 8 | vast-ai/vast-python repo | https://github.com/vast-ai/vast-python | Código | 1 |
| 9 | Tailscale: Remote ML on Windows + Docker + WSL2 (Immich) | https://tailscale.com/blog/remote-gpus-docker-wsl2-immich | Blog oficial | 1 |
| 10 | Tailscale: Install on Windows with WSL2 | https://tailscale.com/kb/1295/install-windows-wsl2 | Doc oficial | 1 |
| 11 | Tailscale SSH | https://tailscale.com/docs/features/tailscale-ssh | Doc oficial | 1 |
| 12 | Tailscale issue #4140 — MTU SSH tunnel | https://github.com/tailscale/tailscale/issues/4140 | Issue | 1 |
| 13 | Tailscale issue #4833 — MTU WSL2 | https://github.com/tailscale/tailscale/issues/4833 | Issue | 1 |
| 14 | Tailscale issue #6189 — SSH WSL2 MS Store | https://github.com/tailscale/tailscale/issues/6189 | Issue | 1 |
| 15 | Tailscale issue #10328 — systemd WSL2 requirement | https://github.com/tailscale/tailscale/issues/10328 | Issue | 1 |
| 16 | Tailscale + SSH for Windows + WSL2 (Steenhoek 2026-03) | https://benjijang.com/posts/2026/03/tailscale-wsl2-ssh/ | Blog | 1 |
| 17 | Microsoft Learn: Enable NVIDIA CUDA on WSL2 | https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl | Doc oficial | 1 |
| 18 | Microsoft Learn: GPU accelerated ML training in WSL | https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute | Doc oficial | 1 |
| 19 | microsoft/WSL issue #11589 — nvidia-smi "CUDA Version: ERR!" | https://github.com/microsoft/WSL/issues/8179 | Issue | 1 |
| 20 | Autodistill GroundingDINO repo | https://github.com/autodistill/autodistill-grounding-dino | Código | 1 |
| 21 | Roboflow: Deploy Grounding DINO to a GPU | https://roboflow.com/how-to-deploy/deploy-grounding-dino-to-a-gpu | Doc | 1 |
| 22 | Replicate cog WSL2 guide | https://github.com/replicate/cog/blob/main/docs/wsl2/wsl2.md | Doc | 1 |
| 23 | Hanselman blog: Tailscale + WSL2 + VS Code | https://www.hanselman.com/blog/using-tailscale-on-windows-to-network-more-easily-with-wsl2-and-visual-studio-code | Blog | 1 |
| 24 | Tailscale: Self-host a local AI stack | https://tailscale.com/blog/self-host-a-local-ai-stack | Blog | 1 |
| 25 | tfoldi/jupyterapi_nbrunner — execute notebook via REST | https://github.com/tfoldi/jupyterapi_nbrunner | Código | 1 |
| 26 | Computestacker: Spot vs On-Demand GPU 2026 | https://computestacker.com/insights/spot-vs-on-demand-gpu-instances/ | Blog | 1 |

## Cross-refs con otras investigaciones

- `2026-05-15-mejoras-modelo-deteccion-plasticos.md` (Ronda 1) — hiperparámetros augmentation post-labeling (`copy_paste=0.5, hsv_v=0.6, freeze=10, lr0=0.0005`) y la decisión de no double-augment con Roboflow.
- `HANDOFF-track-b-2026-05-13.md` — patrón `bootstrap.sh` + `CommitScheduler` + auto-destroy que se reusa parcialmente en el nuevo notebook.
- Skill `embebidos3-nano` — recordar verificar estado del Nano antes de redeploy (existing engine, sudoers, systemd state).
