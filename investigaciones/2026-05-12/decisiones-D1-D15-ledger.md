# Ledger de decisiones vinculantes D1–D15 — embebidos-3

**Proyecto:** `embebidos-3` — clasificador de residuos (glass, paper, plastic) para Jetson Nano B01.
**Materia:** IA en Embebidos (Prof. Juan Camilo Giraldo, UAO Cali, semestre 7).
**Entrega:** 2026-05-26.
**Fecha de cierre del ledger:** 2026-05-12 (Ronda 5 cerrada).
**Estado:** las decisiones D1–D15 listadas aquí son **vinculantes**. Cualquier cambio requiere nueva ronda `/investiga` documentada.

Este documento es el **registro maestro** de las decisiones técnicas tomadas en las rondas 4 y 5 de investigación. Sirve como índice de cross-references a los cuatro documentos temáticos consolidados:

- [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) — Por qué Vast.ai, stack TF/PyTorch, compatibilidad con JetPack 4.6.1.
- [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) — Notebook persistente `.ipynb` + `tmux + nbconvert`, `uv` dual venv, HF Hub persistence, auto-destroy.
- [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) — Gates TFLite y ONNX, Polygraphy en Docker NGC, drop INT8 Maxwell.
- [`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md) — Dataset Roboflow, Ultralytics 8.4.46, export ONNX opset 11, W&B.

---

## 1. Tabla maestra D1–D15 (resumen)

| # | Decisión (resumen) | Ronda | Estado | Fuente primaria |
|---|--------------------|-------|--------|-----------------|
| **D1** | Container Vast.ai `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` | R4 | Vigente | [compatibilidad](compatibilidad-stack-cloud-jetson.md) |
| **D2** | GPU RTX 4090 on-demand (0,35–0,50 USD/h) | R4 | Vigente | [compatibilidad](compatibilidad-stack-cloud-jetson.md) |
| **D3** | Repo HF Hub `mitgar14/embebidos-3-models` privado, estructura `track_{a,b}/{runs,checkpoints,exports,logs}/` | R4 | ✅ Creado | [infraestructura](infraestructura-training-vastai-uv-hf.md) |
| **D4** | ~~jupytext `--to py:percent` para `.ipynb` → `.py` headless~~ | R4 | ❌ **Reemplazada por D9** | — |
| **D5** | `HfApi.upload_folder(run_as_future=True)` + `CommitScheduler(every=5)` + `upload_large_folder` resumible | R4 | Vigente | [infraestructura](infraestructura-training-vastai-uv-hf.md) |
| **D6** | Track B → W&B nativo (`yolo train wandb=True`). Track A → TensorBoard hosted en HF Hub vía `HFSummaryWriter` | R4 | Vigente | [infraestructura](infraestructura-training-vastai-uv-hf.md) |
| **D7** | ~~Auto-destroy con `trap EXIT` en `run.sh`~~ | R4 | ❌ **Reemplazada por D11** | — |
| **D8** | Engine TRT siempre compilado **en el Nano**, nunca transferido | R4 | Vigente | [validación](validacion-artefactos-pre-deploy.md) |
| **D9** | Mantener `.ipynb` interactivo. Ejecutar con `jupyter nbconvert --execute --inplace` dentro de `tmux` (o `papermill` equivalente) | R5 | Vigente | [infraestructura](infraestructura-training-vastai-uv-hf.md) |
| **D10** | Dos `uv venv` separados (`/opt/venv/tracka` + `/opt/venv/trackb`) registrados como kernels `ipykernel` distintos | R5 | Vigente | [infraestructura](infraestructura-training-vastai-uv-hf.md) |
| **D11** | Auto-destroy vía **cron watchdog interno** (1 min) + última celda del notebook que crea `/workspace/embebidos-3/.training_done` y llama `vastai destroy instance ${VAST_CONTAINERLABEL#C.}` | R5 | Vigente | [infraestructura](infraestructura-training-vastai-uv-hf.md) |
| **D12** | Validación TFLite pre-deploy: `tflite==2.5.0` para inspeccionar `op_version`, carga test con wheel Coral `tflite_runtime-2.5.0.post1` CP38 x86, export con `experimental_new_quantizer=False` y `experimental_new_converter=False` | R5 | Vigente | [validación](validacion-artefactos-pre-deploy.md) |
| **D13** | Validación ONNX pre-deploy vía Docker `nvcr.io/nvidia/tensorrt:21.11-py3` (TRT 8.2.1.8 idéntico a JetPack 4.6.1). `polygraphy run --trt --onnxrt --atol 1e-2 --rtol 1e-2 --input-shapes images:[1,3,416,416]` | R5 | Vigente | [validación](validacion-artefactos-pre-deploy.md) |
| **D14** | Track B **FP16-only por default**. Experimento INT8 opcional 45–60 min en el Nano. Criterio binario: si `FPS_INT8 < FPS_FP16 × 1,10` **O** `mAP_INT8 < mAP_FP16 − 5 pp`, abandonar | R5 | Vigente | [validación](validacion-artefactos-pre-deploy.md) |
| **D15** | Si wheel NVIDIA `tensorflow==2.5.0+nv21.8` no incluye `TFLite_Detection_PostProcess`, fallback wheel Coral `tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl` (sha256 `7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df`) | R5 | Plan B | [validación](validacion-artefactos-pre-deploy.md) |

---

## 2. Razones del usuario (verbatim, fuente de las decisiones)

### Decisión del usuario en Ronda 4 (motivación de migración a Vast.ai)

> *"Quiero mantener las arquitecturas y frameworks que han sido demostradas como estables y compatibles dentro de la Jetson Nano. Usar la mejor GPU. No me importan costos, pero mantener logging robusto y persistencia de TODO en HF Hub."*

Contexto: Track A en Colab falló por la cadena `condacolab.check()` lanza `AssertionError` → fix `install_from_url(MINIFORGE_URL)` → Miniforge 23.11.0-0 trae Python 3.12 (no 3.10) → `mamba install python=3.10` falla porque `google-colab` pin a 3.12 bloquea el solver. Decisión: abandonar Colab para Track A.

### Decisión del usuario en Ronda 5 (motivación de renegociar D4 y D7)

> *"Necesito negociar [...] correrlo como notebook [...] que su ejecución no se afecte porque cerré la pestaña [...] todo sea lo más compatible entre sí (ver uv) [...] garantizar que los productos de esta fase sean compatibles/se puedan usar desde la Jetson Nano de forma estable y robusta."*

Contexto: D4 (jupytext) y D7 (`trap EXIT`) no satisfacían "ejecutar como `.ipynb` con tolerancia a cierre de pestaña". Se renegociaron a D9, D10, D11. Se añadieron D12–D15 para cerrar gaps de validación pre-deploy.

---

## 3. Trail de cambios entre rondas

| Ronda | Fecha | Decisiones introducidas | Decisiones reemplazadas |
|-------|-------|--------------------------|--------------------------|
| **R4** | 2026-05-12 | D1, D2, D3, D4, D5, D6, D7, D8 | — |
| **R5** | 2026-05-12 | D9, D10, D11, D12, D13, D14, D15 | D4 → D9; D7 → D11 |

**Decisiones vigentes finales:** D1, D2, D3, D5, D6, D8, D9, D10, D11, D12, D13, D14. **Plan B:** D15.
**Decisiones obsoletas (no aplicar):** D4, D7.

---

## 4. Decisiones detalladas

### D1 — Container Vast.ai (Ronda 4, vigente)

**Decisión:** Container Docker `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` (last_pushed 2026-03-26, tag status active, cuDNN 8.9+ vía variante `cudnn-devel`).

**Razón:** TF 2.15.0 oficial requiere CUDA 12.2 / cuDNN 8.9 según [tensorflow.org/install/source](https://www.tensorflow.org/install/source); CUDA 12.4 satisface minor compat con drivers 12.x. El wheel `tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (PyPI subido 2023-11-14) confirmado existente. Un solo container con dos virtualenvs (Track A + Track B) reduce complejidad operativa frente a dos containers separados.

**Detalles:**
- Tags Vast.ai históricos verificados (insight `fba73ac3` mnemon 2026-04-16): el repo `vastai/pytorch` está obsoleto (PyTorch 1.0 + CUDA 10.0). Usar `vastai/base-image` para imágenes custom. Tags compuestos para apps específicas siguen patrón `vastai/openwebui:v0.5.7-cuda-12.1-pytorch-2.5.1-py311`, `vastai/vllm:v0.8.1-...`.
- Subcomando `vastai search templates` NO existe en el CLI; los `template_hash` se extraen de la UI `cloud.vast.ai/templates/` y se pasan con `--template_hash`.
- Alternativa robusta si build custom da problemas: imagen oficial `tensorflow/tensorflow:2.15.0-gpu` preinstalada para Track A.

**Fuente:** [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) §"Vast.ai y stack del container".

---

### D2 — GPU RTX 4090 on-demand (Ronda 4, vigente)

**Decisión:** GPU **RTX 4090 on-demand** (Ada Lovelace `sm_89`, 24 GB VRAM, 1008 TFLOPS FP16). Mediana Vast.ai mayo 2026: 0,40 USD/h on-demand, 0,14–0,31 USD/h spot.

**Razón:** SSD MV2 320 + YOLOv8n 416 son modelos pequeños; A100/H100 son desperdicio (batch 32–64 entra holgado en 24 GB). Con 1,72 USD de saldo: 4–12 h disponibles, holgado para 1–3 h de training por track.

**Detalles:**
- Decisión del usuario verbatim (R4): *"Usar la mejor GPU. No me importan costos."* Se elige 4090 sobre 5090 por madurez de drivers CUDA 12.4 y disponibilidad consistente en marketplace.
- **Alternativas (si Vast.ai falla):** RunPod (limitado a PyTorch 2.6–2.9, eliminó 2.1 del catálogo en 2025), Lambda Labs, Paperspace, cluster UAO `uaodeepia11306`. Memoria `1d6d237e` mnemon documenta esta cascada.
- **Quirk Vast.ai SSH (insight `d5f717eb` mnemon 2026-04-16):** `--ssh` reemplaza ENTRYPOINT del container e inyecta `sshd` desde fuera; si `/usr/sbin/sshd` no está en la imagen (caso `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`), Vast.ai intenta `apt-get install openssh-server` en runtime → falla con "Connection refused" si el host bloquea `archive.ubuntu.com:80`. La imagen `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` ya incluye sshd, no aplica.

**Fuente:** [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) §"GPU on-demand y costos".

---

### D3 — Repo HF Hub `mitgar14/embebidos-3-models` (Ronda 4, ✅ creado)

**Decisión:** Repo **privado** `mitgar14/embebidos-3-models` en Hugging Face Hub con estructura dual-track:

```
mitgar14/embebidos-3-models/
├── track_a/
│   ├── runs/
│   ├── checkpoints/   # TF OD API ckpt-*
│   ├── exports/       # detect_int8.tflite + manifest.json + pipeline.config
│   └── logs/          # tfevents para TensorBoard hosted
└── track_b/
    ├── runs/          # logs W&B mirror si se elige stack avanzado
    ├── checkpoints/   # best.pt, last.pt
    ├── exports/       # best.onnx + manifest.json
    └── logs/
```

**Razón:** HF Hub free tier 2026 ofrece 100 GB privado total, sin límite de repos privados, 500 GB max por archivo, sin cap de bandwidth (cf. discover-14). Artefactos totales del proyecto ~250–450 MB, holgado. Xet storage por default desde mayo 2025 (dedup chunks). Permite cambiar a público antes de la entrega 2026-05-26 sin re-uploadear.

**Detalles:**
- Token en `~/.cache/huggingface/token` (autenticado `mitgar14` vía `hf auth login`).
- 10 archivos iniciales: `README.md` + 8 `.gitkeep` (uno por subcarpeta dual-track).
- TensorBoard hosted gratis: HF Hub detecta archivos `tfevents` y monta una instancia automática (cf. [docs HF Hub](https://huggingface.co/docs/hub/tensorboard)).

**Fuente:** [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) §"Persistencia HF Hub".

---

### D4 — ❌ REEMPLAZADA POR D9 (Ronda 4, obsoleta)

**Decisión original:** convertir notebooks `.ipynb` → `.py` con `jupytext --to py:percent` para ejecutar headless.

**Por qué se reemplazó:** el usuario pidió en R5 mantener formato `.ipynb` interactivo con tolerancia a cierre de pestaña. `jupytext --to py:percent` produce un `.py` ejecutable pero rompe la navegabilidad como notebook al reconectar; además, ejecutar el `.py` resultante con `python` no captura outputs en el `.ipynb` original (queda en stdout del proceso).

**Decisión actual:** ver D9.

---

### D5 — Persistencia HF Hub: `CommitScheduler` + `upload_folder` (Ronda 4, vigente)

**Decisión:** Combinar tres patrones del SDK `huggingface_hub`:

1. **`CommitScheduler(every=5)`** — push automático cada 5 minutos al repo HF mientras el training corre. Pattern documentado en `huggingface_hub.utils._commit_scheduler.CommitScheduler`.
2. **`HfApi.upload_folder(run_as_future=True)`** — push asíncrono de checkpoints finales sin bloquear el kernel.
3. **`HfApi.upload_large_folder()`** — push resumible para artefactos grandes (e.g., `track_b/exports/best.engine` si se hiciera, aunque D8 dice que no).

**Razón:** la persistencia debe ser tolerante a fallos de red intermitentes en Vast.ai (saliendo de USA). `CommitScheduler` cada 5 min limita la pérdida en caso de crash; `run_as_future=True` evita bloquear si la red es lenta; `upload_large_folder` reanuda chunks.

**Detalles:**
- Snippet bootstrap Track A:
  ```python
  from huggingface_hub import CommitScheduler
  scheduler = CommitScheduler(
      repo_id="mitgar14/embebidos-3-models",
      repo_type="model",
      folder_path="/workspace/embebidos-3/track_a/checkpoints",
      path_in_repo="track_a/checkpoints",
      every=5,  # minutos
      private=True,
      squash_history=False,  # mantener historial completo
  )
  ```
- TensorBoard via `HFSummaryWriter` (subclase de `torch.utils.tensorboard.SummaryWriter`):
  ```python
  from huggingface_hub import HFSummaryWriter
  writer = HFSummaryWriter(
      repo_id="mitgar14/embebidos-3-models",
      logdir="/workspace/embebidos-3/track_a/logs",
      commit_every=5,
  )
  ```

**Fuente:** [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) §"HF Hub upload patterns".

---

### D6 — Logging stack dual (Ronda 4, vigente)

**Decisión:**
- **Track B (YOLOv8):** `yolo train wandb=True` — integración nativa de W&B en Ultralytics 8.4.46. Crear proyecto W&B `embebidos-3` con run name `track_b_yolov8n`. Free tier W&B 2026: sin cap conocido en runs.
- **Track A (TF OD API):** TensorBoard local en `model_dir/` + `HFSummaryWriter` para push automático al repo HF. Hub detecta `tfevents` y monta TensorBoard hosted gratis.

**Razón:** W&B free tier es generoso y la integración nativa de Ultralytics evita boilerplate. Para Track A, TFOD API no integra W&B fácilmente (requeriría callbacks custom en `model_main_tf2.py`) → TensorBoard es el path nativo, y HF Hub lo hostea gratis.

**Detalles:**
- Heartbeat custom de los notebooks actuales se conserva (logs cada N segundos a stdout) como red de seguridad si W&B o TensorBoard fallan.
- Variables de entorno requeridas: `WANDB_API_KEY` (Track B), `HF_TOKEN` (ambos).

**Fuente:** [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) §"Logging".

---

### D7 — ❌ REEMPLAZADA POR D11 (Ronda 4, obsoleta)

**Decisión original:** `trap EXIT` en `run.sh` para auto-destroy de la instancia Vast.ai al terminar el training.

**Por qué se reemplazó:** `trap EXIT` requiere un proceso bash longevo. Al ejecutar el notebook como kernel detached con `nbconvert --execute --inplace` dentro de `tmux`, el bash inicial (entrypoint del container) muere temprano y el `trap` se dispara antes de que termine el training. El proceso del kernel sobrevive a `trap`, pero el shutdown nunca se dispara.

**Decisión actual:** ver D11 (cron watchdog + última celda).

---

### D8 — Engine TRT compilado en el Nano (Ronda 4, vigente)

**Decisión:** El archivo `.engine` de TensorRT **siempre se compila en el propio Jetson Nano**, nunca en Vast.ai ni transferido entre máquinas.

**Razón:** TensorRT engines son **GPU-architecture-specific y TRT-version-specific**. Un engine compilado en RTX 4090 (Ada `sm_89`, TRT 10) NO ejecutará en Jetson Nano (Maxwell `sm_53`, TRT 8.2.1). Incluso con la misma versión de TRT (8.2.1 vía Docker NGC en Vast.ai), las fusiones de kernels difieren entre `sm_89` y `sm_53` por el catálogo de tactics disponible.

**Detalles:**
- En Vast.ai se exporta hasta `.onnx` (Track B); el `.engine` se construye en el Nano con `trtexec`:
  ```bash
  # En el Nano (post-deploy)
  trtexec --onnx=best.onnx \
          --fp16 \
          --workspace=4096 \
          --saveEngine=best.engine
  ```
- Track A no toca TRT (corre en CPU TFLite XNNPACK).
- El gate de validación en Docker NGC `21.11-py3` (D13) sirve como **gate necesario pero no suficiente** del parsing ONNX. La validación final del engine en hardware Maxwell solo se hace en el Nano.

**Fuente:** [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §"Limitación x86 vs aarch64 Maxwell".

---

### D9 — Mantener `.ipynb` interactivo + `tmux + nbconvert` (Ronda 5, vigente, **reemplaza D4**)

**Decisión:** Mantener formato `.ipynb` para los notebooks de training. Ejecutar con `jupyter nbconvert --execute --inplace` (o `papermill` equivalente) **dentro de una sesión `tmux` detached**. La UI de JupyterLab sirve para abrir el `.ipynb` al reconectar y revisar resultados; la ejecución pasa por un proceso bash detached que escribe outputs al `.ipynb` en disco.

**Razón:** Único patrón verificado que combina:
- (a) preservación del `.ipynb` con outputs visibles al reabrir desde la UI.
- (b) supervivencia del kernel al cierre de pestaña (`tmux` mantiene el proceso vivo).
- (c) recuperación **completa** del output al reconectar a JupyterLab horas después.

JupyterLab + kernel detached "puro" desde la UI NO recupera output al reconectar — solo el output guardado en el `.ipynb` vía autosave (intervalo 120 s) sobrevive, perdiendo lo emitido entre autosaves.

**Detalles técnicos (config crítica de JupyterLab):**
```python
# /root/.jupyter/jupyter_server_config.py
c.MappingKernelManager.cull_idle_timeout = 0       # deshabilita el culler
c.MappingKernelManager.cull_busy = False
c.MappingKernelManager.cull_connected = False
c.ServerApp.shutdown_no_activity_timeout = 0
c.ZMQChannelsWebsocketConnection.iopub_msg_rate_limit = 0       # sin truncado verboso
c.ZMQChannelsWebsocketConnection.iopub_data_rate_limit = 10_000_000  # 10 MB/s
```

**Comando de ejecución canónico:**
```bash
tmux new-session -d -s training
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
```

**Trampas conocidas:**
- "Notebook is too large to be saved" cuando output > ~25 MB. Mitigación: `%%capture` en celdas de instalación, `IPython.display.clear_output()` antes de loops verbosos, log a archivo con `tqdm.write()`.
- Buffer overflow del frontend con `iopub_data_rate_limit=0`: log verboso a archivo, métricas resumidas con `tqdm` en pantalla.
- WebSocket ping/pong timeout: usar `--ping-interval 30` al lanzar JupyterLab.

**Fuente:** [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) §"Notebook persistente y tmux".

---

### D10 — Dos `uv venv` separados (Ronda 5, vigente)

**Decisión:** Crear dos virtualenvs `uv` independientes en `/opt/venv/tracka` y `/opt/venv/trackb`, registrados como kernels `ipykernel` separados:

- Kernel `tracka` display name `Track A (TF 2.15)`: TF 2.15 + `tf-models-official` 2.15 + Pillow 10.4 + protobuf 3.20.3 + grpcio-tools 1.64.1 + huggingface_hub + ipykernel.
- Kernel `trackb` display name `Track B (YOLOv8)`: PyTorch 2.1.0+cu121 + torchvision 0.16.0+cu121 + Ultralytics 8.4.46 + `numpy<2.0` + huggingface_hub + wandb + onnxslim + roboflow + ipykernel.

**Razón:**
1. **Deps incompatibles:** TF 2.15 pin `protobuf 3.20.3` + Ultralytics 8.4.46 con `numpy<2.0` y torch CUDA 12.1 son intrínsecamente irreconciliables en un solo venv (resolver de pip/uv no encuentra solución).
2. **JupyterLab aísla por kernel:** abre un proceso Python independiente por kernel; `libcudart`, `libcudnn`, `libnvinfer` se cargan independientemente. No hay conflicto de runtime.
3. **Overhead mínimo:** `uv venv` es 10-100x más rápido que `python -m venv`; resolver dual es trivial.

**Tres opciones evaluadas:**

| Opción | Lockfile | Aislamiento | Decisión |
|--------|----------|-------------|----------|
| Monorepo único con `[project.optional-dependencies]` + `[tool.uv] conflicts` | Único, compartido | Mismo venv, mismo proceso | ❌ Overhead innecesario |
| Dos `pyproject.toml` invocados con `uv sync --project <archivo>` | Independiente | Venvs separados | ❌ **GAP:** `--project` acepta directorios, no archivos |
| **Dos `uv venv` separados con `uv pip install`** | Sin lockfile (deps explícitas en bootstrap) | Procesos completamente aislados | ✅ **ELEGIDA** |

**Comandos canónicos:**
```bash
# Track A
uv venv /opt/venv/tracka --python 3.10
source /opt/venv/tracka/bin/activate
uv pip install tensorflow==2.15.0 tf-models-official==2.15.0 \
  "Pillow==10.4.0" "protobuf==3.20.3" "grpcio-tools==1.64.1" \
  huggingface_hub ipykernel
python -m ipykernel install --user --name tracka \
  --display-name "Track A (TF 2.15)"

# Track B
uv venv /opt/venv/trackb --python 3.10
source /opt/venv/trackb/bin/activate
uv pip install "numpy<2.0"
uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install ultralytics==8.4.46 huggingface_hub wandb onnxslim \
  roboflow ipykernel
python -m ipykernel install --user --name trackb \
  --display-name "Track B (YOLOv8)"
```

**Validación post-instalación:**
```bash
jupyter kernelspec list
# Esperado:
#   tracka     ~/.local/share/jupyter/kernels/tracka
#   trackb     ~/.local/share/jupyter/kernels/trackb
cat ~/.local/share/jupyter/kernels/trackb/kernel.json
# El campo "argv" debe apuntar a /opt/venv/trackb/bin/python
```

**Fuente:** [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) §"uv dual venv".

---

### D11 — Auto-destroy: cron watchdog + última celda (Ronda 5, vigente, **reemplaza D7**)

**Decisión:** Patrón de tres componentes para auto-destroy compatible con notebook detached:

1. **Cron watchdog interno** (frecuencia 1 min) instalado en bootstrap:
   ```bash
   echo "* * * * * test -f /workspace/embebidos-3/.training_done && \
     vastai destroy instance \${VAST_CONTAINERLABEL#C.} 2>&1 | \
     tee -a /workspace/embebidos-3/logs/watchdog.log" | crontab -
   ```
2. **Última celda del notebook** que sube artefactos finales a HF Hub, crea el archivo señal, y dispara el destroy como plan B inmediato:
   ```python
   # Última celda — auto-destroy señaling
   import subprocess, os, time
   from pathlib import Path
   # ... HfApi.upload_folder(folder_path="track_b/exports", ...) ...
   Path("/workspace/embebidos-3/.training_done").touch()
   print(f"[{time.strftime('%H:%M:%S')}] Training done. Watchdog will destroy instance within 60s.")
   container_id = os.environ.get("VAST_CONTAINERLABEL", "").lstrip("C.")
   if container_id:
       subprocess.run(["vastai", "destroy", "instance", container_id], check=False)
   ```
3. **Variable `$VAST_CONTAINERLABEL`** disponible como env var del container (formato `C.<id>`, extraer con `${VAST_CONTAINERLABEL#C.}`).

**Razón:**
- **Vast.ai CLI NO tiene flags `--auto-stop`, `--max-runtime` ni `--idle-timeout`** en `create instance`. GAP confirmado leyendo `vast-ai/vast-python` y [docs.vast.ai/cli/reference/create-instance](https://docs.vast.ai/cli/reference/create-instance). Los flags `--end_date`, `--day`, `--hour` pertenecen a `add_scheduled_job`, no a `create instance`.
- **Vast.ai NO tiene "Idle Shutdown" automático** basado en utilización de GPU. GAP confirmado: la doc no menciona ningún mecanismo de shutdown por inactividad. Vast.ai cobra por segundo de instancia activa pero no detiene automáticamente.
- **HF Hub webhook → AWS Lambda → `vastai destroy`** añade 3 puntos de fallo para ahorrar US$0,05; no vale la pena para proyecto académico.

**Detalles:**
- API key dentro del container: precedencia `--api-key <KEY>` > `$VAST_API_KEY` (env var) > `~/.config/vastai/vast_api_key` (archivo). Inyectar con `--env '-e VAST_API_KEY=<key>'` al crear instancia o con `vastai set api-key <KEY>` en `onstart` (más seguro, escribe al archivo).
- Confirmación FAQ Vast.ai verbatim: *"A special instance API key is pre-installed. Install the CLI and use it: `pip install vastai` / `vastai stop instance $CONTAINER_ID`"* — [docs.vast.ai/guides/reference/faq/instances](https://docs.vast.ai/guides/reference/faq/instances).

**Fuente:** [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) §"Auto-destroy y watchdog".

---

### D12 — Validación TFLite pre-deploy (Ronda 5, vigente)

**Decisión:** Tres sub-pasos obligatorios desde Vast.ai antes de bajar el `.tflite` al Nano:

1. **Inspección de `op_version`** con paquete `tflite==2.5.0` (PyPI):
   ```python
   import tflite.Model
   with open("track_a/exports/model.tflite", "rb") as f:
       buf = bytearray(f.read())
   model = tflite.Model.Model.GetRootAsModel(buf, 0)
   sg = model.Subgraphs(0)
   op_codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]
   for i in range(sg.OperatorsLength()):
       op = sg.Operators(i)
       code = op_codes[op.OpcodeIndex()]
       print(f"Op {i}: builtin={code.BuiltinCode()} version={code.Version()}")
   ```
2. **Carga test** con `tflite_runtime==2.5.0.post1` wheel Coral CP38 x86 (idéntico a runtime del Nano):
   ```bash
   pip install "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl"
   python -c "
   import tflite_runtime.interpreter as tflite
   interp = tflite.Interpreter('track_a/exports/model.tflite')
   interp.allocate_tensors()
   print('OK: runtime 2.5 acepta el modelo')
   "
   ```
3. **Export con flags conservadores** del converter TFLite:
   ```python
   converter.experimental_new_quantizer = False  # cuantizador legacy
   converter.experimental_new_converter  = False # converter TOCO legacy
   ```

**Razón:** TF 2.15 converter puede generar `op_version` > 2.5 max para ops como `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`. Bug histórico confirmado en issues [tensorflow/tensorflow#41943](https://github.com/tensorflow/tensorflow/issues/41943) (`mgalgs`, 2020), [#50652](https://github.com/tensorflow/tensorflow/issues/50652) (`djbacad`, TF 2.5.0 Python 3.6.9, 2021), [#43232](https://github.com/tensorflow/tensorflow/issues/43232) (`juanpbotero98`, RPi, 2020).

**Tabla de op_version por riesgo (TF 2.15 INT8 PTQ → runtime 2.5 max):**

| Op | Versión máx TFLite 2.5 | Versión generada por TF 2.15 INT8 PTQ | Riesgo |
|----|------------------------|----------------------------------------|--------|
| `CONV_2D` | 5 | 5 con activaciones estándar | Bajo |
| `DEPTHWISE_CONV_2D` | 4 | 4–5 según flags | **Medio** |
| `FULLY_CONNECTED` | 4 | 4 para MobileNet v2 | Bajo |
| `QUANTIZE` / `DEQUANTIZE` | 2 | 2 | Bajo |
| `PAD`, `ADD`, `MUL`, `RESHAPE`, `CONCATENATION` | 1–2 | 1–2 | Ninguno |
| `TFLite_Detection_PostProcess` | custom | custom (ver D15) | Ver D15 |

**GAP residual:** las versiones exactas de `DEPTHWISE_CONV_2D` y `FULLY_CONNECTED` que TF 2.15 genera con INT8 PTQ desde TFOD API no están documentadas públicamente. Verificación obligatoria post-conversión vía inspección del flatbuffer (paso 1 arriba).

**Workaround si falla:** archivo `tensorflow/lite/tools/flatbuffer_utils.py` permite bajar manualmente el campo `version` de cada operador en el flatbuffer. Frágil, no documentado oficialmente.

**Fuente:** [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §"Gate Track A (TFLite)".

---

### D13 — Validación ONNX pre-deploy vía Docker NGC `tensorrt:21.11-py3` (Ronda 5, vigente)

**Decisión:** Validación obligatoria del `.onnx` exportado de Track B con Polygraphy + TRT 8.2.1.8 idéntico al JetPack 4.6.1:

```bash
docker pull nvcr.io/nvidia/tensorrt:21.11-py3
docker run --rm --gpus all \
  -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tensorrt:21.11-py3 \
  bash -c "
    pip install -q polygraphy onnx &&
    polygraphy run /workspace/track_b/exports/best.onnx \
      --onnxrt --trt \
      --atol 1e-2 --rtol 1e-2 \
      --input-shapes images:[1,3,416,416]
  "
```

**Razón crítica:** si en el container Vast.ai (CUDA 12.4) corres `pip install polygraphy tensorrt`, obtienes **TRT 10+, no 8.2**. Para validar contra el TRT exacto del Nano (8.2.1.8), **es obligatorio** usar el Docker NGC `21.11-py3` que tiene TRT 8.2.1 + CUDA 11.5 + Ubuntu 20.04.

**Imágenes NGC TensorRT candidatas:**

| Imagen NGC | Versión TensorRT | CUDA | Ubuntu | Python | Coincide JetPack 4.6.1 |
|------------|-------------------|------|--------|--------|------------------------|
| `nvcr.io/nvidia/tensorrt:21.10-py3` | 8.0.x | 11.4 | 20.04 | 3.8 | No |
| **`nvcr.io/nvidia/tensorrt:21.11-py3`** | **8.2.1** | 11.5 | 20.04 | 3.8 | ✅ **Sí** |
| `nvcr.io/nvidia/tensorrt:22.01-py3` | 8.2.3 | 11.5 | 20.04 | 3.8 | Aproximado (no exacto) |

**Polygraphy 0.49.x soporta TRT 8 Y TRT 10.** Fuente: [`NVIDIA/TensorRT/tools/Polygraphy/CHANGELOG.md`](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/CHANGELOG.md) v0.49.5 (2024-01-16): *"Fixed a bug where `explicit_batch` would be provided by default on TRT 10.0, where it has been removed."* PyPI lista hasta 0.49.27.

**Limitación arquitectural (x86 vs aarch64 Maxwell):** la validación en x86 es **gate necesario pero no suficiente**:

| Aspecto | x86 + TRT 8.2 vía Docker | aarch64 `sm_53` Nano real |
|---------|--------------------------|----------------------------|
| Validación parser ONNX | ✅ | ✅ |
| Detección ops fuera de opset | ✅ | ✅ |
| Comparación numérica `--onnxrt` | ✅ | ✅ |
| Tiempos reales en Maxwell | ❌ | ✅ |
| Fusiones de kernels Maxwell | ❌ | ✅ |
| Comportamiento INT8 calibrador | ❌ (no hay TC) | ✅ |

**Ops blacklist TRT 8.2 verificada** contra [onnx-tensorrt/docs/operators.md `release/8.2-GA`](https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md):

- `GridSample` — **ausente de la lista**, riesgo si aparece (solo en YOLOv8-seg, no detección).
- `ConstantOfShape` — soportado **FP32 únicamente**; riesgo si onnxslim genera tipos INT64.
- `NonMaxSuppression` — `[EXPERIMENTAL]` FP32 only; evitar con `nms=False` en export.
- `DFT`, `IsInf`, `IsNaN`, `MelWeightMatrix`, `STFT`, `SequenceInsert`, `CumSum` — no soportados.
- `Reciprocal`, `NonZero`, `RoiAlign`, `QLinearConv/MatMul` — no soportados (improbables en YOLOv8n detect).

**Fuente:** [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §"Gate Track B (ONNX)".

---

### D14 — Track B FP16-only por default + experimento INT8 opcional (Ronda 5, vigente)

**Decisión:** Track B se queda en **FP16-only por default**. Experimento INT8 opcional de **45–60 min** en el propio Jetson Nano con criterio binario:

- **Si** `FPS_INT8 < FPS_FP16 × 1,10` (menos de 10% de ganancia) **O** `mAP_INT8 < mAP_FP16 − 5 pp`, **abandonar INT8** y consolidar FP16.

**Razón decisiva (mecanismo de hardware):** Maxwell `sm_53` **carece de la instrucción `dp4a`** (dot product de 4 × INT8) introducida en Pascal `sm_61` (2016). Sin `dp4a`, TensorRT tiene tres opciones:

1. **Usar kernels CUDA INT8 SIMD vía `dp4a`** → NO disponible en `sm_53`.
2. **Emular INT8 vía FP16/FP32** → elimina cualquier beneficio de velocidad y añade overhead.
3. **Mixed precision fallback** → TensorRT revierte la capa a FP16, generando grafo mixto con conversiones adicionales.

Confirmación NVIDIA (issue [`NVIDIA/TensorRT#3762`](https://github.com/NVIDIA/TensorRT/issues/3762)): *"`--int8` means Enable int8 precision, in addition to fp32."* INT8 nunca reemplaza FP32, lo complementa; las capas no cuantizables revierten.

**Evidencia contradictoria de la literatura (gap irreductible):**

| Fuente | FPS reportado YOLOv8n | mAP | Conclusión |
|--------|------------------------|-----|------------|
| [Qengineering/YoloV8-TensorRT-Jetson_Nano](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano) (rama `tensorrt8`) | FP16: 19 FPS — INT8: no publica tabla | mAP INT8 *"significantly worse"* | INT8 inútil en Nano B01 |
| [espstack.com YOLOv8 on Jetson Nano](https://espstack.com/blogs/posts/yolov8-jetson-nano.html) | FP16: 18–22 — INT8: 28–32 (+50%) | mAP50 FP16 0,885 → INT8 0,878 | INT8 mejora; pero sin metodología verificable, sin código reproducible |
| [the0807/YOLOv8-ONNX-TensorRT](https://github.com/the0807/YOLOv8-ONNX-TensorRT) (Orin Nano `sm_87` con TC INT8) | FP16: 60 — INT8: 63 (+5%) | mAP50-95 FP16 37,1 → INT8 33,0 (−4,1 pp) | Even en hardware moderno con TC INT8 el speedup es modesto y mAP cae |

**Conclusión:** el speedup INT8 en `sm_53` es **estructuralmente nulo** por arquitectura de hardware. La degradación de mAP ocurre igualmente (cuantización modifica pesos y activaciones independientemente del hardware) pero sin la contraparte de velocidad.

**Protocolo experimento INT8 opcional en Nano (45–60 min):**
```bash
# Solo en el Nano (post-deploy)
trtexec --onnx=yolov8n_custom.onnx \
        --saveEngine=yolov8n_int8.engine \
        --int8 \
        --calib=calib_list.txt \
        --workspace=1024

# Medir mAP@0.5 en val set completo
python validate_engine.py --engine yolov8n_int8.engine --data data.yaml

# Medir FPS empírico
trtexec --loadEngine=yolov8n_int8.engine --iterations=100
```

**Si el experimento queda en zona gris (entre +0% y +10% FPS):** abandonar (criterio binario del 10%).

**Fuente:** [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §"INT8 Maxwell `sm_53`".

---

### D15 — Plan B Coral wheel CP36 aarch64 si falla `TFLite_Detection_PostProcess` (Ronda 5, plan B)

**Decisión:** Si el wheel NVIDIA `tensorflow==2.5.0+nv21.8` (instalado en el Nano vía JetPack) lanza al ejecutar el `.tflite`:

```
Didn't find custom op TFLite_Detection_PostProcess
```

**Aplicar fallback** al wheel oficial Coral:

```
tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
URL: https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
sha256: 7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df
```

Fuente verbatim: [google-coral.github.io/py-repo/tflite-runtime/](https://google-coral.github.io/py-repo/tflite-runtime/).

**Razón:** `TFLite_Detection_PostProcess` no es un Select TF op; es un op nativo compilado en `tensorflow/lite/kernels/detection_postprocess.cc`. **Debería estar incluido** en cualquier build completo de TFLite. Pero **no hay confirmación verbatim** para el wheel NVIDIA específico — el wheel Coral fue compilado por Google para el Edge TPU con el op incluido, sirve igual en el Nano (no requiere TPU).

**Configuración exacta del Nano:** Python 3.6, aarch64, JetPack 4.6.1 → coincide con `cp36-cp36m-linux_aarch64`.

**Uso en el código de inferencia del Nano:**
```python
import tflite_runtime.interpreter as tflite  # NO `from tensorflow.lite`
interp = tflite.Interpreter("model.tflite")
interp.allocate_tensors()
```

**Detección anticipada en Vast.ai** (Gate D12, paso 2 con wheel x86 equivalente):
```python
try:
    interp = tflite.Interpreter("track_a/exports/model.tflite")
    interp.allocate_tensors()
    print("OK: TFLite_Detection_PostProcess registrado en runtime 2.5")
except ValueError as e:
    if "TFLite_Detection_PostProcess" in str(e):
        print("FALLO: custom op no encontrado. Aplicar D15 fallback en el Nano.")
    else:
        raise
```

**Fuente:** [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §"Plan B Coral wheel CP36 aarch64".

---

## 5. GAPs residuales y mitigaciones (resumen)

| # | GAP / Riesgo | Mitigación documentada |
|---|--------------|-------------------------|
| R1 | `op_version` de `DEPTHWISE_CONV_2D` y `FULLY_CONNECTED` en TF 2.15 INT8 PTQ sin docs públicas | Inspección obligatoria flatbuffer (D12). Workaround `flatbuffer_utils.py` si falla. |
| R2 | Wheel NVIDIA `tensorflow==2.5.0+nv21.8` y `TFLite_Detection_PostProcess` sin confirmación verbatim | Fallback Coral wheel CP36 aarch64 (D15). |
| R3 | Drop INT8 YOLOv8n Maxwell `sm_53` no caracterizado en literatura | Cerrado por mecanismo (`dp4a` ausente). FP16-only por default (D14). Experimento opcional en Nano. |
| R4 | Polygraphy en container Vast.ai (CUDA 12.4) instala TRT 10 por defecto, no 8.2 | Docker NGC `tensorrt:21.11-py3` obligatorio (D13). |
| R5 | Validación x86 con TRT 8.2.1 ≠ Maxwell `sm_53` (fusiones de kernels difieren) | Gate necesario no suficiente: verificación rápida en Nano antes de training completo. |
| R6 | Vast.ai sin idle-shutdown ni `--auto-stop` | Cron watchdog + última celda (D11). |
| R7 | Output del notebook > 25 MB rompe autosave de JupyterLab | `%%capture` en instalaciones, `tqdm` resumido, log verboso a archivo. |
| R8 | Roboflow SDK 1.3.9 (mayo 2026) sin fix de bug `dataset.location` | Cascada de búsqueda + `os.chdir(WORK_DIR)` + heurística por contenido (ver [`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md)). |
| R9 | `jupyter execute` / `papermill` pueden fallar silenciosamente con kernels custom | Especificar `--ExecutePreprocessor.kernel_name=trackb` explícito. Validar con `jupyter kernelspec list` antes. |
| R10 | Drop INT8 en zona gris (entre +0% y +10% FPS) | Criterio binario explícito en D14 (margen < 10% → abandonar). |

---

## 6. Cómo usar este ledger

- **Para tomar decisiones nuevas:** verificar que no entren en conflicto con D1–D15. Si lo hacen, abrir ronda `/investiga` documentada.
- **Para implementar (#2'-#5' del HANDOFF):** cada decisión apunta a su doc temático en el campo "Fuente"; ahí están los comandos, flags y validaciones copy-paste.
- **Para auditar:** la tabla de §1 es el listado canónico; las secciones §4 dan el contexto verbatim.
- **Para entender la historia:** la sección §3 trail muestra qué Ronda introdujo qué decisión y cuáles se reemplazaron.

---

## 7. Documentos hermanos

| Doc | Foco | Decisiones relacionadas |
|-----|------|--------------------------|
| [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) | Hardware target, stack TF/PyTorch, pin SHA TF Models, por qué Vast.ai sobre Colab/Kaggle | D1, D2 |
| [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) | Notebook persistente, `tmux + nbconvert`, `uv` dual venv, HF Hub upload, auto-destroy, `bootstrap.sh` | D3, D5, D6, D9, D10, D11 |
| [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) | Gates TFLite y ONNX, Polygraphy Docker NGC, drop INT8 Maxwell, plan B Coral | D8, D12, D13, D14, D15 |
| [`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md) | Dataset Roboflow `embebidos3/waste-3class-lwld8`, Ultralytics 8.4.46, `numpy<2.0`, export ONNX opset 11, W&B nativo | D6 (W&B), D8 (export ONNX) |
| [`HANDOFF-implementacion-vastai-hf.md`](HANDOFF-implementacion-vastai-hf.md) | Plan operativo §5 (#2'-#5'), gotchas, credenciales, lectura prioritaria | Todas |

---

**Fin del ledger.** Cualquier cambio a D1–D15 requiere nueva ronda `/investiga`.
