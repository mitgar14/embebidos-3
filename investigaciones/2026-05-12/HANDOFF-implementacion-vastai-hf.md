# Handoff — Implementación Vast.ai + HF Hub para embebidos-3

> **Fecha de cierre de sesión:** 2026-05-12 (Ronda 5 cerrada; consolidación §11 ejecutada)
> **Sesión origen:** Claude Code (Opus 4.7), conversaciones con context compaction
> **Estado:** Rondas 4 y 5 cerradas. Repo GitHub pusheado, HF Hub repo creado, docs consolidados a 5 archivos autocontenidos. **Implementación de scripts y notebooks NO iniciada.**
> **Próxima sesión:** ejecutar plan operativo §5 (#2'-#5').

Este documento es un handoff completo. Léelo de arriba a abajo antes de
ejecutar cualquier acción. Asume que el nuevo agente no tiene memoria
de la sesión previa.

---

## 1. Contexto del proyecto

**`embebidos-3`** — clasificador de residuos para entrega académica de
la materia *IA en Embebidos* (Prof. Juan Camilo Giraldo, UAO Cali,
semestre 7). Deadline: **2026-05-26**.

- **3 clases:** glass, paper, plastic
- **Hardware target:** Jetson Nano Developer Kit 4 GB **B01**
- **JetPack:** 4.6.1 (fijo, no actualizable)
- **Cámara:** Logitech C920 OG (Rev 1, PID `046d:082d`) — pipeline
  GStreamer canónico ya validado en
  `investigaciones/2026-05-10/2026-05-10-camara-usb-jetson-nano.md`
- **Working directory:** `C:\Users\mitgar14\Documentos\embebidos-3`

### Arquitectura dual

| Track | Modelo | Input | Export final | Runtime en Nano |
|-------|--------|-------|--------------|-----------------|
| **A** | SSD MobileNet v2 plain (no FPN) | 320×320 | `.tflite` INT8 PTQ | CPU + XNNPACK + NEON |
| **B** | YOLOv8n | 416×416 | `.onnx` opset 11 → `.engine` FP16 | GPU Maxwell `sm_53` |

### Constraints hardware inmutables (Jetson Nano B01)

- Maxwell **128 CUDA cores, SIN Tensor Cores INT8, sin instrucción `dp4a`**
- Python 3.6.9 (system)
- TensorFlow 2.5+nv21.8 (NVIDIA build oficial)
- TFLite runtime 2.5
- TensorRT 8.2.1.8
- CUDA 10.2 + cuDNN 8.2.1
- L4T R32.7.x, kernel Linux 4.9.337

**Implicación crítica (gap residual cerrado en R5):** Maxwell sm_53 no
tiene `dp4a` (Pascal sm_61+) → el speedup INT8 es estructuralmente
nulo. Track B se queda en **FP16-only por default** (D14); experimento
INT8 opcional en el propio Nano con criterio binario.

---

## 2. Por qué migramos a Vast.ai

Track A (Colab) se intentó con `condacolab` pero falló por dos razones
encadenadas:

1. **`condacolab.check()` lanza `AssertionError`**, no devuelve `False`.
   Fix: llamada directa a `install_from_url(MINIFORGE_URL)`.
2. **Miniforge 23.11.0-0 trae Python 3.12** (no 3.10 como sugerían las
   release notes). Intento de downgrade con
   `mamba install python=3.10` falla porque `google-colab` pin a 3.12
   bloquea la resolución del solver.

**Decisión del usuario (verbatim):** *"Quiero mantener las arquitecturas
y frameworks que han sido demostradas como estables y compatibles
dentro de la Jetson Nano. Usar la mejor GPU. No me importan costos,
pero mantener logging robusto y persistencia de TODO en HF Hub."*

**Decisión del usuario Ronda 5 (verbatim):** *"Necesito negociar [...]
correrlo como notebook [...] que su ejecución no se afecte porque
cerré la pestaña [...] todo sea lo más compatible entre sí (ver uv)
[...] garantizar que los productos de esta fase sean compatibles/se
puedan usar desde la Jetson Nano de forma estable y robusta."*

Vast.ai tiene saldo confirmado: **1,72 USD** disponibles.

---

## 3. Decisiones vinculantes consolidadas D1–D15

Decisiones D1–D8 (Ronda 4) y D9–D15 (Ronda 5), consolidadas en
`decisiones-D1-D15-ledger.md` (índice maestro con cross-refs a los
otros 4 docs).

**No cambiar sin nueva ronda `/investiga`.**

| # | Decisión | Estado |
|---|----------|--------|
| **D1** | Container Vast.ai `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` | Vigente |
| **D2** | GPU RTX 4090 on-demand (0,35–0,50 USD/h) | Vigente |
| **D3** | Repo HF Hub `mitgar14/embebidos-3-models` privado con subcarpetas `track_a/{runs,checkpoints,exports,logs}` y `track_b/{...}` | ✅ Creado |
| **D4** | ~~jupytext `--to py:percent` para convertir `.ipynb` → `.py` headless~~ | **REEMPLAZADA por D9** |
| **D5** | `HfApi.upload_folder(run_as_future=True)` + `CommitScheduler(every=5)` + `upload_large_folder` resumible | Vigente |
| **D6** | Track B → W&B nativo (`yolo train wandb=True`). Track A → TensorBoard hosted en HF Hub vía `HFSummaryWriter` | Vigente |
| **D7** | ~~Auto-destroy con `trap EXIT` en `run.sh`~~ | **REEMPLAZADA por D11** |
| **D8** | Engine TRT siempre compilado **en el Nano**, nunca transferido | Vigente |
| **D9** | Mantener `.ipynb` interactivo. Ejecutar con `jupyter nbconvert --execute --inplace` dentro de `tmux` (o papermill equivalente) | Vigente |
| **D10** | Dos `uv venv` separados (`/opt/venv/tracka` + `/opt/venv/trackb`) registrados como kernels `ipykernel` distintos (`tracka` y `trackb`) | Vigente |
| **D11** | Auto-destroy vía cron watchdog interno + última celda del notebook que crea `/workspace/embebidos-3/.training_done` y llama `vastai destroy instance ${VAST_CONTAINERLABEL#C.}` | Vigente |
| **D12** | Validación TFLite pre-deploy: `tflite==2.5.0` para inspeccionar `op_version`, carga test con wheel Coral `tflite_runtime-2.5.0.post1` CP38, export con `experimental_new_quantizer=False` y `experimental_new_converter=False` | Vigente |
| **D13** | Validación ONNX pre-deploy vía Docker `nvcr.io/nvidia/tensorrt:21.11-py3` (TRT 8.2.1.8 idéntico a JetPack 4.6.1). `polygraphy run --trt --onnxrt --atol 1e-2 --rtol 1e-2 --input-shapes images:[1,3,416,416]` | Vigente |
| **D14** | Track B **FP16-only por default**. Experimento INT8 opcional 45–60 min en Nano. Criterio binario: si `FPS_INT8 < FPS_FP16 × 1.10` O `mAP_INT8 < mAP_FP16 − 5 pp`, abandonar | Vigente |
| **D15** | Si wheel NVIDIA `tensorflow==2.5.0+nv21.8` no incluye `TFLite_Detection_PostProcess`, fallback wheel Coral `tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl` | Plan B |

---

## 4. Archivos relevantes en el repo

```
C:\Users\mitgar14\Documentos\embebidos-3\
├── notebooks\
│   ├── train_track_a_ssd.ipynb       # Notebook con bootstrap defensivo Colab (a refactorizar, NO convertir a .py)
│   └── train_track_b_yolov8.ipynb    # Notebook Colab Track B (no intervenido aún)
├── investigaciones\
│   ├── 2026-05-05\                   # Rondas 1-2 (arquitectura, datasets, dual-track, preprocessing)
│   ├── 2026-05-10\                   # Cámara USB Jetson Nano
│   └── 2026-05-12\                   # Rondas 4-5 (consolidados, ver §11)
│       ├── HANDOFF-implementacion-vastai-hf.md              # ESTE DOCUMENTO
│       ├── decisiones-D1-D15-ledger.md                      # Índice maestro D1-D15 + trail entre rondas (602 líneas)
│       ├── compatibilidad-stack-cloud-jetson.md             # Hardware Nano + por qué Vast.ai + stack Track A/B (636 líneas)
│       ├── infraestructura-training-vastai-uv-hf.md         # uv dual venv + tmux+nbconvert + CommitScheduler + W&B (999 líneas)
│       ├── validacion-artefactos-pre-deploy.md              # 4 gates TFLite/ONNX + Polygraphy NGC + plan B Coral (846 líneas)
│       └── dataset-roboflow-yolov8.md                       # Dataset 416×416 + bug Roboflow SDK + export ONNX (679 líneas)
├── docs\
│   ├── NV_Jetson_Nano_Developer_Kit_User_Guide.pdf
│   └── validacion-camara-windows.md
├── scripts\
│   ├── bench_jetson.py               # Harness FPS/latencia/RAM en Nano (R1-2, ya existía)
│   └── archive\                      # 4 archivos legacy
├── prototipos\                       # 4 PNG/JPG/paint de prototipos físicos
├── main.py                            # Stub
├── pyproject.toml                     # uv project (Python ≥3.10)
├── uv.lock
├── README.md                          # Documentación dual-track (escrito antes R4-R5)
└── .gitignore                         # Actualizado: .obsidian/, .env*, modelos, wandb/runs/outputs
```

**Estado del repo:**
- Commit inicial `387c485` con 43 archivos / 10.463 inserciones.
- Repo GitHub privado `mitgar14/embebidos-3` ✅ pusheado, branch `main` tracking `origin/main`.
- Repo HF Hub privado `mitgar14/embebidos-3-models` ✅ creado con 10 archivos (README + 8 `.gitkeep` + estructura dual-track).
- `gh auth setup-git` aplicado (workaround Git Credential Manager → token gh).

**Notebook Track A — estado actual:**
- `cell-8 v4` con bootstrap defensivo idempotente (`if "banner" not in dir(): ...`)
- Flujo dos restarts secuencial: Run 1 instala condacolab, Run 2 ejecuta `mamba install python=3.10`
- **EL FLUJO 2 RESTARTS NO FUNCIONA EN COLAB** por pin de `google-colab`. **A DEPRECAR / refactorizar** para Vast.ai (D9: mantener formato `.ipynb`, eliminar bootstrap Colab, conectar a kernel `tracka`).

**Notebook Track B — estado actual:**
- No fue intervenido en sesiones previas. Asumir mismo nivel de bootstrap defensivo que Track A.
- Validar antes de adaptar.

---

## 5. Próximos pasos priorizados (4 tareas operativas)

> **Tarea #1 del handoff original (crear repo HF Hub) ya completada.** GitHub repo también pusheado.

### #2' — Adaptar `notebooks/train_track_a_ssd.ipynb` para Vast.ai (Track A)

**Mantener formato `.ipynb`** (D9). NO convertir a `.py` con jupytext.

Adaptaciones requeridas:
- **Eliminar** bootstrap defensivo Colab (`condacolab`, `mamba install python=3.10`, `sys.exit(0)`, `IPython.get_ipython().kernel.do_shutdown`).
- **Eliminar** detección `IS_COLAB` (siempre `False` en Vast.ai).
- **Header del notebook**: añadir celda markdown con instrucciones de uso (kernel `tracka`, ejecución vía `tmux + nbconvert`, watchdog).
- **Pin SHA** en `git clone tensorflow/models`: `git checkout 9cafa3d150`.
- **Convertidor TFLite con flags D12**:
  ```python
  converter.experimental_new_quantizer = False
  converter.experimental_new_converter  = False
  ```
- **Inyectar `CommitScheduler`** para checkpoints cada 5 min al repo HF.
- **Inyectar `HFSummaryWriter`** para TensorBoard logs.
- **Variables de entorno** para configuración: `DATASET_DIR`, `OUTPUT_DIR`, `HF_REPO_ID`, `HF_TOKEN`, `VAST_CONTAINERLABEL`, `VAST_API_KEY`.
- **Validación in-notebook obligatoria (D12)** post-export:
  - Inspección `op_version` con paquete `tflite==2.5.0`.
  - Carga test con `tflite_runtime==2.5.0.post1` (wheel Coral CP38).
  - Si custom op falta, plan B D15 (wheel Coral CP36 para deploy).
- **Última celda obligatoria** (D11):
  ```python
  from pathlib import Path
  import subprocess, os
  # Upload artefactos finales a HF Hub
  # ... HfApi.upload_folder(folder_path="track_a/exports", path_in_repo="track_a/exports", repo_id="mitgar14/embebidos-3-models") ...
  Path("/workspace/embebidos-3/.training_done").touch()
  container_id = os.environ.get("VAST_CONTAINERLABEL", "").lstrip("C.")
  if container_id:
      subprocess.run(["vastai", "destroy", "instance", container_id], check=False)
  ```

### #3' — Adaptar `notebooks/train_track_b_yolov8.ipynb` (Track B)

Análogo a #2'. Adaptaciones específicas:
- **Header del notebook**: kernel `trackb`, ejecución vía `tmux + nbconvert`.
- **Pin `numpy<2.0` ANTES** de `pip install ultralytics==8.4.46` (PR #24028 fix).
- **`onnxslim` (NO `onnxsim`)** para optimización del `.onnx`.
- **Export ONNX** con flags D13:
  ```python
  model.export(format="onnx", imgsz=416, opset=11,
               simplify=True, dynamic=False, nms=False)
  ```
- **W&B nativo:** `model.train(..., project="embebidos-3", name="track_b_yolov8n", wandb=True)`.
- **Cascada Roboflow** conservada (descargar dataset desde Roboflow universe vía SDK).
- **Validación in-notebook obligatoria (D13)** post-export:
  - Inspección ops contra blacklist TRT 8.2 (`GridSample`, `ConstantOfShape` con tipo no-FP32).
  - `polygraphy run` vía Docker NGC `nvcr.io/nvidia/tensorrt:21.11-py3` (si Docker-in-Docker está disponible en el container Vast.ai; alternativa: corrida en máquina x86 separada).
- **NO entrenar INT8** (D14: FP16-only por default).
- **Push final** `best.pt` + `best.onnx` + metadata a `track_b/exports/`.
- **Última celda obligatoria** análoga a #2' con watchdog signal.

### #4' — Implementar `scripts/bootstrap.sh` (entrypoint Vast.ai)

Plantilla completa documentada en `2026-05-12-notebook-persistente-uv-jetson.md` §E.1. Reglas clave:
- Variables requeridas: `HF_TOKEN`, `WANDB_API_KEY`, `VAST_API_KEY`, `ROBOFLOW_API_KEY`.
- Crear dos `uv venv` separados (`/opt/venv/tracka` + `/opt/venv/trackb`).
- Registrar ambos como kernels ipykernel (`tracka`, `trackb`).
- Instalar `vastai` CLI + registrar cron watchdog.
- Configurar `/root/.jupyter/jupyter_server_config.py` con:
  ```python
  c.MappingKernelManager.cull_idle_timeout = 0
  c.MappingKernelManager.cull_busy = False
  c.MappingKernelManager.cull_connected = False
  c.ServerApp.shutdown_no_activity_timeout = 0
  c.ZMQChannelsWebsocketConnection.iopub_msg_rate_limit = 0
  c.ZMQChannelsWebsocketConnection.iopub_data_rate_limit = 10_000_000
  ```
- Reiniciar JupyterLab al final (`supervisorctl restart jupyter`).
- Logs a `/workspace/embebidos-3/logs/bootstrap.log`.

### #5' — Implementar `scripts/validate_artifacts.py` (validación pre-deploy)

CLI con flags `--track {A,B}` y `--model <path>`. Tres gates:

**Gate Track A (TFLite):**
```python
# 1. Inspección op_version contra runtime 2.5 max
# 2. Carga test con tflite_runtime==2.5.0.post1 wheel Coral CP38
# 3. Reportar ops con version > soportada (CONV_2D v5, DEPTHWISE_CONV_2D v4-5, etc.)
# 4. Si TFLite_Detection_PostProcess falta, marcar D15 fallback necesario
```

**Gate Track B (ONNX):**
```python
# 1. Inspección ops contra TRT82_UNSUPPORTED blacklist
# 2. Verificar ConstantOfShape no usa tipos no-FP32
# 3. polygraphy run via Docker NGC 21.11-py3 (--trt --onnxrt --atol 1e-2)
# 4. Reportar engines build success + tiempo estimado
```

Salida estructurada (JSON + markdown). Exit code != 0 si gates fallan.

---

## 6. Credenciales y recursos externos

### Confirmados / configurados

| Recurso | Estado | Notas |
|---------|--------|-------|
| **Vast.ai** | 1,72 USD saldo | Suficiente para 3-5 h RTX 4090 |
| **HF Hub** | ✅ Cuenta `mitgar14` autenticada + repo creado | Token en `~/.cache/huggingface/token` |
| **GitHub** | ✅ Cuenta `mitgar14`, repo `embebidos-3` privado pusheado | `gh auth setup-git` aplicado |
| **Roboflow** | Dataset Track B en Roboflow universe | Verificar API key en `.env` |
| **W&B** | Asumir cuenta existente | Verificar `WANDB_API_KEY` antes del primer run |

### Por configurar antes del primer run

- [x] Push del repo `embebidos-3` a GitHub
- [x] Crear repo `mitgar14/embebidos-3-models` en HF Hub
- [ ] Configurar `vastai` CLI local con API key (`vastai set api-key <KEY>`)
- [ ] Definir `.env` con: `HF_TOKEN`, `WANDB_API_KEY`, `ROBOFLOW_API_KEY`, `VAST_API_KEY`
- [ ] Generar deploy key o personal access token con scope `repo` para `git clone` desde el container Vast.ai

---

## 7. Memorias `mnemon` clave

El nuevo agente debe ejecutar al inicio:

```bash
mnemon recall "embebidos-3 Ronda 5 decisiones D9-D15" --limit 5
mnemon recall "embebidos-3 Vast.ai uv JupyterLab notebook persistente" --limit 5
mnemon recall "Jetson Nano JetPack 4.6.1 stack frameworks" --limit 5
mnemon recall "Logitech C920 OG pipeline GStreamer" --limit 3
```

Memorias críticas confirmadas (IDs):

- **`61bcbacb-833d-4b48-b80e-9e7982254f40`** (importance 5) — **R5 decisiones D9–D15 vinculantes** (reemplazó la `6695ef85` previa de R4)
- **`6ada3f23-c7d0-46ce-9d84-ba7fdda3db34`** (importance 4) — Preferencia uv sobre virtualenv
- **`087d0816-ab17-4980-b8f7-727e4b97ca00`** (importance 4) — Fix `gh auth setup-git` vs Git Credential Manager en Windows
- **`9dbed942-...`** (importance 4) — Cámara C920 OG seleccionada (no C930e)
- **`355dd78a-...`** (importance 4) — JetPack 4.6.x usa `focus_auto` (legacy), no `focus_automatic_continuous`
- **`050d8504-...`** (importance 5) — Perfil del usuario (UAO, semestre 7, IA en Embebidos)
- **`1d6d237e-...`** (importance 4) — Alternativas Vast.ai si falla: RunPod, Lambda Labs, Paperspace

---

## 8. Gotchas y errores conocidos

### Colab (a evitar — solo informativo)
1. `condacolab.check()` → `AssertionError`, NO `False`. Usar `install_from_url()` directo.
2. Miniforge 23.11.0-0 → Python **3.12**, no 3.10. Release notes engañan.
3. `mamba install python=3.10` falla en Colab por pin `google-colab` → 3.12. **No es fixeable.** Migración a Vast.ai cerrada.
4. `do_shutdown(True)` es **asíncrono** → necesita `sys.exit(0)` después.
5. Heredoc bash con `\\n` rompe f-strings Python triple-quoted. Usar archivo `.py` separado.
6. `list(string)` produce chars individuales. Usar `splitlines(keepends=True)`.

### Vast.ai
7. **Container storage NO portable** entre instancias. Todo lo persistente debe ir a HF Hub.
8. **PyTorch 2.1+cu121** debe instalarse con `--index-url https://download.pytorch.org/whl/cu121`.
9. **`numpy<2.0` ANTES** de `ultralytics==8.4.46` o el solver lo arrastra a numpy 2.x.
10. **Pin SHA TF Models** `9cafa3d150` — versiones posteriores rompen con TF 2.15.
11. **Vast.ai CLI NO tiene `--auto-stop` ni `--idle-timeout`** (GAP confirmado R5). Imprescindible cron watchdog + última celda con `vastai destroy` (D11).
12. **GPU idle shutdown NO existe** en Vast.ai (GAP confirmado R5).
13. **`$VAST_CONTAINERLABEL`** viene como `C.<id>`. Extraer `${VAST_CONTAINERLABEL#C.}`.
14. **`pip install polygraphy tensorrt`** en container CUDA 12.4 instala TRT 10+, **no 8.2**. Obligatorio Docker NGC `21.11-py3` (D13).

### JupyterLab persistente (R5)
15. **Kernel sobrevive disconnect** solo si `cull_idle_timeout=0`, `cull_busy=False`, `cull_connected=False`, `shutdown_no_activity_timeout=0`.
16. **Output NO se recupera al reconectar** — solo el guardado en `.ipynb` (autosave 120 s). Patrón obligatorio: `tmux + nbconvert --execute --inplace`.
17. **`iopub_msg_rate_limit` y `iopub_data_rate_limit` default truncan output** de runs verbosos. Subir a 0 / 10 MB.
18. **Notebook > 25 MB** rompe autosave. Usar `%%capture` en instalaciones, `tqdm` resumido, log a archivo.

### Jetson Nano (deploy final)
19. TRT engines son **GPU-specific + TRT-version-specific**. Siempre compilar en el Nano.
20. `TFLite_Detection_PostProcess` es **custom op**. Si el wheel NVIDIA no lo incluye, fallback wheel Coral CP36 aarch64 (D15).
21. `Cast v2` requiere TF 2.7+ (Nano tiene 2.5). `BatchMatMul v5+` requiere TF 2.6+. **CONV_2D v5** confirmado problemático (issues #41943, #50652, #43232). Usar `experimental_new_quantizer=False`.
22. Maxwell `sm_53` **sin `dp4a`** → INT8 estructuralmente sin speedup. Track B FP16-only por default (D14).
23. JetPack 4.6.x usa `focus_auto` v4l2 control (legacy), no `focus_automatic_continuous`.

### Generales
24. Tildes obligatorias en español (regla CLAUDE.md global, bug #34779 Claude Code).
25. `mnemon` solo en sub-agent (NUNCA en conversación principal) — esta regla se aplicó consistentemente en R4-R5.
26. Comentarios en español con technical code-switching.
27. **Git Credential Manager en Windows** puede tener credenciales viejas. Si `git push` falla con "repository is disabled" pero `gh repo view` muestra el repo OK, aplicar `gh auth setup-git`.

---

## 9. Gaps de evidencia (estado actualizado)

### Cerrados o mitigados en R5

1. ~~Drop INT8 YOLOv8n en Maxwell `sm_53` — toda evidencia es `sm_75+`.~~ → **Cerrado** por mecanismo físico: `dp4a` no existe en sm_53. Speedup INT8 estructuralmente nulo. Fallback FP16-only (D14).
2. ~~Compatibilidad TFLite forward de `tf-models-official 2.15` con runtime TF 2.5.~~ → **Mitigado** con D12 (`experimental_new_quantizer=False`) + validación in-notebook obligatoria.
3. ~~Polygraphy + TRT 8.2.1 en versiones recientes.~~ → **Cerrado**: polygraphy 0.49.27 funciona con TRT 8 vía Docker NGC `tensorrt:21.11-py3`.

### Abiertos (mitigaciones documentadas)

4. **`op_version` de `DEPTHWISE_CONV_2D` y `FULLY_CONNECTED` en TF 2.15 INT8 PTQ** sin docs públicas. Mitigación: inspección flatbuffer obligatoria (D12 + script `validate_artifacts.py`).
5. **Wheel NVIDIA `tensorflow==2.5.0+nv21.8` y `TFLite_Detection_PostProcess`** sin confirmación verbatim. Mitigación: D15 fallback wheel Coral.
6. **Custom op resolver runtime en Python 3.6.9 del Nano** — verificar `tflite_runtime 2.5` esté disponible vía pip wheel Coral (verificado disponible en R5).
7. **Drop empírico de PTQ TFLite SSD MV2 plain en dataset waste 3-clase** — sin medir hasta ejecutar Track A. No mitigable sin training.

---

## 10. Cómo retomar (instrucciones para el nuevo agente)

1. **Lee este documento de arriba a abajo.** No tomes atajos.
2. Ejecuta los `mnemon recall` de §7 para cargar contexto.
3. **Verifica el estado del repo:**
   ```bash
   cd C:\Users\mitgar14\Documentos\embebidos-3
   git status
   git log --oneline -3
   git remote -v
   ```
4. **Decisión clave del próximo paso:**
   - Consolidación §11 **YA ejecutada**. Carpeta `investigaciones/2026-05-12/` reducida a 6 archivos (HANDOFF + 5 consolidados).
   - Arrancar por **#2' + #3'** en paralelo (recomendado), o solo #4' (`bootstrap.sh`) si quieres ver el flujo completo antes de tocar notebooks.
5. **No iniciar implementación sin confirmación explícita del usuario.**
6. **Si surge ambigüedad técnica**, consulta primero `decisiones-D1-D15-ledger.md` (índice maestro) y desde ahí salta al doc consolidado correspondiente. Si persiste, preguntar al usuario.

### Lectura prioritaria (en orden)

| Doc | Por qué |
|-----|---------|
| **ESTE handoff** | Plan operativo, decisiones D1-D15 consolidadas, gotchas |
| `decisiones-D1-D15-ledger.md` | Índice maestro D1-D15 verbatim + razones + trail entre rondas + cross-refs |
| `compatibilidad-stack-cloud-jetson.md` | Hardware Nano + por qué Vast.ai + stack Track A/B + pin SHA `9cafa3d150` |
| `infraestructura-training-vastai-uv-hf.md` | D9 (`.ipynb` + tmux+nbconvert) + D10 (uv dual venv) + D5 (CommitScheduler) + D6 (W&B) + D11 (cron+vastai destroy) + `bootstrap.sh` completo |
| `validacion-artefactos-pre-deploy.md` | 4 gates (op_version TFLite + ONNX ops + Polygraphy NGC) + plan B Coral CP36 + protocolo experimento INT8 opcional |
| `dataset-roboflow-yolov8.md` | Dataset Roboflow Version 1-B 416×416 + bug `location` workaround + hyperparameters YOLOv8n + export ONNX flags |
| `notebooks/train_track_a_ssd.ipynb` | Estado actual cell-8 v4 (a refactorizar, NO deprecar el .ipynb en sí) |
| `notebooks/train_track_b_yolov8.ipynb` | Estado actual Track B |

### Lo que NO debes hacer

- ❌ Reintentar Colab (decisión cerrada).
- ❌ Cambiar las decisiones D1-D15 sin nueva ronda `/investiga`.
- ❌ Implementar sin confirmación explícita del usuario.
- ❌ Transferir TRT engines entre máquinas — siempre compilar en Nano.
- ❌ Hardcodear paths Colab en los notebooks.
- ❌ Convertir `.ipynb` a `.py` con jupytext (D9 reemplazó D4).
- ❌ Usar `trap EXIT` puro para auto-destroy (D11 reemplazó D7).
- ❌ Instalar `polygraphy tensorrt` directo en container CUDA 12.4 (instala TRT 10, no 8.2). Usar Docker NGC `21.11-py3`.

---

## 11. Consolidación de `investigaciones/2026-05-12/` (EJECUTADA)

> **Estado:** ✅ Ejecutada 2026-05-12. Reducción 20 → 6 archivos (handoff + 5 consolidados autocontenidos). Pendiente solo el commit final en working tree (decisión del usuario: agrupar todo en un commit único al cierre).

### 11.1 Resultado

| Doc nuevo | Líneas | Foco |
|-----------|--------|------|
| `decisiones-D1-D15-ledger.md` | 602 | Índice maestro D1–D15 verbatim + razones + trail R4↔R5 + cross-refs |
| `compatibilidad-stack-cloud-jetson.md` | 636 | Hardware Nano B01 + JetPack 4.6.1 + por qué NO Colab/Kaggle + por qué SÍ Vast.ai + stack Track A/B + pin SHA `9cafa3d150` + tabla `op_version` |
| `infraestructura-training-vastai-uv-hf.md` | 999 | D9 (`.ipynb`+tmux+nbconvert) + D10 (uv dual venv `/opt/venv/{tracka,trackb}` + kernels ipykernel) + D5 (CommitScheduler + upload_folder) + D6 (W&B + TensorBoard hosted) + D11 (cron+vastai destroy) + `bootstrap.sh` completo |
| `validacion-artefactos-pre-deploy.md` | 846 | Gate 1 (op_version TFLite 2.5 max) + Gate 2 (carga test wheel Coral CP38) + Gate 3 (ONNX ops blacklist TRT 8.2) + Gate 4 (Polygraphy NGC `tensorrt:21.11-py3`) + drop INT8 Maxwell sm_53 + plan B Coral CP36 aarch64 (sha256) + protocolo experimento INT8 opcional |
| `dataset-roboflow-yolov8.md` | 679 | Dataset Roboflow `embebidos3/waste-3class-lwld8` Version 1-B 416×416 yolov8 + bug `location` workaround cascada (~50 líneas Python) + Ultralytics 8.4.46 + `numpy<2.0` + `onnxslim` + hyperparameters YOLOv8n + export ONNX flags + 14 gotchas Track B |

**Total consolidado:** 3 762 líneas absorbidas en 5 docs autocontenidos.

### 11.2 Mapeo 19 → 5 (auditoría)

| Doc viejo eliminado | → Absorbido en |
|--------------------|----------------|
| `2026-05-12-compatibilidad-notebooks-training.md` (R4, ~1300 líneas) | `compatibilidad-stack-cloud-jetson.md` + `decisiones-D1-D15-ledger.md` |
| `2026-05-12-notebook-persistente-uv-jetson.md` (R5, ~750 líneas) | `infraestructura-training-vastai-uv-hf.md` + `validacion-artefactos-pre-deploy.md` + `decisiones-D1-D15-ledger.md` |
| `discover-1-jetpack-tflite.md` | `compatibilidad-stack-cloud-jetson.md` §JetPack |
| `discover-2-colab-tfod.md` | `compatibilidad-stack-cloud-jetson.md` §por qué NO Colab |
| `discover-3-mediapipe-mm.md` | `decisiones-D1-D15-ledger.md` (D1 trail) |
| `discover-4-yolov8-trt.md` | `compatibilidad-stack-cloud-jetson.md` §Track B + `validacion-artefactos-pre-deploy.md` §Gate 3 |
| `discover-5-roboflow-bug.md` | `dataset-roboflow-yolov8.md` §3 bug `location` |
| `discover-6-tfod-api-pillow-protobuf.md` | `compatibilidad-stack-cloud-jetson.md` §Track A pins |
| `discover-7-model-main-tf2-output.md` | `compatibilidad-stack-cloud-jetson.md` §pin SHA `9cafa3d150` |
| `discover-8-condacolab-subprocess.md` | `decisiones-D1-D15-ledger.md` (D1 descartado por D9) |
| `discover-9-ultralytics-onnx.md` | `dataset-roboflow-yolov8.md` §4-5 + `validacion-artefactos-pre-deploy.md` §Gate 3 |
| `discover-10-kaggle-stack.md` | `compatibilidad-stack-cloud-jetson.md` §por qué NO Kaggle |
| `discover-11-onnx-trt-nano.md` | `validacion-artefactos-pre-deploy.md` §Gate 3-4 |
| `discover-12-roboflow-sdk.md` | `dataset-roboflow-yolov8.md` §3 workaround |
| `discover-13-vastai-dual-stack.md` | `infraestructura-training-vastai-uv-hf.md` §D10 |
| `discover-14-hf-hub-training.md` | `infraestructura-training-vastai-uv-hf.md` §D5 |
| `discover-15-headless-notebooks.md` | `infraestructura-training-vastai-uv-hf.md` §D9 |
| `discover-16-tflite-fwdcompat.md` | `validacion-artefactos-pre-deploy.md` §Gate 1 |
| `discover-17-onnx-trt-validation.md` | `validacion-artefactos-pre-deploy.md` §Gate 3-4 |

### 11.3 Garantías de no-pérdida verificadas

- ✅ Cada decisión D1-D15 aparece verbatim en `decisiones-D1-D15-ledger.md` con razón + fuente primaria + estado + Ronda de origen.
- ✅ Comandos copy-paste (uv venv, ipykernel install, tmux+nbconvert, polygraphy, vastai destroy) preservados en `infraestructura-training-vastai-uv-hf.md`.
- ✅ URLs y sha256 (wheel Coral CP38 x86, wheel Coral CP36 aarch64 sha256 `7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df`) preservados en `validacion-artefactos-pre-deploy.md`.
- ✅ Versiones pinned (TF 2.15, Pillow 10.4, protobuf 3.20.3, grpcio-tools 1.64.1, ultralytics 8.4.46, numpy<2.0, torch 2.1+cu121) preservados en `compatibilidad-stack-cloud-jetson.md`.
- ✅ Cada gotcha de §8 de este handoff tiene su correlato extendido en el doc consolidado correspondiente.
- ✅ Cada gap residual de §9 aparece con su mitigación en el doc consolidado correspondiente.

### 11.4 Cómo auditar si surge duda

```powershell
# Reconstruir docs viejos desde el commit anterior al rm:
git log --oneline -- investigaciones/2026-05-12/
git show <SHA>:investigaciones/2026-05-12/<doc-viejo>.md
```

El commit que stage la eliminación (working tree actual) deja el snapshot pre-rm como parent → siempre recuperable vía `git show`.

---

**Fin del handoff.** Si algo no es claro, vuelve a leer §3 (decisiones)
y §5 (próximos pasos). Las Rondas 4 y 5 están cerradas y la
consolidación §11 ejecutada — el siguiente paso es ejecutar
**#2'-#5'** del plan operativo.
