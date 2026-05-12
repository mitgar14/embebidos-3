# Handoff — Implementación Vast.ai + HF Hub para embebidos-3

> **Fecha de cierre de sesión:** 2026-05-12
> **Sesión origen:** Claude Code (Opus 4.7), conversación con context compaction
> **Estado:** Ronda 4 /investiga cerrada y documentada. **No se ha iniciado implementación.**
> **Próxima sesión:** Implementar plan operativo (5 tareas priorizadas, ver §6).

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
| **A** | SSD MobileNet v2 FPNLite | 320×320 | `.tflite` INT8 | CPU + XNNPACK + NEON |
| **B** | YOLOv8n | 416×416 | `.onnx` opset 11 → `.engine` FP16 | GPU Maxwell `sm_53` |

### Constraints hardware inmutables (Jetson Nano B01)

- Maxwell **128 CUDA cores, SIN Tensor Cores INT8**
- Python 3.6.9 (system)
- TensorFlow 2.5+nv21.8 (NVIDIA build oficial)
- TensorRT 8.2.1
- CUDA 10.2 + cuDNN 8.2.1
- L4T R32.7.x, kernel Linux 4.9.337

**Implicación crítica:** sin Tensor Cores INT8, el drop empírico de
PTQ en YOLOv8n no está documentado en literatura (toda evidencia es
`sm_75+`). Karimov 2025 reporta Static INT8 −7,2 pp en GPUs modernas;
en Maxwell puede ser peor. **No mitigable sin medición empírica.**

---

## 2. Por qué migramos a Vast.ai

Track A (Colab) se intentó con condacolab pero falló por dos razones
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

Vast.ai tiene saldo confirmado: **1,72 USD** disponibles.

---

## 3. Decisiones vinculantes Ronda 4 (8 ítems)

Documentadas en
`investigaciones/2026-05-12/2026-05-12-compatibilidad-notebooks-training.md`
sección Ronda 4. **No cambiar sin nueva ronda /investiga.**

| # | Decisión | Justificación |
|---|----------|---------------|
| 1 | **Container Vast.ai:** `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` con dual virtualenv (`/opt/venv/tracka` + `/opt/venv/trackb`) | Python 3.10 nativo, CUDA 12.4 compatible con PyTorch 2.1+cu121 y TF 2.15 |
| 2 | **GPU:** RTX 4090 on-demand (0,35 – 0,50 USD/h) | Mejor relación calidad/precio para training corto (1-2 h Track A, 30-60 min Track B) |
| 3 | **Repo HF Hub:** `mitgar14/embebidos-3-models` privado, subcarpetas `track_a/` y `track_b/` | Persistencia de checkpoints, logs y exports. HF free tier: 100 GB privado, max 500 GB/archivo, dedup Xet |
| 4 | **Workflow notebooks:** `jupytext --to py:percent` para convertir `.ipynb` → `.py` headless. Eliminar bootstrap defensivo Colab | papermill no soporta `.py` directamente, jupytext sí. Mantener percent format permite re-conversión bidireccional |
| 5 | **Patrones HF Hub:** `HfApi.upload_folder(run_as_future=True)` + `CommitScheduler(every=5)` para checkpoints + `upload_large_folder` resumible para tar/zip pesados | Async + idempotente + reanudable |
| 6 | **Logging:** Track B usa W&B nativo (`yolo train wandb=True`). Track A usa TensorBoard hosted en HF Hub (`HFSummaryWriter`) | W&B integrado en Ultralytics. TensorBoard se renderiza automáticamente en HF Hub si hay `.tfevents.*` |
| 7 | **Auto-destroy:** `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` con `trap EXIT` | Evitar billing por instancia idle tras finalizar training |
| 8 | **Engine TRT siempre en Nano**, nunca transferido | TRT engines no son portables. `--versionCompatible` solo TRT 8.6+, `--hardwareCompatibilityLevel=ampere+` excluye Maxwell `sm_53` |

---

## 4. Archivos relevantes en el repo

```
C:\Users\mitgar14\Documentos\embebidos-3\
├── notebooks\
│   ├── train_track_a_ssd.ipynb       # Notebook Colab, bootstrap defensivo v4 (a deprecar)
│   └── train_track_b_yolov8.ipynb    # Notebook Colab Track B
├── investigaciones\
│   ├── 2026-05-05\                   # Rondas 1-2 (arquitectura, datasets, dual-track, preprocessing)
│   ├── 2026-05-10\                   # Cámara USB Jetson Nano
│   └── 2026-05-12\
│       ├── 2026-05-12-compatibilidad-notebooks-training.md  # ~1300 líneas, Rondas 1-4
│       ├── HANDOFF-implementacion-vastai-hf.md              # ESTE DOCUMENTO
│       └── discover-{1..17}-*.md     # 17 discoveries Track B
├── docs\                              # Vacío o auxiliares
├── scripts\                           # Por crear (paso #4 de la implementación)
├── prototipos\                        # Por crear (después de pilot runs)
├── pyproject.toml                     # uv-managed
├── uv.lock
└── .gitignore
```

**Notebook Track A — estado actual:**
- `cell-8 v4` con bootstrap defensivo idempotente (`if "banner" not in dir(): ...`)
- Flujo dos restarts secuencial: Run 1 instala condacolab, Run 2 ejecuta `mamba install python=3.10`
- **EL FLUJO 2 RESTARTS NO FUNCIONA EN COLAB** por pin de `google-colab`. **A DEPRECAR al convertir a `.py` headless.**

**Notebook Track B — estado actual:**
- No fue intervenido en esta sesión. Asumir mismo nivel de bootstrap defensivo que Track A.
- Validar antes de convertir a `.py`.

---

## 5. Próximos pasos priorizados (5 tareas)

### #1 — Crear `mitgar14/embebidos-3-models` en HF Hub (privado)

**Pre-requisito de #2, #3, #4 y #5.** Sin esto, los scripts no tienen
destino para `CommitScheduler` ni `upload_folder`.

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    private=True,
    exist_ok=True,
)
# Estructura inicial vía commit dummy:
# track_a/runs/, track_a/checkpoints/, track_a/exports/, track_a/logs/
# track_b/runs/, track_b/checkpoints/, track_b/exports/, track_b/logs/
```

Verificar token HF en `$env:HF_TOKEN` (Windows) o `~/.cache/huggingface/token`.

### #2 — Convertir `train_track_a_ssd.ipynb` → `scripts/train_track_a.py`

```bash
uv run jupytext --to py:percent notebooks/train_track_a_ssd.ipynb \
  -o scripts/train_track_a.py
```

Adaptaciones obligatorias en `scripts/train_track_a.py`:
- **Eliminar** bootstrap defensivo Colab (`condacolab`, `mamba install python=3.10`, `sys.exit(0)`, `IPython.get_ipython().kernel.do_shutdown`).
- **Eliminar** detección `IS_COLAB` (siempre `False` en Vast.ai).
- **Pin SHA** en `git clone tensorflow/models`: `git checkout 9cafa3d150`.
- **Instalación TF Object Detection API** con `--no-deps` + force-reinstall de `Pillow==10.4.0`, `protobuf==3.20.3`, `grpcio-tools==1.64.1`.
- **Inyectar `CommitScheduler`** para checkpoints cada 5 min al repo HF.
- **Inyectar `HFSummaryWriter`** para TensorBoard logs en lugar de `tf.summary.FileWriter`.
- **Variables de entorno** para configuración (NO hardcodear paths Colab): `DATASET_DIR`, `OUTPUT_DIR`, `HF_REPO_ID`, `HF_TOKEN`.

### #3 — Convertir `train_track_b_yolov8.ipynb` → `scripts/train_track_b.py`

Análogo a #2:
```bash
uv run jupytext --to py:percent notebooks/train_track_b_yolov8.ipynb \
  -o scripts/train_track_b.py
```

Adaptaciones:
- **W&B nativo:** `model.train(..., project="embebidos-3", name="track_b_yolov8n", wandb=True)`.
- **Pin `numpy<2.0` ANTES** de `pip install ultralytics==8.4.46` (PR #24028 fix).
- **`onnxslim` (NO `onnxsim`)** para optimización del `.onnx`.
- **Cascada Roboflow** conservada (descargar dataset desde Roboflow universe vía SDK).
- **Push final** `best.pt` + `best.onnx` + metadata a `track_b/exports/`.

### #4 — Implementar `scripts/run.sh` (entrypoint Vast.ai)

Plantilla base (del agente research-code, ya documentada en Ronda 4):

```bash
#!/usr/bin/env bash
set -euo pipefail

# Variables requeridas (export antes de invocar)
: "${HF_TOKEN:?requerido}"
: "${WANDB_API_KEY:?requerido}"
: "${CONTAINER_ID:?requerido}"
: "${CONTAINER_API_KEY:?requerido}"

REPO_URL="https://github.com/mitgar14/embebidos-3.git"
TRACK="${1:-A}"

# Auto-destroy en cualquier salida (éxito o error)
trap 'vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"' EXIT

git clone --depth=1 "$REPO_URL" /workspace/embebidos-3
cd /workspace/embebidos-3
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [ "$TRACK" = "A" ]; then
  uv venv /opt/venv/tracka --python 3.10
  source /opt/venv/tracka/bin/activate
  uv pip install tensorflow==2.15.0 tf-models-official==2.15.0 \
    Pillow==10.4.0 protobuf==3.20.3 grpcio-tools==1.64.1 \
    huggingface_hub jupytext

  git clone https://github.com/tensorflow/models.git /workspace/tf_models
  cd /workspace/tf_models && git checkout 9cafa3d150
  cd /workspace/embebidos-3

  PYTHONPATH=/workspace/tf_models python scripts/train_track_a.py
elif [ "$TRACK" = "B" ]; then
  uv venv /opt/venv/trackb --python 3.10
  source /opt/venv/trackb/bin/activate
  uv pip install "numpy<2.0"
  uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
  uv pip install ultralytics==8.4.46 huggingface_hub wandb onnxslim roboflow

  python scripts/train_track_b.py
else
  echo "Track desconocido: $TRACK"
  exit 1
fi
```

### #5 — Implementar `scripts/validate_artifacts.py` (validación pre-deploy)

**Antes de empacar para el Nano**, validar que los artefactos son
compatibles con TFLite runtime 2.5 y TensorRT 8.2.1.

```python
# Track A — validar .tflite contra TF 2.5
from tensorflow.lite.tools import flatbuffer_utils
model = flatbuffer_utils.read_model("track_a/exports/model.tflite")
# Gate: Cast op_version <= 1 (v2 requiere TF 2.7+)
# Gate: BatchMatMul op_version <= 4 (v5+ requiere TF 2.6+)
# Verificar custom op TFLite_Detection_PostProcess presente
# Si falla: regenerar con TF 2.5 compat o agregar resolver.AddCustom() en Nano

# Track B — validar .onnx contra TRT 8.2.1
import subprocess
subprocess.run([
    "polygraphy", "run",
    "track_b/exports/best.onnx",
    "--trt", "--onnxrt",
    "--tol", "1e-3",
    "--input-shapes", "images:[1,3,416,416]",
])
# Gate: opset == 11, IR <= 10
# Gate: ninguna op en TRT82_UNSUPPORTED blacklist
# NMS debe estar fuera del grafo (postprocess CPU)
```

`TRT82_UNSUPPORTED` blacklist documentado en
`discover-17-onnx-trt-validation.md` y en sección Ronda 4 del doc principal.

---

## 6. Credenciales y recursos externos

### Confirmados / disponibles

| Recurso | Estado | Notas |
|---------|--------|-------|
| **Vast.ai** | 1,72 USD saldo | Suficiente para 3-5 h RTX 4090 |
| **HF Hub** | Cuenta `mitgar14` autenticada (vía `hf-mcp-server`) | Token vigente |
| **GitHub** | Cuenta `mitgar14`, repo `embebidos-3` privado | Verificar push del código antes de Vast.ai run |
| **Roboflow** | Dataset Track B en Roboflow universe | Verificar API key en `.env` |
| **W&B** | Asumir cuenta existente | Verificar `WANDB_API_KEY` antes del primer run |

### Por configurar antes del primer run

- [ ] Push del repo `embebidos-3` a GitHub (verificar `git remote -v`)
- [ ] Crear repo `mitgar14/embebidos-3-models` en HF Hub (paso #1)
- [ ] Configurar `vastai` CLI local con API key (`vastai set api-key <KEY>`)
- [ ] Definir `.env` con: `HF_TOKEN`, `WANDB_API_KEY`, `ROBOFLOW_API_KEY`, `CONTAINER_API_KEY`

---

## 7. Memorias `mnemon` clave

El nuevo agente debe ejecutar al inicio:

```bash
mnemon recall "embebidos-3 Ronda 4 vastai HF Hub decisiones" --limit 5
mnemon recall "Jetson Nano JetPack 4.6.1 stack frameworks" --limit 5
mnemon recall "Logitech C920 OG pipeline GStreamer" --limit 3
```

Memorias críticas confirmadas (IDs):

- **`3a13d0dc-...`** (importance 5) — Decisiones vinculantes Ronda 4 completas
- **`9dbed942-...`** (importance 4) — Cámara C920 OG seleccionada (no C930e)
- **`355dd78a-...`** (importance 4) — JetPack 4.6.x usa `focus_auto` (legacy), no `focus_automatic_continuous`
- **`050d8504-...`** (importance 5) — Perfil del usuario (UAO, semestre 7, materia IA en Embebidos)
- **`1d6d237e-...`** (importance 4) — Alternativas Vast.ai si falla: RunPod (PyTorch 2.6-2.9 solo), Lambda Labs, Paperspace, cluster UAO `uaodeepia11306`

---

## 8. Gotchas y errores conocidos

### Colab (a evitar — solo informativo)
1. `condacolab.check()` → `AssertionError`, NO `False`. Usar `install_from_url()` directo.
2. Miniforge 23.11.0-0 → Python **3.12**, no 3.10. Release notes engañan.
3. `mamba install python=3.10` falla en Colab por pin `google-colab` → 3.12. **No es fixeable.** Es por esto que migramos a Vast.ai.
4. `do_shutdown(True)` es **asíncrono** → necesita `sys.exit(0)` después.
5. Heredoc bash con `\\n` rompe f-strings Python triple-quoted. Usar archivo `.py` separado.
6. `list(string)` produce chars individuales. Usar `splitlines(keepends=True)`.

### Vast.ai (aplicable a la implementación)
7. **Container storage NO portable** entre instancias. Todo lo persistente debe ir a HF Hub.
8. **PyTorch 2.1+cu121** debe instalarse con `--index-url https://download.pytorch.org/whl/cu121` (no PyPI default).
9. **`numpy<2.0` ANTES** de `ultralytics==8.4.46` o el solver lo arrastra a numpy 2.x.
10. **Pin SHA TF Models** `9cafa3d150` — versiones posteriores rompen con TF 2.15.

### Jetson Nano (deploy final)
11. TRT engines son **GPU-specific + TRT-version-specific**. Siempre compilar en el Nano.
12. `TFLite_Detection_PostProcess` es **custom op**, no built-in. Necesita `resolver.AddCustom(...)` en el runtime de inferencia.
13. `Cast v2` requiere TF 2.7+ (en Nano hay 2.5). `BatchMatMul v5+` requiere TF 2.6+. Si el exporter usa estas versiones, el `.tflite` fallará con `kTfLiteError`.
14. Maxwell `sm_53` **sin Tensor Cores INT8** → drop empírico de PTQ no caracterizado en literatura.
15. JetPack 4.6.x usa `focus_auto` v4l2 control (legacy), no `focus_automatic_continuous`.

### Generales
16. Tildes obligatorias en español (regla CLAUDE.md global, bug #34779 Claude Code).
17. `mnemon` solo en sub-agent (NUNCA en conversación principal) — esta sesión tuvo excepciones porque eran operaciones triviales.
18. Notebooks autocontenidos con heartbeat monitoring (preservar este patrón al convertir a `.py`).
19. Comentarios en español con technical code-switching.

---

## 9. Gaps de evidencia (no resueltos)

Documentados en Ronda 4. **Riesgos reales sin mitigación posible
hasta medir empíricamente:**

1. **Drop INT8 YOLOv8n en Maxwell `sm_53`** — toda evidencia es `sm_75+`.
   Si excede 10 pp, fallback a FP16 only en Track B.
2. **Compatibilidad TFLite forward de `tf-models-official 2.15`** con
   runtime TF 2.5 del Nano — no probado directamente con detector SSD.
3. **Custom op resolver runtime** en Python 3.6.9 del Nano — verificar
   que `tflite_runtime` 2.7 esté disponible vía pip wheel.
4. **Polygraphy + AssemblyAI**-style tools en TRT 8.2.1 — versiones
   recientes asumen TRT 10+.

---

## 10. Cómo retomar (instrucciones para el nuevo agente)

1. **Lee este documento de arriba a abajo.** No tomes atajos.
2. Ejecuta los `mnemon recall` de §7 para cargar contexto de memoria
   persistente.
3. **Verifica el estado del repo:**
   ```bash
   cd C:\Users\mitgar14\Documentos\embebidos-3
   git status
   git remote -v
   ```
4. **Confirma con el usuario por cuál tarea arrancar.** Recomendación:
   **#1 + #2 en paralelo** — el repo HF Hub es trivial (un
   `HfApi.create_repo()`) y la conversión jupytext de Track A es la de
   mayor riesgo de regresión.
5. **No iniciar implementación sin confirmación explícita del usuario.**
   El usuario ha sido específico sobre decisiones (no me importan
   costos, mejor GPU, HF Hub persistencia) pero no ha dicho
   *"implementa"* aún.
6. **Si surge ambigüedad técnica**, consulta primero el doc principal
   (`2026-05-12-compatibilidad-notebooks-training.md`) sección Ronda 4
   antes de preguntar al usuario.

### Lectura prioritaria

| Doc | Por qué |
|-----|---------|
| `investigaciones/2026-05-12/2026-05-12-compatibilidad-notebooks-training.md` (Ronda 4) | Plan operativo completo, gotchas, decisiones técnicas |
| `investigaciones/2026-05-12/discover-13-vastai-dual-stack.md` | Detalles del container Vast.ai |
| `investigaciones/2026-05-12/discover-14-hf-hub-training.md` | Patrones `CommitScheduler`, `upload_large_folder` |
| `investigaciones/2026-05-12/discover-15-headless-notebooks.md` | jupytext + papermill workflow |
| `investigaciones/2026-05-12/discover-16-tflite-fwdcompat.md` | TFLite forward compat con TF 2.5 |
| `investigaciones/2026-05-12/discover-17-onnx-trt-validation.md` | Polygraphy + TRT 8.2.1 blacklist |
| `notebooks/train_track_a_ssd.ipynb` | Estado actual cell-8 v4 (a deprecar) |
| `notebooks/train_track_b_yolov8.ipynb` | Estado actual Track B |

### Lo que NO debes hacer

- ❌ Reintentar Colab (decisión cerrada).
- ❌ Cambiar las 8 decisiones vinculantes sin nueva ronda /investiga.
- ❌ Implementar sin confirmación explícita del usuario.
- ❌ Transferir TRT engines entre máquinas — siempre compilar en Nano.
- ❌ Hardcodear paths Colab en los `.py` headless.
- ❌ Omitir el `trap EXIT` en `run.sh` (riesgo de billing acumulado).

---

**Fin del handoff.** Si algo no es claro, vuelve a leer §3 (decisiones)
y §5 (próximos pasos). La Ronda 4 está cerrada — el siguiente paso
es ejecutar, no investigar más.
