# Dataset Roboflow + Track B YOLOv8n + Ultralytics SDK

**Proyecto:** `embebidos-3` (clasificador de residuos Jetson Nano B01, entrega 2026-05-26).
**Dominio:** especificidades de Track B no cubiertas en los otros docs: dataset Roboflow `embebidos3/waste-3class-lwld8`, Ultralytics SDK 8.4.46, bug `location` del SDK Roboflow y su workaround, hyperparameters de training YOLOv8n, export ONNX (flags pre-D13), integración W&B nativa.
**Documentos hermanos:** [`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md) (D6 W&B; D8 export ONNX) · [`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md) (§7 stack Track B) · [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) · [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) (Gates 3 y 4 sobre el `.onnx` exportado aquí).
**Fecha de cierre:** 2026-05-12.

---

## 1. Resumen ejecutivo Track B

Track B es el clasificador GPU del proyecto: **YOLOv8n 416×416**, entrenado en Vast.ai con Ultralytics 8.4.46, exportado a ONNX opset 11, finalmente desplegado como engine TensorRT FP16 en el Jetson Nano B01 (Maxwell `sm_53`, TRT 8.2.1.8).

Cuatro decisiones cierran el pipeline Track B:

| Capa | Decisión | Razón |
|------|----------|-------|
| **Dataset** | Roboflow `embebidos3/waste-3class-lwld8`, Version 1-B, 416×416, formato `yolov8` | 3 clases visualmente distintas (glass/paper/plastic) curado por el usuario; resolución 416 balance velocidad/accuracy en Maxwell |
| **SDK** | Ultralytics `>=8.4.46,<8.5` + `roboflow>=1.3.6,<1.4` + `numpy<2.0` PIN ANTES de ultralytics + `onnxslim>=0.1.82` (no `onnxsim`) | PR #24028 fix INT8 calib non-square; bug `dataset.location` aún sin fix en 1.3.9; Ultralytics 8.3+ migró internamente a `onnxslim` |
| **Training** | `imgsz=416`, `mosaic=1.0`, `close_mosaic=10`, `mixup=0.15`, `fliplr=0.5`, `wandb=True` | Hyperparameters validados en R1-2 para waste-3class; W&B nativo via flag Ultralytics (D6) |
| **Export ONNX** | `opset=11`, `simplify=True`, `dynamic=False`, `nms=False`, `device='cpu'`, `half=False`, `int8=False` | Opset 11 cabe en TRT 8.2-GA (limit 13); shapes fijas para engine determinístico; NMS en CPU NumPy en Nano por `EfficientNMS_TRT` roto Maxwell (#1538); FP16-only por D14 |

**Bug recurrente del SDK Roboflow** (`dataset.location` devuelve path mal resuelto): aplicar workaround de cascada de búsqueda + `os.chdir(WORK_DIR)` + `DATASET_DIRECTORY` env var (§3.4). Sin esto, el `.ipynb` falla con `FileNotFoundError: Dataset '...data.yaml' not found`.

---

## 2. Dataset Roboflow `embebidos3/waste-3class-lwld8`

### 2.1 Identificación del dataset

| Campo | Valor |
|-------|-------|
| **Workspace** | `embebidos3` |
| **Project slug** | `waste-3class-lwld8` |
| **Project type** | Object Detection |
| **Clases** | 3: `glass`, `paper`, `plastic` |
| **IDs de clase** | 0=glass, 1=paper, 2=plastic (ordenamiento alfabético YOLOv8) |
| **Version usada** | **Version 1-B**, formato `yolov8`, resolución 416×416 |

> **Nota:** el ID exacto del dataset puede no coincidir con `embebidos3/waste-3class-lwld8` si el usuario re-curó/exportó tras Ronda 1. Validar antes del primer training con:
>
> ```bash
> python -c "
> import os; os.environ['ROBOFLOW_API_KEY']='<KEY>'
> from roboflow import Roboflow
> rf = Roboflow()
> print([w for w in rf.workspaces])
> "
> ```

### 2.2 Spec de la Version 1 (preprocessing + augmentations, ya validado)

Configuración aplicada al generar **Version 1-B** desde el dashboard de Roboflow (ya validada en R1-2):

**Preprocessing:**

| Stage | Configuración |
|-------|---------------|
| Auto-orient | Activado (corrige EXIF rotation tags) |
| Resize | Stretch a **416×416** (no `Fit (white edges)` para evitar márgenes blancos en train) |
| Tile | (deshabilitado) |
| Modify classes | (sin remapeo: 3 clases tal cual) |

**Augmentation** (aplicado solo a train split, no val/test):

| Augmentation | Valor / rango |
|--------------|---------------|
| Flip horizontal | ✅ activado |
| Flip vertical | ❌ desactivado (waste-3class no es invariante a vertical flip — etiquetas se invierten) |
| 90° rotation | ❌ desactivado |
| Crop (zoom) | ±15% |
| Rotation | ±15° |
| Shear | ±5° H, ±5° V |
| Brightness | ±25% |
| Exposure | ±15% |
| Blur | hasta 1,5 px |
| Noise | hasta 2% pixels |
| Mosaic (Roboflow) | ❌ desactivado (Ultralytics aplica su propio mosaic en training, mejor calidad) |

**Generate version:** 3 outputs (3× train images con augmentations aplicadas). Split 70/20/10 (train/valid/test) por default Roboflow.

### 2.3 Variables de entorno requeridas

```bash
# .env (gitignored, NO commitear)
ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxxxxxxxx

# Set ANTES de instanciar Roboflow() para forzar dir de descarga
DATASET_DIRECTORY=/workspace/embebidos-3/datasets
```

Sin `DATASET_DIRECTORY` set, el SDK Roboflow descarga al CWD del proceso Python — que en JupyterLab + tmux + `nbconvert` puede no ser el directorio del notebook (es el dir donde se invocó `jupyter nbconvert`).

---

## 3. Roboflow SDK bug `location` y workaround obligatorio

### 3.1 Causa raíz

Source de [`roboflow/core/version.py`](https://github.com/roboflow/roboflow-python/blob/main/roboflow/core/version.py) (verificado en v1.3.9, SHA `1e4cbc04`):

```python
def download(self, model_format=None, location=None, overwrite: bool = False):
    if location is None:
        location = self.__get_download_location()  # ← devuelve path RELATIVO al CWD
```

Cuando `location is None`, `__get_download_location()` retorna algo como `"./waste-3class-1"` (relativo). Combinado con `os.chdir` interno o llamadas posteriores que cambian el CWD, el dataset termina en una ubicación inesperada.

### 3.2 Bug residual cuando `location` SÍ se pasa

Aunque el usuario pase `location` explícito, el `data.yaml` generado por el SDK contiene paths **relativos** que asumen ubicación distinta. Issue [`roboflow/roboflow-python#240`](https://github.com/roboflow/roboflow-python/issues/240) verbatim:

> *"RuntimeError: Dataset 'TennisBallTracker-9/data.yaml' error [...] missing path '/.../datasets/TennisBallTracker-9/TennisBallTracker-9/valid/images'"*

El path resultante tiene **el slug del proyecto duplicado** (`TennisBallTracker-9/TennisBallTracker-9/...`) porque el `data.yaml` no respeta el `location` pasado.

### 3.3 Issue #88 — clases fantasma en `data.yaml`

Issue [`roboflow/roboflow-python#88`](https://github.com/roboflow/roboflow-python/issues/88) (open desde 2022-12-21, **sin fix** a 2026-05-12): si se eliminan clases en Roboflow Universe sin regenerar la version, el `data.yaml` puede contener `names: [glass, paper, plastic, deprecated_class]` con `nc: 4` mal. **Mitigación:** validar `nc:` y `names:` manualmente post-download.

### 3.4 Workaround completo (cascada de búsqueda)

Patrón canónico aplicado en `notebooks/train_track_b_yolov8.ipynb` cell-10:

```python
# === Track B — cell-10 Roboflow download con workaround ===
import os
import shutil
from pathlib import Path
from roboflow import Roboflow

# CRÍTICO: set ANTES de Roboflow()
WORK_DIR = Path("/workspace/embebidos-3/datasets")
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATASET_DIRECTORY"] = str(WORK_DIR)

# Forzar CWD conocido ANTES de version.download()
os.chdir(WORK_DIR)

# Descargar
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("embebidos3").project("waste-3class-lwld8")
version = project.version(1)  # Version 1-B
ds = version.download("yolov8", location=str(WORK_DIR / "waste-3class-1"), overwrite=False)
# ds.location es lo que el SDK CREE haber descargado

# === Cascada de búsqueda para data.yaml ===
# (a) primero en ds.location (lo más confiable si el SDK no rompió)
# (b) luego en WORK_DIR (si el SDK lo descargó al CWD por bug)
# (c) fallback global con heurística por contenido

def find_data_yaml(ds_location: str, work_dir: Path) -> Path:
    """Busca data.yaml con cascada y heurística."""
    candidates = [
        Path(ds_location) / "data.yaml",                        # (a)
        work_dir / "waste-3class-1" / "data.yaml",              # (b1)
        work_dir / "data.yaml",                                  # (b2)
    ]

    # Heurística por contenido (c) — buscar globalmente cualquier data.yaml con las 3 clases
    if not any(c.exists() for c in candidates):
        print("[fallback] Buscando data.yaml globalmente con heurística glass+paper+plastic...")
        for yaml_path in Path("/workspace").rglob("data.yaml"):
            content = yaml_path.read_text(errors="ignore").lower()
            if all(cls in content for cls in ["glass", "paper", "plastic"]):
                print(f"[fallback] Encontrado: {yaml_path}")
                candidates.insert(0, yaml_path)
                break

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"data.yaml no encontrado en cascada:\n" +
        "\n".join(f"  - {c}" for c in candidates)
    )

DATA_YAML = find_data_yaml(ds.location, WORK_DIR)
print(f"✅ DATA_YAML resuelto: {DATA_YAML}")

# === Si está fuera del WORK_DIR esperado, copiar/mover ===
EXPECTED_DIR = WORK_DIR / "waste-3class-1"
if not str(DATA_YAML).startswith(str(EXPECTED_DIR)):
    print(f"[fix] Copiando dataset a {EXPECTED_DIR}...")
    src_root = DATA_YAML.parent
    shutil.copytree(src_root, EXPECTED_DIR, dirs_exist_ok=True)
    DATA_YAML = EXPECTED_DIR / "data.yaml"

# === Validación post-download (defensa contra issue #88) ===
import yaml
with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

assert cfg["nc"] == 3, f"nc esperado 3, got {cfg['nc']} — issue #88 (clases fantasma)"
assert sorted(cfg["names"]) == ["glass", "paper", "plastic"], \
    f"Clases inesperadas: {cfg['names']}"
print(f"✅ data.yaml válido: nc={cfg['nc']}, names={cfg['names']}")

# === Validación de splits (cada uno con >= 10 img por clase) ===
ROOT = DATA_YAML.parent
for split in ["train", "valid", "test"]:
    img_dir = ROOT / split / "images"
    if img_dir.exists():
        n = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        print(f"   {split}: {n} imágenes")
        if split in ["train", "valid"]:
            assert n >= 30, f"{split} tiene solo {n} imágenes (< 30 = < 10 por clase)"
```

### 3.5 Estado del bug en versiones recientes (R3 verificado)

- **Roboflow SDK v1.3.9 (2026-05-07, SHA `1e4cbc04`):** SIN fix. Releases 1.3.7–1.3.9 enfocados en soft-delete y device-management, no en path resolution.
- **Roboflow SDK v1.3.6:** mismo bug, sin fix.
- **Workaround cascada de cell-10 es obligatorio** hasta nueva ronda `/investiga` que confirme fix.

Issues relacionados (todos open):

- [`#125`](https://github.com/roboflow/roboflow-python/issues/125) "data.yaml file has different references for image paths" (2022)
- [`#240`](https://github.com/roboflow/roboflow-python/issues/240) "Incorrect Data Path in YOLOv8 Dataset Configuration" (2024)
- [`#306`](https://github.com/roboflow/notebooks/issues/306) "dataset.location empty" (en `roboflow/notebooks`)
- [`#333`](https://github.com/roboflow/roboflow-python/issues/333) "Issue with relative paths in data.yaml file when trying to train yolo custom model"
- [`#108`](https://github.com/roboflow/roboflow-python/issues/108) ".download() re-downloads the same version even if it already exists on disk"

### 3.6 PR #113 (fix histórico, no aplicable a nuestro caso)

[`roboflow/roboflow-python#113`](https://github.com/roboflow/roboflow-python/pull/113) "Fix for v8>=8.0.29 breaking changed to dataset loader" — fix de 2023 para una breaking change distinta de Ultralytics 8.0.30. Mencionado por completitud, no resuelve nuestro caso.

---

## 4. Ultralytics 8.4.46 + `numpy<2.0` requirement (PR #24028)

### 4.1 Por qué pinear `>=8.4.46,<8.5`

- **PR [`ultralytics/ultralytics#24028`](https://github.com/ultralytics/ultralytics/pull/24028)** "INT8 calibration non-square imgsz fix" merged 2026-03-28 en v8.4.31. Bug previo: si `imgsz` no era cuadrado (e.g., 640×480), la calibración INT8 producía resultados inválidos. Como usamos `imgsz=416` (cuadrado), el bug no nos afecta directamente, pero el pin garantiza que si en el futuro se cambia `imgsz` a no-cuadrado, no regresione.
- **v8.4.48** (2026-05-08) es la última release a 2026-05-12 (verificado con `gh api repos/ultralytics/ultralytics/releases?per_page=10`). El pin `<8.5` captura v8.4.46 → v8.4.48 sin permitir saltos a 8.5 (que aún no existe).
- **No existe 8.5** a fecha 2026-05-12; el cap es defensivo contra breaking changes futuros.

### 4.2 Por qué `numpy<2.0` ANTES de `ultralytics`

`ultralytics/pyproject.toml` declara `numpy<2.0.0` con comentario *"TF 2.20 compatibility"*. Heredado automáticamente al instalar `ultralytics`, pero en Kaggle/Colab base la cadena de deps transitivas puede dejar NumPy 2.x ya instalado **antes** de que ultralytics se evalúe. Resultado: NumPy 2.4.x quedó instalado y `import ultralytics` falla con:

```
ImportError: numpy.core.multiarray failed to import
```

Confirmado en issue [`ultralytics/ultralytics#22346`](https://github.com/ultralytics/ultralytics/issues/22346) "Installation + runtime failure: pip dependency conflicts & NumPy 2.2.6 import errors when running on Kaggle T4x2".

**Defensa robusta:** instalar `numpy<2.0` **explícitamente ANTES** de `ultralytics`:

```bash
uv pip install "numpy<2.0"  # PRIMERO
uv pip install "ultralytics>=8.4.46,<8.5"  # DESPUÉS
```

En Vast.ai con `uv venv` limpio el problema no debería ocurrir (no hay deps preinstaladas), pero el pin explícito es defensa en profundidad.

### 4.3 Validación dura post-install

```python
import numpy
assert int(numpy.__version__.split(".")[0]) < 2, \
    f"❌ NumPy {numpy.__version__} viola numpy<2.0. Issue #22346. Reinstalar trackb venv."
print(f"✅ NumPy {numpy.__version__} OK")
```

### 4.4 Stack moderno Ultralytics (referencia)

PRs recientes relevantes (R3 R4):

- **PR [`#23807`](https://github.com/ultralytics/ultralytics/pull/23807)** (2026-03-05): Docker base actualizado a `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`. Confirma torch 2.10 como runtime moderno. **No usamos torch 2.10**: pinneamos a 2.1.0+cu121 por compat con TRT 8.2 en validación.
- **PR [`#23808`](https://github.com/ultralytics/ultralytics/pull/23808)** (2026-03-05): "safer ONNX opset cap for Torch 2.9+ exports". Protección adicional al pin `opset=11` explícito del notebook.

---

## 5. `onnxslim` (no `onnxsim`) — Ultralytics 8.3+

### 5.1 Migración interna

Verificado leyendo source de [`ultralytics/engine/exporter.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py) y [`pyproject.toml`](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) rama `main` 2026-05-12:

Desde Ultralytics **8.3+**, `model.export(simplify=True)` llama internamente a `onnxslim.slim(model_onnx)`, **NO** a `onnxsim`. El pin `onnxsim` en `pyproject.toml` del notebook (heredado de templates antiguos) es **irrelevante** para el exporter.

### 5.2 Diferencias entre `onnxsim` y `onnxslim`

| Herramienta | Estado | Compatibilidad Py 3.12 | Usado por Ultralytics 8.3+ |
|-------------|--------|------------------------|-----------------------------|
| `onnxsim` (`daquexian/onnx-simplifier`) | Mantenimiento de comunidad | 0.4.x ❌ no compila (issue [`#334`](https://github.com/daquexian/onnx-simplifier/issues/334)); 0.6.x ✅ wheels manylinux | ❌ No (deprecado) |
| **`onnxslim`** (`inisis/OnnxSlim`) | **Mantenimiento activo** | ✅ Sí | ✅ **Sí (default desde 8.3)** |

Ambos simplifican ONNX (constant folding + dead-code elimination + shape inference), pero pueden producir grafos ligeramente distintos. `onnxslim` es más nuevo y aceptado por Ultralytics; `onnxsim` está en mantenimiento.

### 5.3 Pin recomendado

```bash
uv pip install "onnxslim>=0.1.82"
# NO usar onnxsim, mantener solo si código custom lo necesita
```

### 5.4 Breaking changes históricos `onnxsim 0.5.0+` (informativo)

Si por alguna razón el código de usuario llama `onnxsim` directamente (no vía Ultralytics):

- `--dynamic-input-shape` removido (era flag deprecated)
- `--input-shape` → `--overwrite-input-shape`
- `--enable-fuse-bn` removido (default `True` ahora)

---

## 6. Hyperparameters de training YOLOv8n

### 6.1 Configuración canónica

```python
from ultralytics import YOLO
import os

# Asegurar W&B autenticado (D6)
assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY no presente — set en bootstrap"

# Cargar modelo pretrained COCO
model = YOLO("yolov8n.pt")  # descarga automática desde Ultralytics CDN

# Training
results = model.train(
    data="/workspace/embebidos-3/datasets/waste-3class-1/data.yaml",

    # === Geometría ===
    imgsz=416,                  # PR #24028 fix; resolución Track B
    rect=False,                 # rectangular training off (dataset ya stretch 416×416)

    # === Schedule ===
    epochs=100,                 # arrancar con 100; early stopping vía patience
    patience=20,                # detener si val no mejora 20 epochs
    batch=32,                   # 4090 24 GB holgado para YOLOv8n 416
    workers=8,                  # data loader threads

    # === Optimizer ===
    optimizer="auto",           # selecciona SGD/AdamW según task
    lr0=0.01,
    lrf=0.01,                   # final lr = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,

    # === Augmentations (training) ===
    mosaic=1.0,                 # 4-image mosaic 100% epochs (off al final)
    close_mosaic=10,            # desactivar mosaic en últimas 10 epochs
    mixup=0.15,                 # 15% chance mixup
    fliplr=0.5,                 # 50% horizontal flip
    flipud=0.0,                 # NO vertical flip (waste-3class no es invariante)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0,                # rotation off (Roboflow ya aplicó ±15°)
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,

    # === Logging ===
    project="embebidos-3",
    name="track_b_yolov8n",
    save=True,
    save_period=10,             # checkpoint cada 10 epochs
    plots=True,
    wandb=True,                 # D6 — W&B nativo

    # === Reproducibilidad ===
    seed=42,
    deterministic=True,

    # === Device ===
    device=0,                   # GPU id (single GPU)
    amp=True,                   # mixed precision FP16 training (no afecta export)
)
```

### 6.2 Justificación de los augmentations

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `mosaic=1.0` | 4-img mosaic 100% epochs | Augmentation estándar Ultralytics; mejora robustez a contextos diversos |
| `close_mosaic=10` | desactivar en últimas 10 epochs | Mosaic distorsiona objetos; al final mejor train sin él para que el modelo aprenda detección "limpia" |
| `mixup=0.15` | 15% chance mixup | Pequeño boost de generalización en clases visualmente distintas |
| `fliplr=0.5` | 50% flip horizontal | Waste invariante a flip horizontal (vidrio brillante igual de izquierda o derecha) |
| `flipud=0.0` | NO vertical flip | Etiquetas se invierten al voltear vertical (top → bottom) — datos perderían validez espacial |
| `degrees=0.0` | NO rotación | Roboflow ya aplicó ±15° en augmentation. Doble rotación destruiría señal |
| `translate=0.1` | 10% translación | Mejora robustez a objetos no centrados |
| `scale=0.5` | escala ±50% | Mejora robustez a tamaño del objeto en cámara |

### 6.3 W&B nativo (D6)

Activado con `wandb=True`. Crea automáticamente:

- **Run** en proyecto W&B `embebidos-3` con name `track_b_yolov8n`.
- **Métricas:** `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/precision`, `val/recall`, `val/mAP50`, `val/mAP50-95`.
- **Media:** confusion matrix, sample predictions (cada N epochs), labels.jpg, results.png.
- **System metrics:** GPU util, GPU memory, CPU util, network I/O.

URL típica: `https://wandb.ai/<user>/embebidos-3/runs/<run_id>`.

**Variables requeridas:** `WANDB_API_KEY` (inyectada al container vía `--env`, ver [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) §6).

**Free tier W&B 2026:** sin cap conocido para proyectos personales. Confirmado en [`wandb.ai/site/pricing`](https://wandb.ai/site/pricing).

### 6.4 Gates de pre-train

Antes de lanzar `model.train()` validar (en una celda anterior del notebook):

```python
# Pre-train sanity check
import yaml
from pathlib import Path

DATA_YAML = Path("/workspace/embebidos-3/datasets/waste-3class-1/data.yaml")
assert DATA_YAML.exists(), f"data.yaml no encontrado en {DATA_YAML}"

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

# Validar 3 clases
assert cfg["nc"] == 3, f"nc={cfg['nc']} ≠ 3 (issue #88)"
assert sorted(cfg["names"]) == ["glass", "paper", "plastic"]

# Validar splits con ≥ 30 imágenes (≈ 10 por clase mínimo)
for split in ["train", "val"]:
    split_path = Path(cfg.get(split, ""))
    if not split_path.is_absolute():
        split_path = DATA_YAML.parent / split_path
    n_imgs = len(list(split_path.parent.glob("**/*.jpg"))) + len(list(split_path.parent.glob("**/*.png")))
    assert n_imgs >= 30, f"{split} tiene solo {n_imgs} imágenes"
    print(f"✅ {split}: {n_imgs} imágenes")

print("✅ Pre-train sanity OK — proceder con model.train()")
```

### 6.5 Gates de post-train

Antes de exportar a ONNX, validar el `.pt`:

```python
# Post-train sanity
results = model.val()  # corre val split
assert results.box.map50 > 0.5, f"mAP50={results.box.map50:.3f} < 0.5 — modelo no aprendió"

# fuse() no debe errorear
fused = model.fuse()
print(f"✅ mAP50={results.box.map50:.3f}, mAP50-95={results.box.map:.3f}, fuse OK")

# Mejor checkpoint disponible
best_pt = Path("runs/detect/track_b_yolov8n/weights/best.pt")
assert best_pt.exists(), "best.pt no generado"
```

---

## 7. Export ONNX canónico (pre-D13)

### 7.1 Comando

```python
from ultralytics import YOLO
model = YOLO('runs/detect/track_b_yolov8n/weights/best.pt')

# Export ONNX con flags explícitos
output_path = model.export(
    format='onnx',
    opset=11,           # CRÍTICO: default 20 con torch 2.9+; TRT 8.2 max opset 13
    simplify=True,      # llama onnxslim.slim() internamente (Ultralytics 8.3+)
    dynamic=False,      # shapes fijas para TRT engine determinístico
    imgsz=416,          # PR #24028 fix INT8 calib non-square (cuadrado, OK)
    device='cpu',       # CUDA host irrelevante para .onnx (export por trazado en CPU)
    half=False,         # FP16 se aplica en trtexec en Nano, no en ONNX
    int8=False,         # FP16-only por D14 (sin dp4a en sm_53)
    nms=False,          # NMS en CPU NumPy en Nano (EfficientNMS_TRT roto Maxwell #1538)
)
print(f"✅ ONNX exportado: {output_path}")
# Output: runs/detect/track_b_yolov8n/weights/best.onnx
```

### 7.2 Justificación verbatim de cada flag

| Flag | Razón | Issue / PR de referencia |
|------|-------|--------------------------|
| `opset=11` | TRT 8.2-GA soporta hasta opset 13; opset 11 es conservador y evita issue [`NVIDIA/TensorRT#4383`](https://github.com/NVIDIA/TensorRT/issues/4383) (Gather rank-0 bug con opset ≥17) | `onnx-tensorrt operators.md release/8.2-GA`, PR Ultralytics #23808 |
| `simplify=True` | Reduce nodos del grafo (constant folding, dead-code elimination) → engine TRT más compacto y rápido | Default Ultralytics 8.3+ usa `onnxslim` |
| `dynamic=False` | Shapes fijas → `ConstantOfShape` genera initializers FP32 (TRT 8.2 solo soporta FP32 en esta op); engine TRT determinístico | onnx-tensorrt operators.md `ConstantOfShape` restriction |
| `imgsz=416` | Resolución entrenada, debe coincidir en export | PR Ultralytics #24028 (INT8 calib non-square fix; no afecta 416×416 cuadrado pero pin defensivo) |
| `device='cpu'` | PyTorch exporta ONNX por trazado en CPU; el CUDA del host es irrelevante para el `.onnx` resultante | Ultralytics docs export |
| `half=False` | FP16 se aplica al compilar engine TRT (`trtexec --fp16` en Nano), NO en ONNX | `validacion-artefactos-pre-deploy.md` §6 |
| `int8=False` | FP16-only por D14 (Maxwell sin `dp4a` → INT8 sin speedup) | Decisión ledger D14 |
| `nms=False` | `EfficientNMS_TRT` plugin roto en Maxwell con TRT 8.x; NMS hecho en CPU NumPy con `cv2.dnn.NMSBoxes` en el Nano | Issue [`NVIDIA/TensorRT#1538`](https://github.com/NVIDIA/TensorRT/issues/1538) |

### 7.3 Verificación post-export (Gate 3 input)

```python
import onnx
m = onnx.load('runs/detect/track_b_yolov8n/weights/best.onnx')

# Validaciones duras
assert m.opset_import[0].version == 11, \
    f"❌ Opset {m.opset_import[0].version} ≠ 11 — flag `opset=11` no aplicado"
assert m.ir_version <= 10, \
    f"❌ IR version {m.ir_version} > 10 puede romper TRT 8.2 / onnxslim 0.6.x"

ops_present = sorted({n.op_type for n in m.graph.node})
print(f"✅ Opset: {m.opset_import[0].version}, IR: {m.ir_version}")
print(f"✅ Ops únicos en grafo: {ops_present}")

# Sanity check input shape
assert m.graph.input[0].type.tensor_type.shape.dim[2].dim_value == 416
assert m.graph.input[0].type.tensor_type.shape.dim[3].dim_value == 416
print(f"✅ Input shape: [1, 3, 416, 416]")
```

El siguiente paso es validar contra **Gate 3 + Gate 4** del doc hermano [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md):

1. **Gate 3** — inspección ops contra blacklist TRT 8.2 (script `validate_onnx_ops.py`).
2. **Gate 4** — `polygraphy run --trt --onnxrt` en Docker NGC `nvcr.io/nvidia/tensorrt:21.11-py3`.

### 7.4 Upload final a HF Hub

Tras validación exitosa de Gates 3 y 4:

```python
from huggingface_hub import HfApi
import json
from pathlib import Path

EXPORT_DIR = Path("/workspace/embebidos-3/track_b/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Copiar artefactos
import shutil
shutil.copy("runs/detect/track_b_yolov8n/weights/best.pt", EXPORT_DIR / "best.pt")
shutil.copy("runs/detect/track_b_yolov8n/weights/best.onnx", EXPORT_DIR / "best.onnx")

# Manifest
manifest = {
    "model": "YOLOv8n",
    "track": "B",
    "imgsz": 416,
    "classes": ["glass", "paper", "plastic"],
    "opset": 11,
    "ir_version": 10,
    "nms": "external (CPU NumPy)",
    "precision_export": "FP32",
    "precision_deploy": "FP16",  # aplicado en Nano via trtexec --fp16
    "dataset": "embebidos3/waste-3class-lwld8 v1-B",
    "ultralytics_version": "8.4.46",  # leer dinámicamente: ultralytics.__version__
    "training_run": "track_b_yolov8n",
    "metrics": {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    },
    "wandb_run": "<URL del run W&B>",
}
(EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

# Push asíncrono
api = HfApi()
future = api.upload_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path=str(EXPORT_DIR),
    path_in_repo="track_b/exports",
    commit_message=f"Track B — best.onnx (mAP50={results.box.map50:.3f})",
    run_as_future=True,
)
print(f"Upload iniciado: {future}")
result = future.result(timeout=600)
print(f"✅ Upload completado: {result}")
```

---

## 8. Gotchas específicos de Track B

| # | Gotcha | Mitigación |
|---|--------|-----------|
| B1 | Ultralytics 8.3+ migró de `onnxsim` a `onnxslim`. Pin `onnxsim` en `pyproject.toml` del notebook es irrelevante para `model.export(simplify=True)` | Pin `onnxslim>=0.1.82` explícito; remover `onnxsim` si no se usa |
| B2 | `best_onnx_opset` con torch 2.9+: cap a opset 20 (no 22 como decía docstring antiguo). Sin `opset=11` explícito, exporta opset 20 → TRT 8.2 falla | Pin `opset=11` defensivo + assert post-export |
| B3 | NumPy 2.x rompe `ultralytics` en runtime aunque pyproject.toml declare `numpy<2.0` | Pin explícito `numpy<2.0` **antes** de `ultralytics`; validación dura post-install |
| B4 | Roboflow SDK 1.3.9 (mayo 2026) SIN fix bug `dataset.location` | Workaround cascada de búsqueda (§3.4); obligatorio en cell-10 |
| B5 | Roboflow `data.yaml` clases fantasma (#88 abierto desde 2022) | Validar `nc:` y `names:` manualmente post-download |
| B6 | `EfficientNMS_TRT` roto en Maxwell con TRT 8.x (issue #1538) | `nms=False` en export ONNX; NMS en CPU NumPy en Nano |
| B7 | `Reciprocal`, `NonZero`, `RoiAlign` NO soportados en TRT 8.2 (improbables en YOLOv8n FP32 estático, pero validar) | Gate 3 detecta y warnea (ver [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §4.4) |
| B8 | `Gather` rank-0 bug en TRT 8.x con opsets ≥17 (issue [`NVIDIA/TensorRT#4383`](https://github.com/NVIDIA/TensorRT/issues/4383)) | Opset 11 lo evita estructuralmente |
| B9 | Polygraphy NO funcional en JetPack 4.6.1 (Python 3.6.9 incompat con polygraphy 0.45+) | Validación pre-deploy en Vast.ai vía Docker NGC; en Nano usar `trtexec` directo |
| B10 | ONNX Runtime + TRT EP NO viable en Nano JP4.6.1 (ORT 1.11+ requiere CUDA 11.4; Nano tiene CUDA 10.2) | Inferencia en Nano: `tensorrt` Python bindings + `cuda-python` 11.0 |
| B11 | TRT engine OOM al export en Nano (issue [`ultralytics#14751`](https://github.com/ultralytics/ultralytics/issues/14751)) | `trtexec --workspace=1024` en Nano (no exportar engine en Vast.ai por D8) |
| B12 | Ultralytics inserta `nvidia_cuda_*` packages al instalar (#23379) | Si se requiere CPU-only en algún paso, pin manual; no aplica a Vast.ai con GPU |
| B13 | `model.train()` `wandb=True` requiere `WANDB_API_KEY` ya en env; sin él, falla silenciosamente y entrena sin loggear | Validar `os.environ["WANDB_API_KEY"]` antes de `model.train()` |
| B14 | `model.export(format="onnx")` puede fallar si torch tiene caché stale | Limpiar `~/.cache/torch/hub` si export errorea con "model not found" |

---

## 9. Fuentes consultadas

| # | Título | URL | Tipo |
|---|--------|-----|------|
| 1 | Ultralytics PR #24028 (INT8 calib non-square fix) | https://github.com/ultralytics/ultralytics/pull/24028 | PR |
| 2 | Ultralytics PR #23807 (Docker pytorch 2.10) | https://github.com/ultralytics/ultralytics/pull/23807 | PR |
| 3 | Ultralytics PR #23808 (safer ONNX opset cap) | https://github.com/ultralytics/ultralytics/pull/23808 | PR |
| 4 | Ultralytics PR #113 (v8>=8.0.29 fix dataset loader) | https://github.com/roboflow/roboflow-python/pull/113 | PR |
| 5 | Ultralytics issue #22346 (NumPy 2.2.6 Kaggle) | https://github.com/ultralytics/ultralytics/issues/22346 | Issue |
| 6 | Ultralytics issue #22336 (Kaggle dep conflicts) | https://github.com/ultralytics/ultralytics/issues/22336 | Issue |
| 7 | Ultralytics issue #22840 (Kaggle environment conflicts) | https://github.com/ultralytics/ultralytics/issues/22840 | Issue |
| 8 | Ultralytics issue #19498 (ONNX IR Version 8) | https://github.com/ultralytics/ultralytics/issues/19498 | Issue |
| 9 | Ultralytics issue #23436 (PyTorch 2.10 support) | https://github.com/ultralytics/ultralytics/issues/23436 | Issue |
| 10 | Ultralytics issue #2821 (TRT INT64 weights) | https://github.com/ultralytics/ultralytics/issues/2821 | Issue |
| 11 | Ultralytics issue #14751 (TRT engine OOM Nano) | https://github.com/ultralytics/ultralytics/issues/14751 | Issue |
| 12 | Ultralytics issue #23379 (nvidia_cuda_* packages) | https://github.com/ultralytics/ultralytics/issues/23379 | Issue |
| 13 | Ultralytics issue #16839 (YOLOv11 opset issues) | https://github.com/ultralytics/ultralytics/issues/16839 | Issue |
| 14 | Ultralytics issue #10298 (Inference code Jetson Nano) | https://github.com/ultralytics/ultralytics/issues/10298 | Issue |
| 15 | Ultralytics issue #7222 (FP16 TRT export) | https://github.com/ultralytics/ultralytics/issues/7222 | Issue |
| 16 | Ultralytics PR #12652 (get_latest_opset() compat torch<1.13) | https://github.com/ultralytics/ultralytics/pull/12652 | PR |
| 17 | Ultralytics releases timeline v8.4.31 → v8.4.48 | https://github.com/ultralytics/ultralytics/releases | Repo oficial |
| 18 | Ultralytics source `engine/exporter.py` main | https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py | Código fuente |
| 19 | Ultralytics source `pyproject.toml` main | https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml | Código fuente |
| 20 | Ultralytics docs Model Export | https://docs.ultralytics.com/modes/export | Doc oficial |
| 21 | Ultralytics docs ONNX Export | https://docs.ultralytics.com/integrations/onnx/ | Doc oficial |
| 22 | Ultralytics docs Kaggle integration | https://docs.ultralytics.com/integrations/kaggle/ | Doc oficial |
| 23 | Ultralytics blog YOLOv8 TensorRT optimization | https://www.ultralytics.com/blog/optimizing-ultralytics-yolo-models-with-the-tensorrt-integration | Blog oficial |
| 24 | docs/en/guides/deepstream-nvidia-jetson.md | https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/deepstream-nvidia-jetson.md | Doc oficial |
| 25 | Zenodo v8.3.193 release notes | https://zenodo.org/records/17054310 | Release notes |
| 26 | Zenodo v8.3.156 release notes | https://zenodo.org/records/15682027 | Release notes |
| 27 | Roboflow SDK source `version.py` v1.3.9 | https://github.com/roboflow/roboflow-python/blob/main/roboflow/core/version.py | Código fuente |
| 28 | Roboflow SDK source `tests/test_version.py` | https://github.com/roboflow/roboflow-python/blob/74885a27/tests/test_version.py | Tests |
| 29 | Roboflow Python repo (README, docs) | https://github.com/roboflow/roboflow-python | Repo oficial |
| 30 | Roboflow Python docs index | https://github.com/roboflow/roboflow-python/blob/74885a27/docs/index.md | Doc oficial |
| 31 | Roboflow Python versions docs | https://roboflow.github.io/roboflow-python/core/version/ | Doc oficial |
| 32 | Roboflow docs Export Data REST API | https://docs.roboflow.com/developer/rest-api/export-data | Doc oficial |
| 33 | Roboflow docs Download Dataset CLI | https://docs.roboflow.com/developer/command-line-interface/download-a-dataset | Doc oficial |
| 34 | Roboflow docs Upload Dataset SDK | https://docs.roboflow.com/developer/python-sdk/upload-a-dataset | Doc oficial |
| 35 | Roboflow docs Create Dataset Version | https://docs.roboflow.com/developer/python-sdk/create-a-dataset-version | Doc oficial |
| 36 | DeepWiki Roboflow dataset-upload | https://deepwiki.com/roboflow/roboflow-python/4.1-dataset-upload | Doc generada |
| 37 | DeepWiki Roboflow dataset-download | https://deepwiki.com/roboflow/roboflow-python/4.2-dataset-download | Doc generada |
| 38 | Roboflow issue #125 (data.yaml paths) | https://github.com/roboflow/roboflow-python/issues/125 | Issue |
| 39 | Roboflow issue #240 (Incorrect Data Path YOLOv8) | https://github.com/roboflow/roboflow-python/issues/240 | Issue |
| 40 | Roboflow issue #88 (Wrong classes data.yaml) | https://github.com/roboflow/roboflow-python/issues/88 | Issue |
| 41 | Roboflow notebooks issue #306 (dataset.location empty) | https://github.com/roboflow/notebooks/issues/306 | Issue |
| 42 | Roboflow issue #333 (Relative paths data.yaml) | https://github.com/roboflow/roboflow-python/issues/333 | Issue |
| 43 | Roboflow issue #108 (Re-download same version) | https://github.com/roboflow/roboflow-python/issues/108 | Issue |
| 44 | Roboflow notebooks issue #69 (FileNotFoundError) | https://github.com/roboflow/notebooks/issues/69 | Issue |
| 45 | Roboflow notebooks issue #82 (Missing dataset name) | https://github.com/roboflow/notebooks/issues/82 | Issue |
| 46 | Roboflow notebooks issue #183 (Replace dataset.location) | https://github.com/roboflow/notebooks/issues/183 | Issue |
| 47 | Roboflow discuss data.yaml not find | https://discuss.roboflow.com/t/data-yaml-not-find-with-all-datasets-when-importing-into-google-colab/1514 | Foro |
| 48 | Stack Overflow #69594288 Roboflow Colab | https://stackoverflow.com/questions/69594288/load-dataset-from-roboflow-in-colab | SO |
| 49 | onnxsim issue #334 (Py 3.12 wheel) | https://github.com/daquexian/onnx-simplifier/issues/334 | Issue |
| 50 | onnxsim issue #367 (ir_version mismatch) | https://github.com/daquexian/onnx-simplifier/issues/367 | Issue |
| 51 | NVIDIA/TensorRT issue #1538 (EfficientNMS Maxwell) | https://github.com/NVIDIA/TensorRT/issues/1538 | Issue |
| 52 | NVIDIA/TensorRT issue #4383 (Gather rank-0 opset 19) | https://github.com/NVIDIA/TensorRT/issues/4383 | Issue |
| 53 | onnx-tensorrt operators.md release/8.2-GA | https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md | Doc oficial |
| 54 | Qengineering/YoloV8-TensorRT-Jetson_Nano | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo |
| 55 | triple-Mu/YOLOv8-TensorRT | https://github.com/triple-mu/YOLOv8-TensorRT | Repo |
| 56 | the0807/YOLOv8-ONNX-TensorRT | https://github.com/the0807/YOLOv8-ONNX-TensorRT | Repo |
| 57 | jws92/YOLOv8-TensorRT | https://github.com/jws92/YOLOv8-TensorRT | Repo |
| 58 | Linaom1214/TensorRT-For-YOLO-Series issue #112 | https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/112 | Issue |
| 59 | Kaggle/docker-python Dockerfile.tmpl | https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl | Repo oficial |
| 60 | Wandb pricing free tier | https://wandb.ai/site/pricing | Doc oficial |
| 61 | YouTube "How to Setup NVIDIA Jetson with Ultralytics YOLOv8" | https://www.youtube.com/watch?v=mUybgOlSxxA | Video |
| 62 | Ultralytics Live Session 6 (Jetson edge, mar 2023) | https://youtu.be/QGeP-Y6KMLM | Video |
| 63 | Ultralytics video "DeepStream Jetson Nano YOLOv8 Ep.82" | https://youtu.be/wWmXKIteRLA | Video |
| 64 | Trtutils YOLOv8 tutorial | https://trtutils.readthedocs.io/en/stable/tutorials/yolo/yolov8.html | Doc oficial |
| 65 | jetson-ai-lab.com YOLOv8 tutorial | https://jetson-ai-lab.com/tutorial_ultralytics.html | Tutorial |

---

## 10. Cross-references

- **[`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md)** — D6 (W&B nativo Track B), D8 (engine TRT en Nano), D14 (FP16-only).
- **[`compatibilidad-stack-cloud-jetson.md`](compatibilidad-stack-cloud-jetson.md)** — §7 stack Track B (pins de PyTorch, Ultralytics, onnxslim, numpy); §8 gotchas #6, #7, #8, #9, #10, #11.
- **[`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md)** — venv `trackb` (§4.5), kernel `trackb` (§4.6), bootstrap deps (§7).
- **[`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md)** — Gates 3 y 4 sobre el `.onnx` generado por §7 de este doc.
- **[`HANDOFF-implementacion-vastai-hf.md`](HANDOFF-implementacion-vastai-hf.md)** — Tarea #3' (adaptar `train_track_b_yolov8.ipynb`) usa la spec de §6 y §7.

---

**Fin del documento.** Cualquier cambio al dataset, hyperparameters, o flags de export ONNX requiere nueva ronda `/investiga`.
