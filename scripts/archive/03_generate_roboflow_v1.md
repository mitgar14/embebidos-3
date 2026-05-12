# 03 — Playbook Generate Version 1 en Roboflow UI

> **Workspace:** `embebidos3` · **Project:** `waste-3class-lwld8` · **Status:** 11.558 imgs / 13.873 bbox uploaded
> **Decisiones fundamentadas en:** `investigaciones/2026-05-05-preprocessing-roboflow.md` (95 fuentes, 2 rondas)
> **Esta acción es manual en la UI de Roboflow.** Los SDK/CLI no exponen toggles equivalentes — la generación de Versions debe hacerse desde [https://app.roboflow.com/embebidos3/waste-3class-lwld8/generate](https://app.roboflow.com/embebidos3/waste-3class-lwld8/generate)

## Estrategia: 2 versiones separadas

| Versión | Target | Resize | Export | Notebook destino |
|---|---|---|---|---|
| **1-A** (track-a-ssd-320) | SSD MobileNet v2 FPNLite 320×320 → TFLite INT8 | Fit (black edges) in **320×320** | `tfrecord` | `notebooks/train_track_a_ssd.py` |
| **1-B** (track-b-yolov8-416) | YOLOv8n 416×416 → ONNX → TensorRT FP16 | Fit (black edges) in **416×416** | `yolov8` | `notebooks/train_track_b_yolov8.py` |

Razón 2 versiones (no 1 a 640 universal): evitar resize en cascada (Roboflow 640 → letterbox interno Ultralytics 416 / fixed_shape_resizer TF 320), que introduce pérdida BILINEAR doble y desalinea bbox.

---

## Paso 1 — Preprocessing común a ambas versiones

> **Estructura real del UI Roboflow Generate (verificada 2026-05):** el bloque "Preprocessing" tiene DOS toggles principales con su propio modal (**Auto-Orient**, **Resize**) y un botón **"+ Add Preprocessing Step"** que abre un modal separado titulado **"Preprocessing Options"** con 10 pasos opcionales que se agregan al pipeline (Tile, Isolate Objects, Static Crop, Dynamic Crop, Grayscale, Auto-Adjust Contrast, Modify Classes, Random Sample, Filter Null, Filter by Tag).
>
> Cada versión generada tiene su propia config de Preprocessing (es per-version, no per-project). Si usas "Continue from previous Version", los toggles vienen prerellenados pero hay que reabrir el modal Resize para reconfirmar el tamaño (especialmente al cambiar de 320 a 416 entre 1-A y 1-B).

### 1.1 Toggles principales del bloque Preprocessing

| Toggle | Valor | Configuración | Justificación |
|---|---|---|---|
| **Auto-Orient** | ✅ ENABLED | (no requiere parámetros) | Roboflow Docs verbatim: *"Roboflow recommends defaulting to leaving this on."* Strip EXIF rotation. Safe en Jetson Nano (OpenCV `VideoCapture` entrega frames sin EXIF). |
| **Resize** | ✅ ENABLED | **`Fit (black edges) in {320 \| 416}` — per-versión, ver Pasos 4 y 5** | Modal separado: dropdown de 6 tipos (Stretch to / Fill with center crop / Fit within / Fit reflect edges / **Fit black edges** / Fit white edges) + input numérico que aparece tras seleccionar tipo. Equivalente a `LetterBox` Ultralytics. |

### 1.2 Pasos a AGREGAR vía "+ Add Preprocessing Step" → modal "Preprocessing Options"

De los 10 pasos que muestra el modal, agregar SOLO 2:

| Paso | Estado | Configuración | Justificación |
|---|---|---|---|
| **Modify Classes** | ✅ AGREGAR | Delete: `cardboard`, `miscellaneous`, `organic`, `metal` · Keep: `plastic`, `glass`, `paper` | Elimina bboxes de las 4 clases descartadas; las imágenes que SOLO tenían esas clases quedan marcadas como "null annotation" (no eliminadas todavía). |
| **Filter Null** | ✅ AGREGAR **(crítico, debe ir DESPUÉS de Modify Classes en el pipeline)** | (sin parámetros) | Roboflow Docs verbatim: *"Images marked as null annotation, or 'unannotated' after applying the Modify Classes tool, are the only ones affected when using Filter Null."* Sin este paso, las imgs vacías post-Modify Classes quedan como background contaminante. |

### 1.3 Pasos del modal "Preprocessing Options" que NO agregar

| Paso | Razón para NO agregar |
|---|---|
| Tile | Mis residuos ocupan 5-15% de la imagen — no son small objects. Tile aumenta volumen sin ganancia. |
| Isolate Objects | Convierte la tarea en classification (recorta a single object por imagen) — incompatible con detection task. |
| Static Crop | No tengo región fija de interés; la cámara ve toda la banda. |
| Dynamic Crop | Aún experimental en Roboflow; introduce variabilidad no controlada. |
| Grayscale | Plastic/glass/paper diferenciables por color (transparente vs blanco vs coloreado). |
| Auto-Adjust Contrast | No queremos modificar distribución de iluminación de training (mejor vía augmentations Brightness/Exposure). |
| Random Sample | Reduce dataset; ya tengo solo 11.558 imgs y desbalance fuerte — no quiero tirar muestras. |
| Filter by Tag | No estoy usando tags en el dataset. |

### 1.4 Verificación esperada post-pipeline

- Imágenes resultantes: ~9.000-11.000 (de 11.558 originales).
- Bboxes resultantes: ~10.439 (de 13.873 originales).
- Distribución bbox: plastic 7.128 / glass 1.927 / paper 1.384 (desbalance 5,15× plastic/paper).

> ⚠️ Si Filter Null elimina >3.500 imágenes, revisar matriz de co-ocurrencia clase-imagen — puede que muchas imágenes solo tuvieran cardboard/miscellaneous/organic/metal y ahora quedan vacías.

---

## Paso 2 — Augmentations comunes (3x multiplicador)

> **Estructura real del UI (verificada 2026-05):** botón **"+ Add Augmentation Step"** abre el modal **"Augmentation Options"** con dos secciones: **IMAGE LEVEL AUGMENTATIONS** (16 cards) y **BOUNDING BOX LEVEL AUGMENTATIONS** (11 cards). Cada card se agrega independientemente al pipeline.
>
> Premium Trial activo permite hasta 10x multiplicador; preferir 3x (Crasto 2024 confirma augmentation excesivo no es monotonic en benefit; con dataset 11k imgs, 3x = ~31k aug ya cubre masa).

### 2.1 Image-level (orden del modal "Augmentation Options")

| # | Augmentation | Decisión | Valor | Justificación |
|---|---|---|---|---|
| 1 | **Flip** | ✅ AGREGAR | Horizontal ON, Vertical OFF | Banda transportadora simétrica izq↔der. Vertical OFF: gravedad — residuos sobre banda nunca aparecen invertidos verticalmente. VisDA 2022 validado. |
| 2 | 90° Rotate | ❌ NO agregar | — | Conflicto con orientación gravitacional. |
| 3 | Crop | ❌ NO agregar | — | Random Crop puede dejar imagen sin objetos de clase positiva — riesgo alto para `paper` (1.384 bbox, clase minoritaria). |
| 4 | **Rotation** | ✅ AGREGAR | ±15° | Cámara diagonal sobre banda. Trash_Classifier (Chile, banda) usa ±10°; mantener ±15° por margen razonable. |
| 5 | **Shear** | ✅ AGREGAR | ±2° H, ±2° V | Proxy de PerspectiveWarp real (no disponible en Roboflow Basic). Leibniz Hannover 2025: perspective+lighting es la combinación más efectiva en líneas de producción. |
| 6 | Grayscale | ❌ NO agregar | — | Plastic/glass/paper diferenciables por color (transparente vs blanco vs coloreado) — convertir a B/N elimina señal discriminativa clave. |
| 7 | Hue | ❌ NO agregar | — | Tinte falso de papel puede confundirse con cartón. |
| 8 | **Saturation** | ✅ AGREGAR | ±20% | Vidrio transparente vs plástico transparente: vidrio tiene microreflexiones más coloreadas. Saturación moderada ayuda generalización. |
| 9 | **Brightness** | ✅ AGREGAR | ±25% | Iluminación industrial varía con hora. VisDA 2022 + Trash_Classifier hsv_v=0.4 ≈ ±25%. |
| 10 | **Exposure** | ✅ AGREGAR | ±15% | Complemento brightness; cubre overexposure típico de superficie metálica. |
| 11 | **Blur** | ✅ AGREGAR | ≤1.5 px | Frames cámara UVC en banda en movimiento → motion blur ligero. Validado VisDA 2022. |
| 12 | **Noise** | ✅ AGREGAR | **0.3% pixels** (alineado al ejemplo oficial Roboflow Docs) | Decisión basada en convergencia de 4 fuentes independientes (subagents research-web + research-academic, ronda 2026-05-05): (1) **Akbiyik ETH Zürich arXiv:2307.06855**: rango safe p<0.013 (1,3%) con MSSIM≥0,8; adversarial p>0.05 (5%) con MSSIM=0,046. (2) **YOLOv5/v8 `hyp.scratch.yaml` verificado**: NO incluye S&P noise en defaults — solo HSV. (3) **imgaug `SaltAndPepper(p=(0.0,0.05))` con activación p=0.5**: media efectiva 1,25% — alineado con threshold safe. (4) **Messai 2025 CNRS arXiv:2509.01332**: rangos 5-10% se usan para entrenar denoisers, NO augmentation. **Argumento crítico para `paper`:** texture de fibra tiene distribución espectral alta frecuencia → S&P >2% añade energía en mismas frecuencias espaciales → modelo confunde granularidad real con ruido. Peor en SSD MobileNet v2 FPNLite por feature fusion multi-escala. **Datasets Roboflow Universe similares (MultiTrash, waste-segregation con paper/glass/plastic, waste-detection-conveyor-belt): ninguno usa Noise.** Warning UI Roboflow probablemente persistirá a 0.3% pero es UX informativo (no bloquea generación). Doc oficial: *"Noise Augmentation set to .3%"* (única cifra ilustrada). |
| 13 | Cutout | ❌ NO agregar en V1 | — | Image-level (no preserva bboxes). Riesgo de borrar único bbox de paper (1.384 bbox = clase minoritaria 5,15× desbalance). Object-Oriented Cutout (Yim 2025) preservaría bboxes pequeños pero NO está en Roboflow. **Re-evaluar en V2** si paper class genera false negatives en eval real. |
| 14 | Mosaic | ❌ NO agregar | — | Roboflow Docs verbatim: *"Many modern models apply Mosaic as an online augmentation during training; applying it twice can cause undesirable results. **We do not recommend using this with Roboflow 3.0 or YOLOv8.**"* Track B ya aplica `mosaic=1.0` online (Ultralytics). |
| 15 | **Motion Blur** | ✅ AGREGAR | **Length 10 px · Angle 0° · Frames 2** | Decisión cuantitativa (subagents 2026-05-05): (1) **Sayed & Brostow CVPR 2021 arXiv:2011.14448**: blur de training NUNCA debe exceder el blur esperado en inferencia — kernel ~25 px sin aug causa caída 26 pp mAP@50 en Faster R-CNN COCO. (2) **Bujimalla et al. arXiv:2106.05437** escala empírica: 6 px=safe / 18 px=límite útil / 45 px=adversarial. (3) **Cálculo físico setup real**: banda 0,3 m/s × 1/30 s × (1.280 px sensor / 0,5 m FoV) = 25,6 px sensor → escalado a 416 px = **8-10 px desplazamiento real**. A 0,5 m/s worst-case = 13-15 px. (4) **Albumentations `MotionBlur` default `blur_limit=(3, 7)`**, YOLOv5/v8 hyps NO incluyen motion blur, ZeroWaste CVPR 2022 (banda real) NO lo usa. (5) **Bbox contamination**: kernel N px sangra N/2 px fuera del bbox → crítico para `paper` minoritaria; ≤15 px controla bleeding. **Length 10 px** = centro del rango realista, 14× menos que el 100 px default UI Roboflow (territorio adversarial). **Angle 0°** porque banda es eje horizontal puro (no usar 0-360° random). **Frames 2** = ejemplo oficial Roboflow Docs Motion Blur page + modela exposure de 2 posiciones consecutivas. **Aviso:** Motion Blur es Basic (no Enhanced/Premium); el icono ⊕ del UI es botón "Trial" no paywall. |
| 16 | Camera Gain | ❌ NO agregar en V1 | — | Simula amplificación sensor (high-ISO). Overlap con Brightness/Exposure. **Re-evaluar en V2** si recall low-light es bajo. |

### 2.2 Bounding-box-level (orden del modal "Augmentation Options")

> Decisión global: **NO agregar ninguna bbox-level augmentation en V1**. Las image-level ya cubren los regímenes de variación esperados, y agregar bbox-level multiplica el costo del aug sin ganancia documentada en datasets de tamaño similar al mío. Reservar bbox-level para V2 si el bench empírico identifica déficit específico por clase.

| # | Augmentation | Decisión | Razón para NO agregar |
|---|---|---|---|
| 1 | Bbox-Flip | ❌ | Redundante con image-level Flip Horizontal. |
| 2 | Bbox-90° Rotate | ❌ | Conflicto con orientación gravitacional. |
| 3 | Bbox-Crop | ❌ | Recorta dentro del bbox — degrada información del objeto en clase minoritaria paper. |
| 4 | Bbox-Rotation | ❌ | Redundante con image-level Rotation ±15°. Bbox-rotation crea artifacts en re-embedding. |
| 5 | Bbox-Shear | ❌ | Redundante con image-level Shear. |
| 6 | Bbox-Brightness | ❌ | Image-level Brightness ya cubre escenarios de iluminación. Aplicar solo a bbox crea inconsistencia objeto-fondo. |
| 7 | Bbox-Exposure | ❌ | Mismo argumento que Bbox-Brightness. |
| 8 | Bbox-Blur | ❌ | Image-level Blur ya cubre. Crear blur solo en objetos genera distribución no realista (en producción, todo el frame se blurra junto). |
| 9 | Bbox-Noise | ❌ | Mismo argumento que Bbox-Blur. |
| 10 | Bbox-Motion Blur | ❌ | Mismo argumento que Bbox-Blur. |
| 11 | Bbox-Camera Gain | ❌ | Mismo argumento que image-level Camera Gain (evaluar V2). |

---

## Paso 3 — Train/Test Split (step #2 del wizard Generate)

> **Estado inicial esperado tras el upload:** Roboflow puede mostrar **Train 11.557 (100%) / Valid 0% / Test 0%** porque las imágenes se subieron sin split previo en Annotate. **Este split inicial NO es usable.**

### 3.1 Acción

1. Click el botón **`Rebalance`** (esquina inferior derecha del bloque Train/Test Split).
2. En el modal de Rebalance, configurar:

| Split | Porcentaje | Imágenes esperadas (de 11.557) |
|---|---:|---:|
| Train | **70%** | ~8.090 |
| Valid | **20%** | ~2.311 |
| Test | **10%** | ~1.156 |

3. Si aparece toggle **"Stratified by class"** (o "Balanced split"): ✅ ENABLED. **Crítico** con desbalance 5,15× plastic/paper — sin estratificación, hay riesgo de que el test split tenga 0 imágenes de paper.
4. Apply → confirmar visualmente que la barra muestra ~70 / ~20 / ~10.
5. Click **Continue** para pasar al step #3 (Preprocessing).

### 3.2 Persistencia entre versiones

El rebalance se aplica **al proyecto, no a la versión**. Una vez rebalanceado durante la generación de 1-A, queda guardado y **la versión 1-B hereda el mismo split** automáticamente. Esto es exactamente lo que queremos:

- **1-A y 1-B son comparables** sobre los mismos splits → métricas mAP@50 directamente comparables.
- **NO hacer Rebalance otra vez en 1-B** (eso reasignaría aleatoriamente las imágenes y rompería la comparación).

### 3.3 Seed y reproducibilidad IEEE

- Roboflow asigna el seed internamente y NO lo expone en la UI. Para reportar reproducibilidad: anotar la fecha+hora del rebalance y el snapshot de números de imágenes por split (`8.090 / 2.311 / 1.156` o lo que reporte Roboflow tras Apply).
- El zip exportado por `project.version(N).download(...)` cachea bit-a-bit, así que la reproducibilidad efectiva está garantizada por (workspace, project, version).

---

## Paso 4 — Generar Version 1-A (Track A SSD)

En el wizard "Generate":

1. **Auto-Orient**: toggle ✅ ENABLED.
2. **Resize**: toggle ✅ ENABLED → click sobre el toggle para abrir modal → en el dropdown seleccionar **`Fit (black edges) in`** → aparece input numérico → escribir **`320`** en ambos lados (width y height) → click **Apply**.
3. **+ Add Preprocessing Step** → modal "Preprocessing Options" → click **Modify Classes** → seleccionar Delete `cardboard`, `miscellaneous`, `organic`, `metal` → Apply.
4. **+ Add Preprocessing Step** otra vez → modal "Preprocessing Options" → click **Filter Null** → Apply. (Verificar que aparezca DESPUÉS de Modify Classes en la lista del pipeline; si aparece antes, eliminar y re-agregar.)
5. Aplicar **Paso 2** (Augmentations comunes 3x).
6. Confirmar **Paso 3** (Splits 70/20/10 estratificado — viene heredado del proyecto).
7. Click **"Generate"**.
8. Tras generar (~5-30 min según volumen), nombrar como `Version 1-A — track-a-ssd-320` en la descripción.
9. **Export** → seleccionar formato `tfrecord` → click `Show Download Code` → guardar el snippet Python.

```python
# Snippet esperado (verificar API key + project ID):
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("embebidos3").project("waste-3class-lwld8")
dataset = project.version(1).download("tfrecord", location="./ds_tfr")
```

### Snippet final:

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="EAI36Thvq2WqxQnzGHY8")
project = rf.workspace("embebidos3").project("waste-3class-lwld8")
version = project.version(1)
dataset = version.download("tfrecord")
```                

> Variable de entorno requerida: `ROBOFLOW_API_KEY` (Private API key, no Publishable).

---

## Paso 5 — Generar Version 1-B (Track B YOLOv8)

Repetir Paso 4 con dos cambios:

1. **Resize**: abrir modal → `Fit (black edges) in` con dimensiones **`416 x 416`** (en lugar de 320). Si Roboflow ofrece "Continue from previous version", aceptar y luego **reabrir el modal Resize obligatoriamente** para cambiar 320 → 416 (no se hace automáticamente).
2. **Export**: formato `yolov8` (en lugar de tfrecord).

Nombrar como `Version 1-B — track-b-yolov8-416`.

> Roboflow asignará VERSION = 2 a esta segunda generación (la 1-A ya tomó VERSION = 1). **Actualizar `VERSION = 2` en `notebooks/train_track_b_yolov8.py`** antes de ejecutar.

```python
# Notebook Track B usará:
project.version(2).download("yolov8", location="./ds_yolo")
```

### Snippet final:

```python
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="EAI36Thvq2WqxQnzGHY8")
project = rf.workspace("embebidos3").project("waste-3class-lwld8")
version = project.version(2)
dataset = version.download("yolov8")
                
```

---

## Paso 6 — Verificación post-generación

Antes de lanzar entrenamiento, validar el zip exportado:

```bash
# Track A
ls -lh ./ds_tfr/
# Esperado: train/*.tfrecord (al menos 1 archivo), valid/*.tfrecord, test/*.tfrecord, _label_map.pbtxt
cat ./ds_tfr/_label_map.pbtxt
# Esperado: 3 items con name "plastic", "glass", "paper"

# Track B
ls -lh ./ds_yolo/
# Esperado: train/{images,labels}/, valid/{images,labels}/, test/{images,labels}/, data.yaml
cat ./ds_yolo/data.yaml
# Esperado: nc=3, names=['plastic','glass','paper'] (orden puede variar — verificar con `sorted()`)
```

Validación gotcha clases fantasma (issue roboflow-python#88):

```python
import yaml
with open("./ds_yolo/data.yaml") as f:
    meta = yaml.safe_load(f)
assert meta["nc"] == 3, f"esperado nc=3, got {meta['nc']}"
assert sorted(meta["names"]) == ["glass", "paper", "plastic"], f"clases inesperadas: {meta['names']}"
print(f"Dataset Track B validado: {meta}")
```

---

## Paso 7 — MD5 + métadatos para reproducibilidad IEEE

```python
import hashlib
from pathlib import Path

for label, path in [("Track A", "./ds_tfr"), ("Track B", "./ds_yolo")]:
    # Suma MD5 de todos los archivos del split (alternativa al MD5 del zip que no tenemos público)
    files = sorted(Path(path).rglob("*"))
    sizes = sum(f.stat().st_size for f in files if f.is_file())
    print(f"{label}: {len(files)} files, {sizes/1024/1024:.1f} MB total")
```

Citar en el informe IEEE como:
- Workspace: `embebidos3`
- Project ID: `waste-3class-lwld8`
- Version: `1-A` (Track A) y `1-B`/`2` (Track B)
- Generated: `2026-05-XX`
- Splits: `70/20/10 estratificado por clase`
- Preprocessing: `Auto-Orient ON, Modify Classes (delete cardboard/miscellaneous/organic/metal), Filter Null ON, Resize Fit (black edges) in {320|416}x{320|416}`
- Augmentations: `3x [Flip H, Rot ±15°, Shear ±2° H/V, Saturation ±20%, Brightness ±25%, Exposure ±15%, Blur ≤1.5px, Noise 0.3%, MotionBlur Length=10px Angle=0° Frames=2]`

---

## Paso 8 (opcional, aporte IEEE) — Version 1-B-alt-stretch

Si se decide ejecutar la ablación letterbox-vs-stretch como minor contribution publicable:

1. Repetir Paso 5 (Track B) pero con **Resize: `Stretch to 416 x 416`** (en lugar de Fit-black).
2. Nombrar como `Version 1-B-alt-stretch — ablation-letterbox-vs-stretch`.
3. Coste: 1 versión Premium Trial extra (~31k imgs aug). Premium Trial 14 días desde activación, dentro del límite 250k imgs/workspace.
4. Métrica a comparar: mAP@0.5 entre 1-B (Fit-black) y 1-B-alt-stretch sobre el mismo test split.
5. Brecha de literatura confirmada en investigación: no existe paper publicado con esta ablación específica en waste detection con aspect ratios mixtos.

---

## Próximos pasos tras generar V1-A y V1-B

1. **Lanzar Track A en Colab T4** con `notebooks/train_track_a_ssd.py` (~2-3 horas para 12k steps QAT).
2. **Lanzar Track B en Kaggle T4** con `notebooks/train_track_b_yolov8.py` (~1-2 horas para 100 epochs).
3. Ambos pueden correr en paralelo (Colab + Kaggle son cuentas independientes).
4. Tras eval, los outputs van al Jetson Nano:
   - Track A: `detect_int8.tflite` (~3 MB) → ejecutar `bench_jetson.py --model_a`
   - Track B: `yolov8n_waste.onnx` (~10 MB) → compilar a engine en el Nano: `trtexec --onnx=... --fp16 --workspace=1024` (15-45 min, RAM pico 3,5 GB) → ejecutar `bench_jetson.py --model_b`
5. Comparar latencia p50/p95/p99 + FPS sostenido + mAP. Decidir frontera de Pareto para deployment final.
