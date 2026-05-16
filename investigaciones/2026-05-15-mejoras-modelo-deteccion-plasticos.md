# Investigación — Mejoras al modelo de detección de plásticos (recall en small objects)

**Dominio:** `embebidos-3` — recall del clasificador YOLOv8n FP16 deployado en Jetson Nano sobre objetos pequeños/translúcidos.
**Proyecto:** clasificador 3 clases (`paper`/`glass`/`plastic`), demo 2026-05-26 (en T-11 días).
**Ronda 1 inaugural** — dominio nuevo. Cruza con HANDOFF-track-b (gates D2-D30), ronda training-headless (NMS Maxwell), ronda deploy-streaming-nano (Sprint 2 Phase A: engine FP16 13 MB, 43 FPS).

---

## Contexto operacional del problema observado

**Captura del 2026-05-15 — escena de testing en lab UAO:**
- 3 piezas pequeñas de plástico sobre fondo de tela cremosa (banda transportadora):
  - paquete amarillo "acks" (arriba, parcialmente fuera de la banda)
  - sobre verde claro (centro, contraste muy bajo con el fondo)
  - bolsa amarilla con texto (inferior, la única detectada)
- **Solo 1 detección**: `plástico 51%` (la bolsa inferior).
- 2 falsos negativos: paquete amarillo y sobre verde claro.

**Hardware/runtime confirmado:**
- Jetson Nano B01, JetPack 4.6.1, TensorRT 8.2.1.8, Maxwell `sm_53`.
- Engine `best_fp16.engine` 13 MB, validado a 43 FPS / 23 ms (Sprint 2 Phase A).
- Inferencia a `imgsz=416` (engine compilado con shape fijo).
- Cámara Logitech C920 OG (FOV 78°), banda 30-40 cm.

**Dataset:**
- Roboflow `embebidos3/waste-3class-lwld8` v1-B
- 17.910 train / 1.739 valid / 844 test
- Preprocessing: **Fit-black=0** (relleno negro) — mismatch teórico con LetterBox=114 default de Ultralytics (HANDOFF §6, sin medir hasta ahora).

**Plazos:**
- Demo: 2026-05-26 (11 días).
- RTX 4090 disponible en Vast.ai (provisioning ya validado).
- Recompilación TRT en Nano: 15-45 min.
- Re-export ONNX desde `.pt`: 0.5 h.

---

## Resumen ejecutivo — quick wins ordenados por costo×impacto

> **Diagnóstico previo más probable:** el problema es **multi-causal** — combinación de (a) `conf` default demasiado alto (0.25), (b) padding mismatch Roboflow Fit-black=0 vs Ultralytics LetterBox=114 sin verificar, (c) `imgsz=416` insuficiente para objetos pequeños, (d) escasa representación del dominio real (fondo cremoso de tela + iluminación lab) en el dataset Roboflow. NO es problema del engine TRT.

**Quick wins (priorizados en orden de ejecución):**

1. **D-0 (2 min, sin riesgo) — `conf=0.10`, `iou=0.45`** en el `model.predict()` del server FastAPI. Esperado: +15-25 pts recall absolutos. Si la escena es limpia (3 piezas sobre tela), el incremento de FP es bajo. **PRIMER PASO**.
2. **D-0 (15 min) — Verificar padding mismatch**: pasar la misma imagen de test con padding=0 y con padding=114 por el engine, comparar scores. Si difiere >5 pts, recompilar preprocesamiento del server con padding=0 (matching training).
3. **D-1 (30 min, sin recompilar engine) — SAHI con `supervision`**: tiles 208×208, overlap 0.4, NMS merge. Cada tile se redimensiona automáticamente al binding del engine (416×416). Esperado: +20-35 pts recall en small objects. Latencia 3-4× por frame (10-14 FPS efectivos en Nano).
4. **D-1 (30 min, offline) — TTA en `best.pt`** (no funciona en TRT) sobre el val set para medir techo de recall y decidir si el problema es pesos o datos.
5. **D-2 a D-3 (4-6 h labeling) — Active learning**: capturar 100-200 imágenes con la C920 real sobre el setup real (banda + tela cremosa + iluminación lab + 3 plásticos translúcidos en distintas posiciones), etiquetar con Roboflow assistant labeling.
6. **D-3 a D-4 (2 h training Vast.ai) — Fine-tune corto** desde `best.pt` con dataset ampliado: 50 épocas, `freeze=10` (backbone), `lr0=0.0005`, `imgsz=640`, augmentation reforzado (`copy_paste=0.5`, `hsv_v=0.6`, `mosaic=1.0`, `close_mosaic=10`).
7. **D-4 (1 h) — Re-export ONNX + recompilar engine a `imgsz=640`**: gana resolución a costo de FPS (43 → ~25-30 FPS, aún por encima del threshold 10 FPS). Si SAHI ya cubre, esto es opcional.
8. **D-5 a D-7 — Re-validar end-to-end con el dataset v2 + engine v2**, medir recall@conf=0.10 por clase (no solo mAP global). Iterar augmentation si alguna clase queda <80% recall.

**Riesgos a evitar (negative constraints aprendidos):**
- **NO aplicar augmentation doble** (Roboflow al exportar + Ultralytics en train) — daña la distribución. Roboflow solo preprocessing, Ultralytics todo el augmentation.
- **NO usar `dynamic=True` en export TRT** sobre Jetson Nano JP 4.6.1 — TensorRT 8.x es inestable con shapes dinámicos en Maxwell. Compilar engines fijos separados.
- **NO modificar arquitectura YAML** (P2 head, attention modules) en plazo de 11 días — aunque la literatura demuestra +5-6 pts mAP (SOD-YOLOv8 [arXiv:2408.04786](https://arxiv.org/abs/2408.04786)), el costo de re-train + recompilar engine no se justifica. Citable en informe IEEE como ablation propuesta.
- **NO MIGRAR a YOLOv10**: operador `fmod`/Mod en cabeza NMS-free **no soportado en TRT 8.2.1.8**. Requiere TRT ≥8.4 o cirugía manual del grafo ONNX. Inviable en JP 4.6.1. Confirmado en issues THU-MIG/yolov10 #75 y #129.
- **NO MIGRAR a YOLOv11/v9** sin medir: en DOTAv1.5 (small objects aerial, análogo a small plastic) **YOLOv8n supera a YOLOv11n por 4 pts mAP@0.5 small** (67.88% vs 64.33%, Tariq & Javed 2025 [arXiv:2504.09900](https://arxiv.org/abs/2504.09900)) — el módulo C2f de v8 preserva resolución espacial mejor que la atención de v11.

**Notas técnicas clave para SAHI en Nano:**
- **PR #1336 obss/sahi (abril 2026)** convierte PyTorch en dependencia opcional. Combinado con Ultralytics que carga el `.engine` directamente vía `YOLO("best.engine")`, SAHI puede orquestar slices con NumPy puro.
- **Verificación rápida (5 min)**: `from ultralytics import YOLO; model = YOLO("best.engine")`. Si funciona en el venv del Nano → SAHI viable vía `model_type="ultralytics"` con path al engine.
- Alternativa C++ (HouYanSong/tensorrtx-yolov8-sahi): **44-46 ms/frame con SAHI, 15 FPS @ 1080p en Orin Nano**. Recompilable a Maxwell sm_53 con ajuste de arch.

---

## Ronda 1 — 2026-05-15 (profundidad media)

### Track A — Investigación de tuning práctico (research-web)

> Agent: `research-web`. Foco: tuning sin/con re-train corto. Output: 7 frentes verificados con docs oficiales Ultralytics 2025-2026 y código ejecutable.

#### A1.1 Confidence threshold y NMS IoU

**Defaults documentados** ([docs.ultralytics.com/usage/cfg](https://docs.ultralytics.com/usage/cfg), 2026-04):
- `conf: float = 0.25` — "Objects detected with confidence below this threshold will be disregarded."
- `iou: float = 0.7` — IoU para NMS en `predict`. (Confusamente, internamente `non_max_suppression()` en `ultralytics/utils/nms.py` usa `iou_thres=0.45` hardcoded como default cuando se llama directamente.)

**Curva PR como herramienta canónica**: ejecutar `model.val(conf=0.001, iou=0.6, plots=True)` genera `runs/val/expN/{P,R,PR,F1}_curve.png` + `confusion_matrix.png`. El umbral óptimo es:
- F1-max → balance precision/recall
- Punto donde `R ≥ 0.90` por clase → si el objetivo es recall (caso del proyecto, no detectar = fallo industrial)

**Snippet aplicable al server FastAPI actual (Sprint 2 Phase B):**

```python
from ultralytics import YOLO
model = YOLO("best.engine")  # engine TRT ya compilado

results = model.predict(
    source=frame_np,
    conf=0.10,             # bajar de 0.25
    iou=0.45,              # bajar de 0.7
    imgsz=416,
    half=True,
    device=0,
    verbose=False,
)
```

**Soft-NMS / DIoU-NMS**: para escena de 3 piezas no superpuestas, la ganancia marginal no justifica parchear `ultralytics/utils/nms.py`. Implementación PyTorch disponible en [github.com/MrParosk/soft_nms](https://github.com/MrParosk/soft_nms) si se quisiera explorar. **Decisión:** mantener NMS estándar, bajar `iou=0.45`.

#### A1.2 Image size en inferencia vs engine compilado

**Hard fact**: un engine TRT compilado a `imgsz=416` **NO** acepta frames 640×640. El binding de entrada es fijo `[1, 3, 416, 416]`. Doc oficial confirma: *"The first few inference calls with a model exported to TensorRT can be expected to have longer preprocessing times. This may also occur when changing imgsz during inference, especially when imgsz is not the same as what was specified during export."* ([github.com/ultralytics/ultralytics — tensorrt.md](https://github.com/ultralytics/ultralytics/blob/main/docs/en/integrations/tensorrt.md)).

**Camino para subir a 640:**
```bash
yolo export model=best.pt format=engine half=True imgsz=640 dynamic=False device=0
```
- ONNX: 0.5 h
- Engine TRT en Nano: 15-45 min (cerrar X11 antes para liberar RAM)
- FPS esperado: 20-28 (vs 43 a 416)

**`dynamic=True` descartado**: inestable en TRT 8.x sobre Maxwell. Compilar dos engines fijos separados (416 para streaming, 640 para validación offline si se necesita).

#### A1.3 Test Time Augmentation (TTA)

```python
results = model.predict(source=img, augment=True, conf=0.10, iou=0.45)
```

**LIMITACIÓN CRÍTICA**: TTA **no está soportado en TRT engines**, solo en `.pt` PyTorch. Ganancia típica: +1-3 pts mAP@0.5, mejor en small objects. Costo: 2-3× latencia.

**Uso recomendado en este proyecto**: TTA offline sobre `.pt` con val set para medir techo de recall alcanzable. Si TTA da +X pts y el engine no, se sabe que el problema NO son los pesos sino la inferencia (slicing, multi-scale).

#### A1.4 SAHI con `supervision` sobre engine TRT compilado

**LA TÉCNICA MÁS POTENTE EN INFERENCIA SIN RE-TRAIN.**

```python
import supervision as sv
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.engine")

def slicer_callback(slice_img: np.ndarray) -> sv.Detections:
    results = model.predict(slice_img, conf=0.10, iou=0.45,
                            imgsz=416, half=True, device=0, verbose=False)[0]
    return sv.Detections.from_ultralytics(results)

slicer = sv.InferenceSlicer(
    callback=slicer_callback,
    slice_wh=(208, 208),       # tiles de mitad del frame de captura
    overlap_ratio_wh=(0.4, 0.4),
    overlap_filter_strategy=sv.OverlapFilter.NON_MAX_MERGE,
    iou_threshold=0.3,
)

frame = cv2.imread("frame.jpg")
detections = slicer(frame)
```

**Compatibilidad con Nano JP 4.6.1 — gotcha crítico:**
- `supervision` requiere Python ≥3.7. Nano viene con Python 3.6.9 system.
- Si el server corre en venv `/opt/venv/embebidos3` con Python ≥3.7 (verificar), funciona.
- **Si solo está disponible Python 3.6**: implementar slicer manualmente con NumPy pure (no librería externa). Patrón:
  1. `np.array_split` el frame en N tiles con overlap 40%.
  2. Por cada tile: `model.predict()`, recoger bboxes en coords del tile.
  3. Trasladar bboxes a coords del frame original (offset).
  4. NMS global con `cv2.dnn.NMSBoxes` o función custom.

**Latencia estimada**: 3-4× por frame (con 4-6 tiles + overlap). En Nano: 23 ms × 4 ≈ 90-100 ms por frame → 10-14 FPS efectivos. **Aún cumple threshold MVP (10 FPS)**.

#### A2 Augmentation para fine-tune corto

**Hiperparámetros directamente en `.train()`** (no requiere `hyp.yaml`):

```python
from ultralytics import YOLO
model = YOLO("best.pt")   # partir del best actual

results = model.train(
    data="waste-3class-v2.yaml",   # dataset ampliado con setup real
    imgsz=640,             # subir de 416
    epochs=50,
    batch=32,              # RTX 4090
    lr0=0.0005,            # LR muy bajo para fine-tune (1/20 del inicial)
    lrf=0.01,
    optimizer="AdamW",
    # --- augmentation reforzado para small/translúcido ---
    mosaic=1.0,
    copy_paste=0.5,        # pega objetos en otras imágenes (IoA<0.3)
    mixup=0.15,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.6,             # alto para variaciones de iluminación lab
    perspective=0.0005,
    scale=0.7,
    fliplr=0.5,
    flipud=0.0,            # banda horizontal — flipud no físico
    degrees=5.0,
    # --- estabilización ---
    close_mosaic=10,       # desactiva mosaic últimos 10 epochs
    freeze=10,             # CONGELA backbone (capas 0-9) — anti-catastrophic forgetting
    patience=15,
    device=0,
    project="runs/finetune",
    name="waste_v1b_ft",
)
```

**Notas clave:**
- `copy_paste`: implementación nativa Ultralytics (basada en arxiv.org/abs/2012.07177), umbral IoA 0.3, ideal para sintetizar plásticos sobre fondos cremosos nuevos.
- `close_mosaic=10`: estabiliza convergencia al final, ve imágenes realistas.
- `freeze=10`: congela las primeras 10 capas del backbone, solo actualiza neck + detection head. **Crítico** para evitar olvido del conocimiento general aprendido en train inicial (17.910 imágenes).

**CutMix añadido en Ultralytics 8.3.119** (PR #19870, 2026-04): `cutmix=0.5` ahora disponible. Probar después de evaluar si copy_paste + mosaic ya son suficientes.

#### A3 Roboflow vs Ultralytics — dónde aplicar augmentation

**Regla canónica documentada por la comunidad:** **NO duplicar**. Si Roboflow exporta con augmentation, Ultralytics aplica el suyo encima → distribución contaminada.

**Decisión para este proyecto:**
- **Roboflow v2** (próxima versión del dataset): solo preprocessing (resize 640×640 letterbox, padding=114) + outputs=1 (sin augmentation). El argumento de "outputs=3" para triplicar el dataset físicamente solo aplica si el setup real es muy escaso (<50 imgs nuevas).
- **Ultralytics `.train()`**: todo el augmentation vía parámetros Python (snippet §A2).

#### A4 Class weights / focal loss

**Ultralytics 8.4.x NO expone `class_weights`** en API pública. Confirmado revisando `docs.ultralytics.com/usage/cfg` 2026-04.

**Workarounds:**
- **Sobre-muestrear** la clase deficitaria en Roboflow (duplicar imágenes antes de exportar v2).
- **Parchear `loss.py`** para focal loss: arriesgado, rompe compatibilidad de updates. **Descartado** para 11 días.

#### A5 Hard negative mining + FiftyOne

**No hay pipeline nativo en Ultralytics**. Flujo manual:

```python
# Paso 1: validación con conf muy bajo
results = model.val(data="waste-3class.yaml", conf=0.001, iou=0.6,
                    save_json=True, plots=True)

# Paso 2: FiftyOne para visualizar FN
import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_dir="datasets/waste-3class/",
    dataset_type=fo.types.YOLOv5Dataset,
)
dataset.load_annotations("runs/val/expN/predictions.json", label_field="predictions")
results = dataset.evaluate_detections("predictions", gt_field="ground_truth",
                                       eval_key="eval")
fn_view = dataset.filter_labels("eval", fo.ViewField("eval") == "fn")
session = fo.launch_app(fn_view)
```

Las imágenes con FN se copian a un subset "hard" y se duplican en `train/images` + `train/labels` para fine-tune con doble peso implícito.

#### A6 Active learning — cuántas imágenes capturar

**Best practice de Ultralytics community ([t/717](https://community.ultralytics.com/t/looking-for-best-practices-for-fine-tuning-yolov8-on-custom-dataset/717), 2025-01):** *"The MOST IMPORTANT factor for improving your model's performance will be collecting more annotated data."*

**Para el caso (dominio shift conocido — banda UAO + C920 + tela cremosa + iluminación lab):**
- 100-200 imágenes nuevas del setup real bastan para corregir el dominio.
- Etiquetar con Roboflow assistant labeling (model-assisted con `best.pt` actual + revisión manual).
- Mezcla: ~70% del dataset original Roboflow v1-B + ~30% capturas reales (peso ajustado por sobre-muestreo si es necesario).

```python
from roboflow import Roboflow

rf = Roboflow(api_key="...")
project = rf.workspace("embebidos3").project("waste-3class-lwld8")

project.upload_model(model_type="yolov8", model_path="best.pt")
project.generate_version(
    preprocessing={
        "resize": {"width": 640, "height": 640, "format": "Letterbox"},
    },
    augmentation={},
    labeling_method="model-assisted",
)
```

#### A7 Diagnóstico cuantitativo del problema

**Roboflow Health Check** (`Dataset → Health Check` en UI): histograma de objetos por imagen, heatmap de anotaciones, distribución de clases en train/val/test, aspect ratio.

**Tamaño de bbox en píxeles** (no expuesto en Health Check, script propio):

```python
import os
import numpy as np

label_dir = "datasets/waste-3class/train/labels"
sizes = []
for f in os.listdir(label_dir):
    with open(os.path.join(label_dir, f)) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 5:
                w, h = float(parts[3]), float(parts[4])
                sizes.append((w * 640, h * 640))

sizes = np.array(sizes)
print(f"Median bbox: {np.median(sizes, axis=0)} px")
print(f"P10 bbox:    {np.percentile(sizes, 10, axis=0)} px")
print(f"< 32px wide: {(sizes[:,0] < 32).mean()*100:.1f}%")
```

**Si bboxes < 32 px (a 640) supera 20%**: small object detection estructural. A 416, equivalente: `ancho_px_416 = ancho_px_640 × 416/640 = 0.65 ×`.

**Métrica prioritaria:** `Recall@conf=0.10` por clase, no `mAP@0.5:0.95`. Para una aplicación industrial donde no detectar es el error crítico:

```python
metrics = model.val(data="waste-3class.yaml", conf=0.10, iou=0.5, plots=True)
print(metrics.box.r)   # recall por clase
print(metrics.box.mp)  # mean precision
print(metrics.box.mr)  # mean recall
```

#### A8 Padding mismatch — sospecha #1 a verificar

**Roboflow Fit-black=0 vs Ultralytics LetterBox=114**: documentado en HANDOFF §6, no medido hasta ahora.

**Verificación rápida en 15 min:**
```python
import cv2, numpy as np
from ultralytics import YOLO

img = cv2.imread("test_3plasticos.jpg")
H, W = img.shape[:2]
target = 416

# Letterbox manual con padding=0
def letterbox(img, new_shape, color):
    h, w = img.shape[:2]
    r = min(new_shape[0]/h, new_shape[1]/w)
    nw, nh = int(w*r), int(h*r)
    resized = cv2.resize(img, (nw, nh))
    pad_w = new_shape[1] - nw
    pad_h = new_shape[0] - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=color)

img_pad0   = letterbox(img, (target, target), (0, 0, 0))
img_pad114 = letterbox(img, (target, target), (114, 114, 114))

model = YOLO("best.engine")
r0   = model.predict(img_pad0,   conf=0.001, iou=0.45, verbose=False)[0]
r114 = model.predict(img_pad114, conf=0.001, iou=0.45, verbose=False)[0]

print(f"pad=0:   {len(r0.boxes)} dets, mean conf={r0.boxes.conf.mean():.3f}")
print(f"pad=114: {len(r114.boxes)} dets, mean conf={r114.boxes.conf.mean():.3f}")
```

**Si pad=0 da significativamente más detecciones / mejor conf** → el server FastAPI debe aplicar padding=0 en su preproceso (matching training). Modificar el `letterbox()` del server:

```python
# En el preprocessing del server (Sprint 2 Phase B):
preprocessed = letterbox(frame, (416, 416), color=(0, 0, 0))  # NO 114
```

**Si pad=114 da significativamente más** → el dataset v2 debe re-exportarse con padding=114 (cambiar Roboflow preprocessing a "Letterbox" en vez de "Fit (Black)").

---

### Track A — Estado del arte (research-academic)

> Agent: `research-academic`. Foco: viabilidad SAHI sobre TRT en Maxwell sm_53, P2 head, papers 2024-2026 waste, decisión de migración YOLO. Output: 4 frentes con benchmarks verificados + 10 referencias académicas con DOIs/arXiv.

#### A9 SAHI en Jetson Nano con TRT FP16 — ¿factible sin PyTorch?

**Paper fundacional:** Akyon, Altinuc, Temizel (IEEE ICIP 2022 oral, [arXiv:2202.06934](https://arxiv.org/abs/2202.06934)). Resultados sobre VisDrone2019: **+6.8 pts AP50 solo con inferencia sliceada** (sin re-train), hasta **+14.5 pts AP50 con slicing en train + inferencia**. Sobre xView (satélite, objetos muy pequeños): FCOS baseline 2.2% AP50 → SF+SAHI 17.1%.

**Restricción crítica del proyecto Nano JP 4.6.1:** Python 3.6.9 system, no PyTorch instalado.

**Tres caminos identificados:**

| Opción | Implementación | Viabilidad Maxwell sm_53 | Riesgo |
|---|---|---|---|
| **A: obss/sahi PR #1336 (torch-opcional)** | `pip install sahi` sin `[torch]` + Ultralytics que carga el `.engine` directamente con `YOLO("best.engine")` | Recomendada. Si Ultralytics se instala con `--no-deps` + deps mínimas en Nano, SAHI orquesta slices con NumPy puro. | Bajo |
| **B: HouYanSong/tensorrtx-yolov8-sahi (C++)** | C++ con engine TRT INT8. Benchmark: **44-46 ms/frame con SAHI, 15 FPS @ 1080p con ByteTrack en Orin Nano**. | Compilado para Orin Nano (Ampere). Recompilar para sm_53 Maxwell requiere ajustar arch. | Medio |
| **C: obss/sahi PR #1046 (cerrado, copiar manualmente)** | `yolov8engine.py` para cargar engines TRT en SAHI directamente. 182 líneas, código disponible aunque PR no mergeado. | Camino quirúrgico si Ultralytics no se puede instalar en Nano. | Bajo-medio |

**Limitación SAHI oficial estable (0.11.36):** batch_size=1 forzado para TRT. Confirmado en discussion #808. **PR #1336 (abril 2026) lo soluciona**.

**Configuración recomendada para imgsz 416:**
```bash
sahi predict --slice_width 256 --slice_height 256 \
  --overlap_height_ratio 0.2 --overlap_width_ratio 0.2 \
  --model_type ultralytics --model_path best.engine
```

(Note: el agent web propuso `(208, 208)` con overlap 0.4. La diferencia es estilística — slice más grande con overlap menor genera ~4 inferencias por frame. Usar `(256, 256, 0.2)` como default canónico Akyon et al., ajustar empíricamente.)

**Benchmark consolidado (datos verificados + 1 inferred):**

| Plataforma | Motor | Modo | FPS | Fuente |
|---|---|---|---|---|
| Jetson Orin Nano 8 GB | TRT INT8 | SAHI 2×2 slices | 15 FPS @ 1080p | HouYanSong (sept 2025) |
| RTX 3090 | TRT FP16 | SAHI 640×640 sobre 1432×4089 | ~1.0 s/img | obss/sahi PR #1042 |
| Jetson Nano 4 GB Maxwell | TRT FP16 | Inferencia directa | 43 FPS / 23 ms | Este proyecto Sprint 2A |
| Jetson Nano 4 GB Maxwell | TRT FP16 | SAHI 2×2 slices (estimado) | **~9-11 FPS** | *Inferred*: 4× overhead lineal |

**Acción inmediata sugerida (snippet validable en 5 min):**
```python
# Verificar que Ultralytics carga el engine
from ultralytics import YOLO
model = YOLO("best.engine")
# Si funciona → SAHI puede usar model_type="ultralytics" + UltralyticsDetectionModel
```

#### A10 P2 small-object head — costo/beneficio cuantificado

**Paper canónico:** Khalili, Smyth (Columbia, IEEE ICIP 2024, [arXiv:2408.04786](https://arxiv.org/abs/2408.04786)) — *SOD-YOLOv8: Enhancing YOLOv8 for Small Object Detection in Traffic Scenes*.

**Resultados sobre VisDrone2019:**

| Modelo | mAP@0.5 | mAP@0.5:0.95 | Recall | Params | Latencia |
|---|---|---|---|---|---|
| YOLOv8s baseline | 40.6% | 24.0% | 40.1% | 11.1 M | 7.8 ms |
| **SOD-YOLOv8s** (P2 head + EMA + PIoU) | **45.1%** | **26.6%** | **43.9%** | 11.5 M | 11.6 ms |

Ganancia: **+4.5 pts mAP@0.5, +3.8 pts recall absolutos** por solo +400K params y +3.8 ms latencia.

**Viabilidad TRT 8.2 sm_53:**
- Operadores: Conv + BN + C2f → totalmente soportados, opset 13.
- Export: `yolo export model=best.pt format=onnx opset=13` → `trtexec --fp16` limpio.
- **OOM risk en Maxwell 4 GB**: P2 head sobre 160×160×64 (FP16) = ~6.5 MB activaciones extra. Engine actual 416 ≈ 400-500 MB VRAM. P2 + imgsz 640 estimado 700-900 MB. Margen suficiente si OS no saturado.

**Costo:** **requiere reentrenamiento completo**. 4-8 h Vast.ai RTX 4090. Ocupa días escasos del sprint.

**Decisión:** **descartado para 11 días**, registrar en informe IEEE como ablation propuesta.

#### A11 imgsz 416 vs 640 — evidencia indirecta

No hay benchmark directo en waste detection 3-class. **Por extrapolación** (Tariq, Javed 2025, [arXiv:2504.09900](https://arxiv.org/abs/2504.09900)):
- DOTAv1.5 aerial 1024×1024: YOLOv8n alcanza 67.88% mAP@0.5 small.
- COCO 640×640: YOLOv8n alcanza 43.36% mAP@0.5 small.
- Escalado 416 → 640 da ~54% más píxeles por objeto → empíricamente +3-6 pts recall en small (sin medir en waste 3-class). **Nuestro registro empírico sería novedoso.**

#### A12 Papers 2024-2026 waste detection en edge

| # | Autor / Año | Aporte | Aplicabilidad al proyecto |
|---|---|---|---|
| 1 | **Abid et al. 2025** ([arXiv:2508.18799](https://arxiv.org/abs/2508.18799)) — *Robust Label-Efficient Deep Waste Detection* | Pseudo-labeling ensemble sobre ZeroWaste alcanza **51.6 mAP**, supera supervised completo. Prompts LLM-optimized para OVOD. | Estrategia pseudo-label aplicable a frames no etiquetados del proyecto. Ambicioso para 11 días. |
| 2 | Pan Li, Xu, Liu 2024 — *Enhanced YOLOv8 Lightweight CNN* | CG-HGNetV2 backbone + MSE-AKConv attention. +4.80% precision, +1.30% mAP@50 sobre YOLOv8s, -6.55% params. | Citable como antecedente de attention en waste. |
| 3 | **Xiao, Luo 2024** — *EA-YOLO* (WaRP dataset) | Adaptive Inner-IOU + ISRA attention + FlexiC2f deformable. **+6.1 pts mAP@0.5, +5.1 mAP@0.5:0.95.** | Adaptive Inner-IOU = hard example mining implícito — directamente relevante para plásticos translúcidos sub-estimados. |
| 4 | Ren, Li, Gao 2024 — *MRS-YOLO* | SlideLoss_IOU + RepViT transformer + multi-dim feature fusion. **+3.6 pts mAP@50, -15.09% model size.** | SlideLoss_IOU es la técnica más relevante para small objects en waste. |
| 5 | **Kovitvadhi et al. 2024** — *YOLO-Based Waste Smart Recycling* | YOLOv9 supera YOLOv8s en escenarios complejos (Medium dataset, múltiples objetos), atribuido a PGI. YOLOv8s gana en Easy. | Confirma que para escenarios simples YOLOv8s sigue siendo competitivo. |
| 6 | Khalili, Smyth 2024 ([arXiv:2408.04786](https://arxiv.org/abs/2408.04786)) — *SOD-YOLOv8* | Ya citado en A10. P2 head + EMA + PIoU. | Future work IEEE. |
| 7 | Kuang, Bhandari, Gao 2024 ([arXiv:2410.09975](https://arxiv.org/abs/2410.09975)) — *Optimizing Waste Management Garbage Classification* | RPI paper. Garbage classification general. | Citable como antecedente. |
| 8 | Alqahtani et al. 2024 ([arXiv:2409.16808](https://arxiv.org/abs/2409.16808)) — *Benchmarking Edge Devices* | Monash/Melbourne benchmark sobre Jetson. | Citable para defender hardware Nano. |

**Augmentation con mayor evidencia para small/translucent waste:**
- **Copy-Paste** (Ghiasi et al. 2021): el paper de helmets (Luong et al. 2024) reporta **+7.43 pts mAP** con CP + ensemble sobre clases imbalanced. Directamente aplicable a clase plastic minoritaria.
- **Mosaic 9-imagen**: combinar 9 imgs en grid fuerza objetos relativamente más pequeños — aplicable vía `mosaic=1.0` o `mosaic9=0.1` en Ultralytics.
- **Albumentations RandomBrightnessContrast + HueSaturationValue**: para plásticos translúcidos en fondo cremoso, las variaciones de bajo contraste/saturación son críticas.

#### A13 Migración de versión YOLO — decisión cuantitativa

**NO MIGRAR. Permanecer en YOLOv8.** Justificación:

| Modelo | mAP@0.5 small (COCO) | mAP@0.5 small (DOTAv1.5 aerial) | TRT 8.2 export | Veredicto |
|---|---|---|---|---|
| **YOLOv8n** | 43.36% | **67.88%** | Limpio, opset 13 | **Mantener (baseline actual)** |
| YOLOv9t | 44.08% | 61.71% (-6.17 pts) | Workarounds menores | Sin ganancia clara |
| **YOLOv10n** | 48.26% | 51.16% (-16.72 pts) | **BLOQUEADO** — operador `fmod`/Mod no soportado en TRT 8.2.1.8 (issue THU-MIG/yolov10 #75, #129) | **Descartar definitivamente** |
| YOLOv11n | 47.26% | 64.33% (-3.55 pts) | Parcial, requiere opset≥15 para deformable attention | Riesgo alto, ganancia incierta |

Fuente métricas: Tariq, Javed 2025 ([arXiv:2504.09900](https://arxiv.org/abs/2504.09900)).

**Negative constraint para informe IEEE y memoria mnemon:**
- **YOLOv10 requiere TRT ≥8.4** o cirugía manual del grafo ONNX (remover `Mod`, post-procesar). Inviable en JP 4.6.1.
- En DOTAv1.5 (objetos pequeños aéreos, similar a nuestro caso de plásticos pequeños), **YOLOv8n supera a YOLOv11n por 4 pts mAP@0.5 small** (67.88% vs 64.33%) — atribuido al módulo C2f que preserva resolución espacial. La arquitectura de atención de YOLOv11 se desborda con muchos objetos pequeños por imagen.

#### A14 Síntesis del agent academic

> *"El problema tiene dos causas probables superpuestas: (1) estructural — con imgsz 416 y 3 cabezas P3/P4/P5, el feature map más fino es 52×52 efectivo, y un objeto 30×30 px ocupa <5% del campo receptivo de P3, límite inferior de detección confiable. (2) umbral — conf 51% en la detección exitosa sugiere que las otras 2 piezas están suprimidas por NMS o por el threshold default 0.25, no por ser invisibles para el modelo."*

**Estrategia recomendada por el academic** (ranking por costo/impacto):
1. SAHI torch-free + bajar threshold conf — sin re-train, no toca engine TRT actual.
2. Re-entrenar con imgsz 640 + Copy-Paste + Mosaic9 — best costo-beneficio si hay 3-4 días.
3. P2 head — gana +4-5 pts mAP@0.5 a costo de re-train + recompilar engine.
4. NO migrar versión YOLO (YOLOv10 bloqueado por TRT 8.2, v11/v9 sin ganancia clara en small).

---

### Track B — Descubrimiento web (3 búsquedas discover.py)

#### B1 — SAHI + TRT en Jetson Nano (14 fuentes)

**Repos canónicos para SAHI + TRT engine** (no PyTorch):

| Repo | Stars | Relevancia |
|---|---|---|
| [obss/sahi PR #1046 — yolov8engine.py](https://github.com/obss/sahi/pull/1046) | (rama, no merged) | **CLAVE** — clase que envuelve YOLOv8 TRT engine como modelo SAHI. Requiere `config_path` JSON con `{task, names, imgsz, half}`. Author dejó código completo + función `record_metadata_json` para extraer config del `.pt` original. PR cerrado pero código reutilizable directamente. |
| [leon0514/trt-sahi-yolo](https://github.com/leon0514/trt-sahi-yolo) | 95 | C++/CUDA + Python bindings. Slicing implementado en CUDA. Soporta YOLOv8/v11/v5/D-FINE. Requiere ONNX dinámico + `trtexec --minShapes/--maxShapes`. **Demuestra factibilidad técnica**, complejidad de adopción alta. |
| [HouYanSong/tensorrtx-yolov8-sahi](https://github.com/HouYanSong/tensorrtx-yolov8-sahi) | (low) | Deploy YOLOv8-SAHI INT8 en Jetson — mismo patrón pero INT8 (no aplica Maxwell sin tensor cores). |
| [IrDIE/YOLO8_SAHI](https://github.com/IrDIE/YOLO8_SAHI) | (low) | Implementación SAHI YOLOv8 vanilla, no TRT-specific. |
| [Qengineering/YoloV8-TensorRT-Jetson_Nano](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano) | 46 | C++ Jetson Nano YOLOv8 TRT. NO incluye SAHI. Útil como referencia de baseline. |

**Documentación oficial:**
- [docs.ultralytics.com/guides/sahi-tiled-inference](https://docs.ultralytics.com/guides/sahi-tiled-inference/) — guía completa SAHI + Ultralytics. Pattern recomendado: `slice_height=256, slice_width=256, overlap_height_ratio=0.2, overlap_width_ratio=0.2`. Para nuestro frame 416, ajustar a `(208, 208)` con overlap 0.4.
- [github.com/ultralytics/notebooks — sahi.ipynb](https://github.com/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-with-sahi.ipynb) — notebook ejecutable.
- [jetson-ai-lab.com/tutorial_ultralytics.html](https://jetson-ai-lab.com/tutorial_ultralytics.html) — tutorial oficial Ultralytics + TRT en Jetson (referencia estándar).

**Forum NVIDIA crítico:**
- [Run YoloV8 with Jetson Inference on Jetson Nano](https://forums.developer.nvidia.com/t/run-yolov8-with-jetson-inference-on-jetson-nano/245322) — referencia de issues comunes.
- [Issue ultralytics #12988](https://github.com/ultralytics/ultralytics/issues/12988) — Docker image YOLOv8 + TRT en Nano (workaround documentado).

#### B2 — Confidence/NMS tuning + waste detection (19 fuentes)

**Issues clave Ultralytics:**

| Issue | Insight aplicable |
|---|---|
| [#5315 Queries on Confidence/NMS thresholds](https://github.com/ultralytics/ultralytics/issues/5315) | Defaults durante validación: `conf=0.001, NMS=0.6, max_det=300`. Defaults durante predict: `conf=0.25, iou=0.7`. Confirmación oficial. |
| [#15713 Increase recall at price of precision](https://github.com/ultralytics/ultralytics/issues/15713) | **Match exacto del caso.** Glenn Jocher: bajar `conf` durante predict + augmentation + recall-focused dataset + (avanzado) modificar varifocal loss en `loss.py`. Author KHC1234 propone modificar VarifocalLoss con Heaviside function sobre `q-p` (GT IoU - pred IoU). Detalles en thread + [issue #1531 sobre one-hot.scatter](https://github.com/ultralytics/ultralytics/issues/1531). |
| [#5737 set threshold values for C++](https://github.com/ultralytics/ultralytics/issues/5737) | Detalles thresholds en C++ deploy. |
| [#19303 Modify YAML for plastic detection PET/HDPE](https://github.com/ultralytics/ultralytics/issues/19303) | User intentó modificar arquitectura YOLOv8 YAML para detección plástico (2 clases PET/HDPE), accuracy bajó. Glenn Jocher: cambiar solo `nc`, `parse_model` en `ultralytics/nn/tasks.py` guía resto. **Confirma que no modificar arquitectura es decisión correcta para 11 días.** |
| [#13984 YOLOV8 at 0.0 confidence?](https://github.com/ultralytics/ultralytics/issues/13984) | Edge case `conf=0.0`: explicación de comportamiento (todas las cajas se devuelven). Útil para curve sweep. |

**Papers waste/plastic relevantes:**

| Paper | Insight |
|---|---|
| [Nature 2025 *Enhanced YOLOv8 for solid floating waste*](https://www.nature.com/articles/s41598-025-10163-2) | ES-YOLOv8: +5.4% mAP@0.5, +6.1% mAP@0.5:0.95 vs baseline. EMA attention + Shape-IoU loss + multiscale "160-80-40-20". Reto idéntico al nuestro. |
| [Nature 2025 *Real-time intelligent garbage monitoring*](https://www.nature.com/articles/s41598-025-99885-x) | YOLOv8-CBAM mejora 5.5% en waste sorting. Referencia general. |
| [PLOS One 2025 *YOLOv8-SST for small floating debris*](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0340822) | Variante SST para small-object floating. Citable. |
| [MDPI 2024 *Solid Waste Detection Enhanced YOLOv8 Lightweight*](https://www.mdpi.com/2227-7390/12/14/2185) | MSE-AKConv + attention. Edge-friendly. |
| [PMC PMC12823588 *Lightweight YOLO for PET/HDPE classification*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12823588/) | YOLOv7 vs YOLOv8 vs YOLOv9 comparativa para PET/HDPE. mAP 0.867-0.912. |
| [IEEE *Transparent Plastic Bottle Detection YOLOv8*](https://ieeexplore.ieee.org/iel8/10622105/10622106/10622125.pdf) | **Match para "translúcido"**. Método específico para botellas transparentes en sistemas de segregación. |
| [norma.ncirl.ie *UNDERWATER PLASTIC DETECTION YOLO V8 vs V10*](https://norma.ncirl.ie/8542/1/somasekharbogisam.pdf) | Comparativa V8/V10 en plásticos pequeños/medianos. |

**Otros repos plástico:**
- [op1009/plastic-in-river](https://github.com/op1009/plastic-in-river) — YOLOv5 plástico en ríos
- [AniLeo-01/Plastic-In-River-Detection](https://github.com/AniLeo-01/Plastic-In-River-Detection) — YOLOv8 plástico ríos
- [venkataramaraoguttikonda/marine-debris-detection-yolov8](https://github.com/venkataramaraoguttikonda/marine-debris-detection-yolov8) — YOLOv8m marine debris geospatial

**Comunidad:**
- [yolov8.org — How to improve YOLOv8 performance](https://yolov8.org/how-to-improve-yolov8-performance/) — guía de métricas mAP/IoU/conf.

#### B3 — Fine-tuning + active learning + small dataset (19 fuentes)

**Issues Ultralytics canónicos:**

| Issue | Insight |
|---|---|
| [#6201 Best practices small datasets](https://github.com/ultralytics/ultralytics/issues/6201) | Glenn Jocher 10 estrategias: pretrained weights, transfer learning, data augmentation, regularización, transfer from similar classes, few-shot, ensemble, domain-specific pretraining, hyperparameter tuning, synthetic data. Cita: *"few-shot learning techniques are designed to work well with a limited number of training samples"*. |
| [#21122 Fine-tune on false positives only](https://github.com/ultralytics/ultralytics/issues/21122) | User: 7000 imágenes train, 100-200 FP. Glenn Jocher: añadir FP al dataset y re-train (no hay pipeline FP-only nativo). Confirma flujo manual hard-negative del §A5. |
| [#3466 Hard negative mining YoloV8](https://github.com/ultralytics/ultralytics/issues/3466) | Confirmación: NO hay hard-negative mining nativo en YOLOv8. Workaround: añadir background images (sin labels) al dataset. |
| PR #19870 (referenciado por #6201) | **CutMix añadido en Ultralytics 8.3.119** (2026-04). Activable con `cutmix=0.5` en `.train()`. Probar en fine-tune. |

**Papers fine-tune waste:**

| Paper | Insight |
|---|---|
| [arxiv 2508.18799 *Robust Label-Efficient Deep Waste Detection*](https://arxiv.org/pdf/2508.18799) | Baselines actualizados ZeroWaste-f. Fine-tuning supervisado supera OVOD zero-shot. Defiende approach fine-tune sobre Roboflow custom. |
| [Springer *YOLO-VG real-time recyclable waste*](https://link.springer.com/doi/10.1007/s11554-025-01655-5) | Variante YOLO para waste sorting robot. mAP+latency optimized. |
| [Springer *Improved lightweight household waste*](https://link.springer.com/article/10.1007/s11760-025-03827-z) | Lightweight architecture household waste. Edge-friendly. |
| [MDPI Sensors 2024 *MRS-YOLO High-Precision Waste*](https://www.mdpi.com/1424-8220/24/13/4339) | MRS-YOLO: mejora arquitectura waste classification. |
| [MDPI Sustainability 2025 *Real-Time Household Waste*](https://www.mdpi.com/2071-1050/17/5/1902) | Comparativa real-time household waste. |
| [Dergipark 2025 *Real-Time Waste Classification YOLOv8n*](https://dergipark.org.tr/en/pub/jdaia/article/1734221) | YOLOv8n específicamente. Comparable directamente. |

**Repos ejemplares:**
- [starwit/waste-detection](https://github.com/starwit/waste-detection) — YOLOv8 + DVC (data version control), template de fine-tune iterativo. Aplicable para tracking versiones del dataset.
- [knu-on-plus/Recycling_Waste_Detector](https://github.com/knu-on-plus/Recycling_Waste_Detector) — recycling waste YOLOv8.
- [AssadUllahKhan/Object-Detection-For-Waste-Management-Project](https://github.com/AssadUllahKhan/Object-Detection-For-Waste-Management-Project) — TACO dataset YOLOv8 multi-class.
- [MeetShroff/YOLOv8-Based-Waste-Detection-System-for-Recycling-Plants](https://github.com/MeetShroff/YOLOv8-Based-Waste-Detection-System-for-Recycling-Plants) — recycling plants.

---

### Track B — YouTube (MCP local — 3 búsquedas, 26 videos descubiertos)

**Estrategia:** se usaron `youtube_search` para descubrimiento, `youtube_get_video` para metadata + `chapters_available` flag, `youtube_get_chapters` para indexar contenido. Transcripts vienen como YT raw JSON (formato auto-generated subs poco utilizable por línea); los chapters son el índice estructural útil para citar timestamps verificables.

#### Videos top SAHI + small object detection

| Video | Channel | Views | Priority | URL |
|---|---|---|---|---|
| **How to detect small objects with SAHI and YOLO** | Nicolai Nielsen | 22.7k | 0.78 | [Ec-v-DEUUgQ](https://youtu.be/Ec-v-DEUUgQ) |
| **Inference with SAHI Episode 60** | Ultralytics official | 4.8k | 0.79 | [tq3FU_QczxE](https://youtu.be/tq3FU_QczxE) |
| **Small Object Detection with SAHI and YOLO** | Kevin Wood Robotics | 6.5k | 0.77 | [dJeLa5RRcAQ](https://youtu.be/dJeLa5RRcAQ) |
| **Detect small objects with high accuracy** | Pysource | 48.3k | 0.47 | [ViedKFKJOV4](https://youtu.be/ViedKFKJOV4) |
| **YOLOv26 for Small Object Detection + SAHI** | SapkotaAI | 645 | 0.49 | [0Rdk_HyMEIs](https://youtu.be/0Rdk_HyMEIs) |
| **How to Significantly Enhance YOLOv8 with SAHI** | Eran Feit | 1.2k | 0.49 | [e0nsRGwd82s](https://youtu.be/e0nsRGwd82s) |

**Chapters de Nicolai Nielsen `Ec-v-DEUUgQ` (16m36s, el más sustancial):**
- [00:00 Intro](https://youtu.be/Ec-v-DEUUgQ?t=0)
- [00:58 SAHI Github Repo](https://youtu.be/Ec-v-DEUUgQ?t=58)
- [05:01 Model Setup](https://youtu.be/Ec-v-DEUUgQ?t=301)
- [06:01 Standard Inference with YOLOv5](https://youtu.be/Ec-v-DEUUgQ?t=361)
- [06:41 Slicing Code](https://youtu.be/Ec-v-DEUUgQ?t=401)
- [08:49 Results Standard Inference](https://youtu.be/Ec-v-DEUUgQ?t=529)
- [10:48 Results Slicing Inference](https://youtu.be/Ec-v-DEUUgQ?t=648)
- [12:26 Extract the Prediction Results](https://youtu.be/Ec-v-DEUUgQ?t=746)
- [14:43 Batch Prediction](https://youtu.be/Ec-v-DEUUgQ?t=883)

#### Videos top Confidence/Tuning

| Video | Channel | Views | Priority | URL |
|---|---|---|---|---|
| **Insights into Model Evaluation and Fine-Tuning (mAP tips)** | Ultralytics official | 5.2k | 0.72 | [-aYO-6VaDrw](https://youtu.be/-aYO-6VaDrw) |
| Pothole Detection System (CV+ML) | Souvik Chai | 1.3k | 0.98 | [vPvTgB9Xt3Q](https://youtu.be/vPvTgB9Xt3Q) |
| ROC and AUC Clearly Explained | StatQuest | 1.9M | 0.81 | [4jRBRDbJemM](https://youtu.be/4jRBRDbJemM) |

**Chapters de Ultralytics `-aYO-6VaDrw` (6m10s, directo al punto):**
- [00:00 Introduction](https://youtu.be/-aYO-6VaDrw?t=0)
- [00:46 Insights on Model Evaluation and Fine-Tuning](https://youtu.be/-aYO-6VaDrw?t=46)
- [01:01 Model Confidence Score](https://youtu.be/-aYO-6VaDrw?t=61)
- [01:22 Model IOU Score](https://youtu.be/-aYO-6VaDrw?t=82)
- [01:57 mAP (Mean Average Precision)](https://youtu.be/-aYO-6VaDrw?t=117)
- [02:53 Three Ways to Improve mAP](https://youtu.be/-aYO-6VaDrw?t=173)
- [03:44 Evaluating Model Performance Post-Training](https://youtu.be/-aYO-6VaDrw?t=224)

#### Videos top Fine-tuning

| Video | Channel | Views | Priority | URL |
|---|---|---|---|---|
| **How to Train YOLO Object Detection in Colab (YOLO26/v11/v8)** | Edje Electronics | 602.9k | 0.85 | [r0RspiLG260](https://youtu.be/r0RspiLG260) |
| **YOLOv8 Train Custom Dataset** | Roboflow | 508.9k | 0.79 | [wuZtUMEiKWY](https://youtu.be/wuZtUMEiKWY) |
| **Train YoloV8 Custom Dataset** | Programming With Nick | 11.7k | 0.87 | [Bzv58L6xYGc](https://youtu.be/Bzv58L6xYGc) |
| **Entrena Yolov8 Detección con Cualquier Dataset** | Ferneutron (ES) | 29.5k | 0.84 | [XgrPC-I7f4Y](https://youtu.be/XgrPC-I7f4Y) |
| **Auto Label Custom Dataset Roboflow 2 Min** | Nicolai Nielsen | 75.1k | 0.76 | [SDV6Gz0suAk](https://youtu.be/SDV6Gz0suAk) |
| **Fine-Tune YOLO PPE Detection** | Labellerr AI | 1.0k | 0.80 | [Q2-ZH9w3bgo](https://youtu.be/Q2-ZH9w3bgo) |

**Notas YouTube:**
- 3 youtube_search × 200u quota = 400u (búsqueda completa + re-tries con queries simplificadas).
- AAI fallback NO se activó (no necesario, transcripts auto disponibles via cascade).
- Para profundidad alta en próxima ronda: bajar `aai_threshold<1.0` con consentimiento explícito (~$0.12/h audio).

---

## Tabla consolidada — Quick wins ordenados por costo×impacto

| # | Acción | Tiempo | Δ recall esperado | Riesgo | Dependencias | Día |
|---|---|---|---|---|---|---|
| 1 | `conf=0.10`, `iou=0.45` en server FastAPI | 2 min | **+15-25 pts** | FP en escena ruidosa (bajo aquí) | Ninguna | D-0 |
| 2 | Verificar padding 0 vs 114 (script §A8) | 15 min | **+5-15 pts** si mismatch real | Ninguno | Imagen test + engine | D-0 |
| 3 | SAHI con `supervision` slicer (208×208, 0.4 overlap) | 30 min setup | **+20-35 pts** small obj | Latencia ×3-4 (10-14 FPS) | Python ≥3.7 en venv Nano | D-1 |
| 4 | Roboflow Health Check + script bbox sizes | 20 min | N/A (diagnóstico) | Ninguno | Acceso Roboflow | D-1 |
| 5 | TTA offline en `.pt` (medir techo recall) | 30 min | +1-3 pts mAP (medición) | Solo `.pt`, no TRT | GPU + best.pt | D-1 |
| 6 | FiftyOne — visualizar FN del val set | 1 h | N/A (priorización imgs) | Ninguno | `pip install fiftyone` | D-2 |
| 7 | Capturar ~150 imgs setup real C920 + label Roboflow | 4-6 h | **+10-30 pts** dominio shift | Tiempo labeling | Banda+plásticos físicos | D-2 a D-3 |
| 8 | Fine-tune 50 ép RTX 4090 (`freeze=10`, copy_paste, hsv_v) | 2-3 h | **+10-30 pts** | Overfit si <100 imgs | Vast.ai + dataset v2 | D-3 a D-4 |
| 9 | Re-export ONNX + recompilar engine `imgsz=640` | 1-1.5 h | +5-15 pts | FPS 43 → 25-30 | Nano + ONNX nuevo | D-4 |
| 10 | Re-validar end-to-end + recall@conf=0.10 por clase | 2 h | Confirmación | Ninguno | Engine v2 + dataset v2 | D-5 |
| 11 | Iterar augmentation si clase <80% recall | 4 h | Variable | Más epochs | Vast.ai | D-6 a D-7 |
| 12 | Polish demo + dry-run 2026-05-24 | 4 h | N/A | Ninguno | Banda completa | D-9 |

**Compatibilidad TRT engine + cambios:**
- **Cambios D-0 a D-1**: NO requieren recompilar engine. Funcionan sobre `best_fp16.engine` actual.
- **Cambios D-3 a D-4**: requieren engine nuevo (ONNX → trtexec en Nano).

---

## Plan operacional — 11 días al demo (T-11 → T-0)

```
T-11  (HOY 2026-05-15)  D-0: Quick wins inferencia
                        ├─ Bajar conf=0.10, iou=0.45 en server
                        ├─ Script padding 0 vs 114
                        └─ Re-test misma escena 3 plásticos

T-10  (2026-05-16)      D-1: SAHI + diagnóstico
                        ├─ Implementar slicer 208×208 sobre engine
                        ├─ Roboflow Health Check + bbox sizes
                        └─ TTA offline en .pt para techo recall

T-9   (2026-05-17)      D-2: Active learning
                        ├─ Capturar 100-200 imgs setup real
                        └─ Pre-label con best.pt + corrección Roboflow

T-8   (2026-05-18)      D-3: Dataset v2 + fine-tune setup
                        ├─ Generar Roboflow v2 (Letterbox 114, sin aug)
                        └─ Provisionar Vast.ai + transferir notebook

T-7   (2026-05-19)      D-4: Fine-tune
                        ├─ Train 50 ép RTX 4090 (freeze=10, copy_paste 0.5)
                        ├─ Export ONNX + Gates 3/4
                        └─ Subir engine ONNX a HF Hub

T-6   (2026-05-20)      D-5: Engine v2 + integración
                        ├─ Recompilar TRT en Nano (imgsz 640 si SAHI no basta)
                        ├─ Re-test escena 3 plásticos
                        └─ Validar recall@conf=0.10 por clase

T-5   (2026-05-21)      D-6: Iteración si necesario
                        └─ Más augmentation o más imgs reales si <80%

T-4   (2026-05-22)      D-7: Pipeline I²C + servos
                        └─ Integración 3 hilos (captura + infer + actuación)

T-3   (2026-05-23)      D-8: Stress test integral

T-2   (2026-05-24)      D-9: Dry-run demo completo

T-1   (2026-05-25)      D-10: Buffer / fixes finales

T-0   (2026-05-26)      DEMO
```

**Hitos de checkpoint:**
- **D-0 (HOY)**: el cambio `conf/iou` debe restaurar al menos 2 de las 3 detecciones.
- **D-1**: SAHI debe detectar las 3 piezas con conf >0.30.
- **D-5**: engine v2 debe alcanzar recall@conf=0.10 ≥0.85 por clase en val set.

---

## Aportes para el informe IEEE (anexo a esta ronda)

1. **Diagnóstico cuantitativo del padding mismatch Roboflow vs Ultralytics** — brecha documentada en HANDOFF §6, ahora medible. Único registro experimental conocido en literatura para waste detection con Roboflow + Ultralytics + Jetson Nano.
2. **Comparativa empírica `conf=0.25` vs `conf=0.10`** sobre escena real (3 plásticos), reportada como precision-recall por clase.
3. **SAHI sobre TRT FP16 engine en Maxwell sm_53** — ningún paper documenta esta combinación específica (la mayoría usa GPU Volta+ o `.pt` PyTorch). PR #1046 obss/sahi nunca mergeado, repo leon0514 implementa CUDA custom. **Nuestro registro empírico de FPS y recall sería novedoso**.
4. **Fine-tune con `freeze=10` + augmentation reforzado** sobre dataset Roboflow v1-B + 150 imgs reales — ablation por componente (cada hiperparámetro aporta cuántos pts mAP).
5. **Validación de Nature 2025 ES-YOLOv8 hipótesis** sobre nuestro dataset: si ES-YOLOv8 reporta +5.4% mAP@0.5 en floating waste, ¿obtenemos mejora similar al implementar Shape-IoU loss y EMA attention en YOLOv8n? (descartado para 11 días, registrar como future work).

---

## Gotchas detectados en esta ronda

- **`supervision` requiere Python ≥3.7** — Nano JP 4.6.1 viene con 3.6.9 system. Verificar venv del server. Si solo Python 3.6: implementar slicer custom NumPy.
- **TRT engine `imgsz` es fijo** — cambiarlo requiere re-export ONNX + recompilar. `dynamic=True` inestable en TRT 8.x sobre Maxwell.
- **TTA NO funciona en TRT engines** — solo en `.pt`. Útil offline para diagnóstico, no para producción.
- **YOLOv8 NO expone `class_weights` en API pública** — workaround: sobre-muestrear en Roboflow.
- **CutMix añadido en Ultralytics 8.3.119** (PR #19870, 2026-04). Verificar versión instalada en notebook training (`pip show ultralytics`).
- **Hard negative mining NO nativo en YOLOv8** — workaround manual con FiftyOne + duplicar imágenes en train/.
- **Roboflow + Ultralytics double augmentation** — distribución contaminada. Roboflow solo preprocessing, Ultralytics todo el aug.
- **YouTube transcript en formato YT raw JSON** — chapters son el índice utilizable; transcripts auto-subs vienen con metadata de timing pero el texto plano se extrae al armar el `text` field de la cascade.

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco | .md |
|-------|-------|-------------|------|-----|
| 1 | 2026-05-15 | media | Mejoras al modelo de detección de plásticos: tuning práctico (conf/NMS/SAHI/TTA) + augmentation + fine-tune corto + diagnóstico padding | este archivo |

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | Ultralytics Configuration / Predict Settings | https://docs.ultralytics.com/usage/cfg | Doc oficial | 1 |
| 2 | Ultralytics Model Prediction | https://docs.ultralytics.com/modes/predict | Doc oficial | 1 |
| 3 | Ultralytics utils/nms.py reference | https://docs.ultralytics.com/reference/utils/nms | Doc oficial | 1 |
| 4 | Ultralytics data/augment.py reference | https://docs.ultralytics.com/reference/data/augment | Doc oficial | 1 |
| 5 | Ultralytics Data Augmentation guide | https://docs.ultralytics.com/guides/yolo-data-augmentation/ | Doc oficial | 1 |
| 6 | Ultralytics Model Export | https://docs.ultralytics.com/modes/export | Doc oficial | 1 |
| 7 | TensorRT Export for YOLO Models | https://github.com/ultralytics/ultralytics/blob/main/docs/en/integrations/tensorrt.md | Doc oficial | 1 |
| 8 | Roboflow integration Ultralytics | https://docs.ultralytics.com/integrations/roboflow | Doc oficial | 1 |
| 9 | TTA YOLOv5 tutorial | https://docs.ultralytics.com/yolov5/tutorials/test_time_augmentation | Doc oficial | 1 |
| 10 | SAHI Tiled Inference Guide | https://docs.ultralytics.com/guides/sahi-tiled-inference/ | Doc oficial | 1 |
| 11 | Notebook SAHI Ultralytics | https://github.com/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-with-sahi.ipynb | Notebook | 1 |
| 12 | Roboflow blog — Small Object Detection | https://blog.roboflow.com/small-object-detection/ | Blog | 1 |
| 13 | Roboflow Dataset Health Check | https://docs.roboflow.com/datasets/dataset-health-check | Doc oficial | 1 |
| 14 | Roboflow how-to-use SAHI YOLOv8 | https://roboflow.com/how-to-use-sahi/yolov8 | Blog | 1 |
| 15 | Ultralytics blog — Improve mAP small objects | https://www.ultralytics.com/blog/how-to-improve-model-map-on-small-objects-a-quick-guide | Blog | 1 |
| 16 | FiftyOne Fine-tune YOLOv8 tutorial | https://docs.voxel51.com/tutorials/yolov8.html | Doc oficial | 1 |
| 17 | FiftyOne Model Evaluation | https://docs.voxel51.com/getting_started_guides/model_evaluation/index.html | Doc oficial | 1 |
| 18 | Best Practices Fine-Tuning YOLOv8 | https://community.ultralytics.com/t/looking-for-best-practices-for-fine-tuning-yolov8-on-custom-dataset/717 | Foro oficial | 1 |
| 19 | Improve detect small object Ultralytics community | https://community.ultralytics.com/t/improve-detection-of-small-object-in-an-image/1748 | Foro oficial | 1 |
| 20 | Seeed Studio YOLOv8-TRT-Jetson Wiki | https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/ | Tutorial | 1 |
| 21 | obss/sahi PR #1046 yolov8engine.py TRT | https://github.com/obss/sahi/pull/1046 | PR GitHub | 1 |
| 22 | leon0514/trt-sahi-yolo | https://github.com/leon0514/trt-sahi-yolo | Repo GitHub (95★) | 1 |
| 23 | HouYanSong/tensorrtx-yolov8-sahi | https://github.com/HouYanSong/tensorrtx-yolov8-sahi | Repo GitHub | 1 |
| 24 | IrDIE/YOLO8_SAHI | https://github.com/IrDIE/YOLO8_SAHI | Repo GitHub | 1 |
| 25 | Qengineering/YoloV8-TensorRT-Jetson_Nano | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo GitHub | 1 |
| 26 | jetson-ai-lab YOLOv8 tutorial | https://jetson-ai-lab.com/tutorial_ultralytics.html | Tutorial NVIDIA | 1 |
| 27 | Forum NVIDIA Run YOLOv8 Jetson Nano | https://forums.developer.nvidia.com/t/run-yolov8-with-jetson-inference-on-jetson-nano/245322 | Foro | 1 |
| 28 | Issue ultralytics #12988 Docker YOLOv8 Nano | https://github.com/ultralytics/ultralytics/issues/12988 | Issue GitHub | 1 |
| 29 | Issue ultralytics #5315 Conf/NMS thresholds | https://github.com/ultralytics/ultralytics/issues/5315 | Issue GitHub | 1 |
| 30 | Issue ultralytics #15713 Increase recall | https://github.com/ultralytics/ultralytics/issues/15713 | Issue GitHub | 1 |
| 31 | Issue ultralytics #5737 thresholds C++ | https://github.com/ultralytics/ultralytics/issues/5737 | Issue GitHub | 1 |
| 32 | Issue ultralytics #19303 Modify YAML PET/HDPE | https://github.com/ultralytics/ultralytics/issues/19303 | Issue GitHub | 1 |
| 33 | Issue ultralytics #13984 YOLO 0.0 confidence | https://github.com/ultralytics/ultralytics/issues/13984 | Issue GitHub | 1 |
| 34 | Issue ultralytics #6201 Best practices small datasets | https://github.com/ultralytics/ultralytics/issues/6201 | Issue GitHub | 1 |
| 35 | Issue ultralytics #21122 Fine-tune false positives | https://github.com/ultralytics/ultralytics/issues/21122 | Issue GitHub | 1 |
| 36 | Issue ultralytics #3466 Hard negative mining | https://github.com/ultralytics/ultralytics/issues/3466 | Issue GitHub | 1 |
| 37 | Roboflow PR #437 LetterBox padding mismatch | https://github.com/roboflow/inference/pull/437 | PR GitHub | 1 |
| 38 | Nature 2025 Enhanced YOLOv8 floating waste | https://www.nature.com/articles/s41598-025-10163-2 | Paper | 1 |
| 39 | Nature 2025 Real-time intelligent garbage | https://www.nature.com/articles/s41598-025-99885-x | Paper | 1 |
| 40 | PLOS One YOLOv8-SST small floating | https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0340822 | Paper | 1 |
| 41 | MDPI 2024 Solid Waste Enhanced YOLOv8 Lightweight | https://www.mdpi.com/2227-7390/12/14/2185 | Paper | 1 |
| 42 | PMC Lightweight YOLO PET/HDPE | https://pmc.ncbi.nlm.nih.gov/articles/PMC12823588/ | Paper | 1 |
| 43 | IEEE Transparent Plastic Bottle YOLOv8 | https://ieeexplore.ieee.org/iel8/10622105/10622106/10622125.pdf | Paper | 1 |
| 44 | NCIRL Underwater Plastic YOLOv8 vs v10 | https://norma.ncirl.ie/8542/1/somasekharbogisam.pdf | Tesis | 1 |
| 45 | arxiv 2508.18799 Robust Label-Efficient Waste | https://arxiv.org/pdf/2508.18799 | Paper | 1 |
| 46 | Springer YOLO-VG real-time recyclable | https://link.springer.com/doi/10.1007/s11554-025-01655-5 | Paper | 1 |
| 47 | Springer Improved lightweight household waste | https://link.springer.com/article/10.1007/s11760-025-03827-z | Paper | 1 |
| 48 | MDPI MRS-YOLO High-Precision Waste | https://www.mdpi.com/1424-8220/24/13/4339 | Paper | 1 |
| 49 | MDPI Real-Time Household Waste Sustainable | https://www.mdpi.com/2071-1050/17/5/1902 | Paper | 1 |
| 50 | Dergipark Real-Time Waste YOLOv8n | https://dergipark.org.tr/en/pub/jdaia/article/1734221 | Paper | 1 |
| 51 | starwit/waste-detection (YOLOv8+DVC) | https://github.com/starwit/waste-detection | Repo GitHub | 1 |
| 52 | knu-on-plus/Recycling_Waste_Detector | https://github.com/knu-on-plus/Recycling_Waste_Detector | Repo GitHub | 1 |
| 53 | op1009/plastic-in-river | https://github.com/op1009/plastic-in-river | Repo GitHub | 1 |
| 54 | AniLeo-01/Plastic-In-River-Detection | https://github.com/AniLeo-01/Plastic-In-River-Detection | Repo GitHub | 1 |
| 55 | venkataramaraoguttikonda/marine-debris-yolov8 | https://github.com/venkataramaraoguttikonda/marine-debris-detection-yolov8 | Repo GitHub | 1 |
| 56 | yolov8.org how to improve performance | https://yolov8.org/how-to-improve-yolov8-performance/ | Blog | 1 |
| 57 | Soft-NMS implementation PyTorch | https://github.com/MrParosk/soft_nms | Repo GitHub | 1 |
| 58 | YouTube — Nicolai Nielsen SAHI | https://youtu.be/Ec-v-DEUUgQ | Video (16m36s) | 1 |
| 59 | YouTube — Ultralytics SAHI Episode 60 | https://youtu.be/tq3FU_QczxE | Video (7m53s) | 1 |
| 60 | YouTube — Kevin Wood Robotics SAHI | https://youtu.be/dJeLa5RRcAQ | Video (7m12s) | 1 |
| 61 | YouTube — Pysource small objects | https://youtu.be/ViedKFKJOV4 | Video (21m46s) | 1 |
| 62 | YouTube — SapkotaAI YOLOv26 + SAHI | https://youtu.be/0Rdk_HyMEIs | Video (15m) | 1 |
| 63 | YouTube — Ultralytics Model Evaluation mAP tips | https://youtu.be/-aYO-6VaDrw | Video (6m10s) | 1 |
| 64 | YouTube — Edje Electronics Train YOLO Colab | https://youtu.be/r0RspiLG260 | Video (21m26s) | 1 |
| 65 | YouTube — Roboflow YOLOv8 Custom Dataset | https://youtu.be/wuZtUMEiKWY | Video (20m31s) | 1 |
| 66 | YouTube — Ferneutron Entrena Yolov8 (ES) | https://youtu.be/XgrPC-I7f4Y | Video (36m30s) | 1 |
| 67 | YouTube — Nicolai Auto Label Roboflow 2 min | https://youtu.be/SDV6Gz0suAk | Video (11m22s) | 1 |
| 68 | YouTube — Labellerr PPE Detection | https://youtu.be/Q2-ZH9w3bgo | Video (17m27s) | 1 |
| 69 | **Akyon, Altinuc, Temizel 2022 — SAHI paper (IEEE ICIP 2022 oral)** | https://arxiv.org/abs/2202.06934 | Paper foundational | 1 |
| 70 | **Tariq, Javed 2025 — Small Object Detection YOLO Performance Analysis** | https://arxiv.org/abs/2504.09900 | Paper | 1 |
| 71 | **Khalili, Smyth (Columbia) 2024 — SOD-YOLOv8 (P2 head + EMA + PIoU)** | https://arxiv.org/abs/2408.04786 | Paper IEEE ICIP 2024 | 1 |
| 72 | Kuang, Bhandari, Gao (RPI) 2024 — Optimizing Waste Management Garbage Classification | https://arxiv.org/abs/2410.09975 | Paper | 1 |
| 73 | Alqahtani et al. (Monash/Melbourne) 2024 — Benchmarking Edge Devices | https://arxiv.org/abs/2409.16808 | Paper | 1 |
| 74 | obss/sahi PR #1336 — torch-free + batch inference TRT | https://github.com/obss/sahi/pull/1336 | PR GitHub (draft) | 1 |
| 75 | obss/sahi PR #1042 — mmdet TRT support | https://github.com/obss/sahi/pull/1042 | PR GitHub | 1 |
| 76 | THU-MIG/yolov10 issue #75 — fmod/Mod operator no soportado TRT 8.2 | https://github.com/THU-MIG/yolov10/issues/75 | Issue GitHub | 1 |
| 77 | THU-MIG/yolov10 issue #129 — TRT export workaround | https://github.com/THU-MIG/yolov10/issues/129 | Issue GitHub | 1 |
| 78 | obss/sahi repo principal (5274★) | https://github.com/obss/sahi | Repo GitHub | 1 |
