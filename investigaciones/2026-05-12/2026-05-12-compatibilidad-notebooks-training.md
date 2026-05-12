# Compatibilidad de notebooks de training para Jetson Nano (JetPack 4.6.1)

**Dominio:** Compatibilidad stack training Colab/Kaggle vs runtime Jetson Nano JetPack 4.6.1
**Fecha:** 2026-05-12
**Profundidad:** Alta
**Estado:** Ronda 1 — completa

---

## Resumen ejecutivo y veredicto vinculante

Esta investigación responde a dos fallos reproducibles del 2026-05-12 al ejecutar los notebooks de training en sus plataformas objetivo:

1. **Track A en Colab:** `pip install tensorflow==2.13.1 tf-models-official==2.13.1` falla porque Colab default es Python 3.12 desde agosto 2025 y TF 2.13 no tiene wheels para esa versión de Python.
2. **Track B en Kaggle:** `FileNotFoundError: Dataset '/kaggle/working/waste-3class-1/data.yaml' not found` porque el SDK `roboflow-python` descarga a `CWD/<project-slug>-<version>/` ignorando o malinterpretando el argumento `location`.

### Veredicto Track A (decisión vinculante)

**Pivote oficial: TF 2.15 + Python 3.10 (vía `condacolab`) + TF OD API legacy + PTQ post-entrenamiento + SSD MobileNet v2 320×320 plain (sin FPN).**

Razones, cada una con evidencia:

- **TF 2.16+ rompe TF OD API irrecuperablemente.** Issue tensorflow/models #13599 (PR sin merge desde 2026-01-11) documenta verbatim: *"Currently, training fails with a cryptic ImportError when using TensorFlow >= 2.16 due to removal of Estimator support. TensorFlow ≤ 2.15: Verified that no early failure occurs and existing behavior is unchanged."* Issue #13575 (2025-06-26) confirma el síntoma: *"the last support for the estimator from tensorflow.compat.v1 was in tensorflow==2.15."*
- **QAT vía `graph_rewriter` en pipeline.config está silently broken en TF2 desde 2021.** Issue #9835 verbatim de `tensorbuffer`: *"The graph_rewriter is not handled in tf2."* Confirmado sin fix en 2025-04-24 por `Petros626`: *"The repo is maintained by people, not tensorflow team anymore."* → No podemos hacer QAT real con la TFOD API.
- **MediaPipe Model Maker (alternativa de QAT real) impone restricciones severas.** `requirements.txt` del repo verbatim: `tensorflow>=2.10,<2.16`, `tf-models-official>=2.13.2,<2.16.0`, `tensorflow-model-optimization<0.8.0`. `setup.py` declara solo Python 3.8/3.9/3.10. Y el modelo exportado **NO incluye `TFLite_Detection_PostProcess`** (confirmado en `model.py`: `tflite_post_processing=configs.common.TFLitePostProcessingConfig(omit_nms=True)`) → requiere implementar decoder + NMS custom en el Nano, lo cual añade riesgo de bug y mantenimiento al runtime que ya está validado para SSD clásico.
- **TFMOT no soporta SSD con FPN out-of-the-box.** No existe receta oficial publicada para aplicar `tfmot.quantization.keras.quantize_model()` a SSD MV2 FPNLite en TF 2.15+. El path recomendado por TF-Vision es `official/projects/qat/vision` con RetinaNet, no SSD.
- **PTQ vs QAT en CPU TFLite ARM Cortex-A57:** Jacob et al. CVPR 2018 (arXiv:1712.05877) establece que QAT mantiene caída < 1.5 pp mAP vs FP32. Karimov et al. 2025 (arXiv:2508.19600) mide PTQ INT8 con caída 3-7 pp mAP50-95 en modelos pequeños. Asumiendo caída de 5 pp para nuestro caso y partiendo de mAP COCO 20.2 del SSD MV2 plain INT8: esperamos ~15-16 pp mAP en COCO, pero en nuestro dataset de 3 clases (waste-3class) con clases visualmente distintas y dataset relativamente balanceado, la caída debería ser menor en términos relativos.
- **FPS en CPU TFLite XNNPACK Cortex-A57:** Estimación a partir de NobuoTsukamoto/benchmarks (Raspberry Pi 4 Cortex-A72, INT8 4 hilos: 78 ms para EfficientDet-Lite0, A57 ~20% más lento): SSD MV2 plain 320 INT8 ≈ 55-70 ms (14-18 FPS). FPNLite ≈ 90-110 ms (9-11 FPS). Para garantizar ≥ 10 FPS con margen, **plain** es la elección segura.

### Veredicto Track B (decisión vinculante)

**Mantener YOLOv8n 416×416 + Ultralytics ≥ 8.4.46 + ONNX opset 11 explícito + onnxsim ≥ 0.6.2 + FP16 TRT engine construido en Nano + NMS en CPU NumPy.**

Razones, cada una con evidencia:

- **Kaggle/Colab traen PyTorch 2.9-2.10 con CUDA 12 host.** No afecta el `.onnx` exportado con `device='cpu'`. El CUDA del host es irrelevante para el archivo `.onnx` resultante (PyTorch exporta por trazado en CPU). El ONNX es portátil.
- **ONNX opset 11 es soportado por TRT 8.2.1.** `onnx-tensorrt/docs/operators.md?ref=release/8.2-GA` verbatim: *"TensorRT 8.2 supports operators up to Opset 13."* Todos los ops del head de YOLOv8n están soportados en opset 11.
- **Ultralytics 8.4.x default opset es 20 con torch 2.9+.** Hay que forzar `opset=11` explícitamente: `model.export(format='onnx', opset=11, simplify=True, dynamic=False)`.
- **`onnxsim==0.4.36` no compila en Python 3.12** (issue #334 daquexian/onnx-simplifier). Pin a `>=0.6.2,<0.7` (wheels manylinux para Py 3.12 desde 0.5.0 feb 2026).
- **`EfficientNMS_TRT` plugin no funciona estable en Maxwell** (issue NVIDIA/TensorRT #1538) → NMS en CPU NumPy con `cv2.dnn.NMSBoxes` o `torchvision.ops.nms` en el Nano.
- **Roboflow SDK bug ya tiene fix aplicado** en cell-10 (cascada de búsqueda + `os.chdir(WORK_DIR)` + heurística de contenido `glass+paper+plastic`).

---

## Matriz de compatibilidad mayo 2026

### Plataformas

| Plataforma | Release | Python | TF default | PyTorch default | CUDA host |
|------------|---------|--------|-----------|-----------------|-----------|
| Google Colab | 2026.04 | 3.12.13 | 2.19.0 | 2.10.0 | 12.x |
| Google Colab | 2026.01 | 3.12.12 | 2.19.0 (pre) | 2.9.0 | 12.x |
| Kaggle GPU | imagen v168 (mar 2026) | 3.10+ | 2.19.x | 2.9-2.10.x | 12.x |

Fuente verbatim FAQ Colab: *"2026.04: Ubuntu 22.04.5 LTS, Python 3.12.13, numpy 2.0.2, TensorFlow 2.19.0"* (https://research.google.com/colaboratory/runtime-version-faq.html).

### Wheels TF × Python 3.12 (Linux x86_64, PyPI)

| TF version | Wheel cp312 | Notas |
|------------|-------------|-------|
| 2.13.x | ❌ NO | Primer fallo del usuario |
| 2.15.x | ❌ NO | Issue tensorflow/tensorflow #62003 |
| 2.16.1 | ✅ primera versión | Blog TF marzo 2024 |
| 2.16-2.20 | ✅ | TF 2.16+ rompe `tf.estimator` → OD API roto |
| 2.21.0 (mar 2026) | ✅ | Sin Python 3.9, mantiene 3.10-3.12 |

**Cuello de botella crítico:** la TF OD API legacy (`research/object_detection/model_main_tf2.py`) requiere TF ≤ 2.15 (limit duro), pero TF 2.15 no tiene wheel cp312 → para Track A necesitamos **bajar Colab a Python 3.10 + TF 2.15**, lo cual se hace con `condacolab`.

### Plan de transición de Python en Colab

```python
# Cell-1 Track A — antes de cualquier deps:
!pip install -q condacolab
import condacolab
condacolab.install()  # Reinicia kernel automáticamente; Python pasa a 3.10
# Tras reinicio, el resto del notebook corre en Py 3.10 + conda env
```

`condacolab` (Jaime Rodríguez-Guerra) instala Mambaforge en Colab y bajba Python a 3.10 con env conda dedicado. Es la forma estándar de pinear Python ≤ 3.10 en Colab moderno.

### Runtime Jetson Nano (JetPack 4.6.1, inmutable)

| Componente | Versión |
|------------|---------|
| OS | Ubuntu 18.04 LTS (L4T R32.7.1) |
| Python | 3.6.9 |
| TF | 2.5.0+nv21.8 wheel NVIDIA |
| TFLite custom op | `TFLite_Detection_PostProcess` incluido |
| TRT | 8.2.1 |
| CUDA | 10.2 |
| cuDNN | 8.2.1 |
| OpenCV | 4.1.1 |
| TFLite Schema | v3 (estable desde TF 2.x) |
| Hardware GPU | Maxwell 128 CUDA cores, **sin Tensor Cores** |

`TFLITE_SCHEMA_VERSION = 3` en `tensorflow/lite/version.h` HEAD (TF 2.21 master), idéntico a TF 2.5 → el schema flatbuffer es estable, la compatibilidad depende de la versión de operador individual (op versioning).

---

## Decisión QAT Track A — análisis comparativo

| Opción | QAT real | TF | Python | Wheel disponible | FPS estimado CPU TFLite Nano | Riesgo | Decisión |
|--------|----------|----|----|-------------------|-------------------------------|--------|----------|
| 1. TFOD API + graph_rewriter | ❌ placebo | 1.15 | 3.7 max | No moderno | N/A | Issue #9835 verbatim | ❌ Descartada |
| 2. MediaPipe Model Maker MOBILENET_V2_I320 | ✅ real preintegrado | 2.15 | 3.10 | Sí | 10-11 FPS estimado (margen ajustado) | NMS no embebido → decoder custom en Nano; caveat accuracy loss documentado | ⚠️ Plan B contingencia |
| 3. TF 2.15 + TFOD API + PTQ post-training (SSD MV2 plain) | ❌ no QAT (PTQ) | 2.15 | 3.10 (vía condacolab) | Sí | 14-18 FPS estimado | Caída 3-7 pp mAP por PTQ; mitigable con representative dataset bien calibrado | ✅ **OPCIÓN ELEGIDA** |
| 4. TFMOT Keras-style QAT | ✅ real | 2.15+ | 3.10+ | Sí | N/A | No soporta SSD con FPN; sin receta oficial | ❌ Descartada |

**Justificación de la elección (Opción 3):**

- **Compatibilidad runtime:** `TFLite_Detection_PostProcess` se incluye en el export → drop-in en TF 2.5.0+nv21.8 del Nano. No requiere desarrollo adicional de decoder.
- **FPS holgado:** 14-18 FPS estimado vs threshold 10 FPS → margen del 40-80%.
- **mAP aceptable para 3 clases visualmente distintas:** dataset waste-3class tiene clases muy diferenciadas (vidrio brillante, papel mate, plástico variado) → la caída de PTQ debería ser pequeña en términos absolutos.
- **Stack maduro:** TF 2.15 + TFOD API + Python 3.10 es la última combinación estable documentada y con tutoriales completos.
- **Plan B factible (Opción 2):** si el mAP de PTQ no alcanza el threshold de calidad, pivotar a MediaPipe Model Maker es viable con el mismo entorno Python 3.10 y requiere ~1-2 días de trabajo adicional para el decoder en Nano.

---

## Deps pinneadas finales

### Track A — `train_track_a_ssd.ipynb` cell-8

```python
# CELDA 8 — Dependencias Track A (TF 2.15 + Python 3.10 vía condacolab)
# Stack runtime target: Jetson Nano JetPack 4.6.1
#   Python 3.6.9, TF 2.5.0+nv21.8, TFLite_Detection_PostProcess incluido,
#   CPU Cortex-A57 + GPU Maxwell sin Tensor Cores.
# Stack training: Colab Py 3.10 (condacolab) + TF 2.15.0 + tf-models-official 2.15.0.

# Paso 1: bajar Python a 3.10 (solo Colab; saltar en Kaggle si Py ya es 3.10+)
import sys
if sys.version_info >= (3, 11):
    !pip install -q condacolab
    import condacolab
    condacolab.install()  # ⚠️ Reinicia kernel — ejecutar celdas siguientes después

# Paso 2 (tras reinicio): pin de versiones del stack training
%pip install -q --upgrade pip
%pip install -q "tensorflow==2.15.0"
%pip install -q "tf-models-official==2.15.0"
%pip install -q "tensorflow-model-optimization>=0.7.5,<0.8.0"
%pip install -q "numpy==1.26.4" "protobuf==3.20.3"
%pip install -q "opencv-python-headless==4.10.0.84"
%pip install -q "pycocotools==2.0.7" "lvis==0.5.3"
%pip install -q "tensorflow-addons==0.23.0" "tensorflow-text==2.15.0"
%pip install -q "Pillow==10.4.0"

# Paso 3: clonar TF Object Detection API en versión congelada (snapshot pre-Pillow12)
!git clone --depth 1 --branch v2.16.0 https://github.com/tensorflow/models /content/models || \
    git -C /content/models pull --ff-only
!cd /content/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install -q --use-deprecated=legacy-resolver .

# Paso 4: validar instalación
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
print(f"TF version: {tf.__version__}")  # esperar 2.15.0
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
print(f"OD API import: OK")

# Paso 5: roboflow client (mismo bug location → fix está en cell-12)
%pip install -q "roboflow>=1.3.6,<1.4"
```

### Track B — `train_track_b_yolov8.ipynb` cell-8

```python
# CELDA 8 — Dependencias Track B (PyTorch + Ultralytics modernos)
# Stack runtime target: Jetson Nano JetPack 4.6.1
#   TRT 8.2.1 (opset 13 max), CUDA 10.2, GPU Maxwell sin Tensor Cores.
# Stack training: Kaggle/Colab Py 3.10+/3.12 + PyTorch 2.9-2.10 (preinstalado).

%pip install -q --upgrade pip

# Ultralytics moderno (incluye fix INT8 calibration PR #24028 desde 8.4.31)
%pip install -q "ultralytics>=8.4.46,<8.5"

# ONNX stack: opset 11 → ir_version compatible con onnxsim 0.6.x
%pip install -q "onnx>=1.16,<1.18" "onnxruntime>=1.18,<1.21"
%pip install -q "onnxsim>=0.6.2,<0.7"  # NO usar 0.4.x: no compila en Py 3.12

# Roboflow client (con fix cell-10 ya aplicado para bug location)
%pip install -q "roboflow>=1.3.6,<1.4"

# Variables de entorno ANTES de instanciar Roboflow() (evita bug location):
import os
os.environ["DATASET_DIRECTORY"] = "/kaggle/working/datasets" if os.path.exists("/kaggle/working") else "/content/datasets"

# Validar instalación
import torch, ultralytics, onnx, onnxsim, roboflow
print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"onnxsim: {onnxsim.__version__}")
print(f"Roboflow: {roboflow.__version__}")
print(f"DATASET_DIRECTORY: {os.environ['DATASET_DIRECTORY']}")
```

### Export ONNX (Track B) — patrón canónico

```python
# CELDA EXPORT Track B — forzar opset 11 (default es 20 con torch 2.9+)
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')

# Export ONNX con opset 11 explícito (TRT 8.2.1 max opset 13; usamos 11 conservador)
model.export(
    format='onnx',
    opset=11,           # CRÍTICO: default es 20, TRT 8.2 no garantiza ops opset 14+
    simplify=True,      # simplifica grafo con onnxsim 0.6.x
    dynamic=False,      # shapes fijas para TRT engine determinístico
    imgsz=416,          # dimensión cuadrada (PR #24028 fix INT8 calib non-square)
    device='cpu',       # CUDA host irrelevante para .onnx
    half=False,         # FP16 se aplica en trtexec, no en ONNX
    int8=False,         # no INT8 → FP16 en Maxwell por DP4A
    nms=False,          # NMS en CPU NumPy en Nano (EfficientNMS_TRT roto Maxwell)
)
# Output: runs/detect/train/weights/best.onnx
# Verificación post-export:
import onnx
m = onnx.load('runs/detect/train/weights/best.onnx')
assert m.opset_import[0].version == 11, f"Opset esperado 11, obtenido {m.opset_import[0].version}"
print(f"✅ Export ONNX exitoso. Opset: {m.opset_import[0].version}, IR: {m.ir_version}")
```

---

## Hallazgos detallados por sub-track

### A. Stack TF training vs runtime Nano (agent `research-code`)

**A1. Wheels TF Python 3.12 — confirmados desde TF 2.16.1:** Blog oficial TF marzo 2024 anuncia *"support for Python 3.12"* en 2.16.1. TF 2.13/2.14/2.15 no tienen wheels cp312 (issue tensorflow/tensorflow #62003 verbatim del usuario: `ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0 (from versions: 2.16.0rc0, 2.16.1)`).

**A2. TFOD API legacy — límite duro TF ≤ 2.15:**

- Commit más reciente en `research/object_detection`: 2026-04-29 (Pillow 12 support). Mantenimiento mínimo, no desarrollo de features.
- Issue #13575 (open 2025-06-26): *"the last support for the estimator from tensorflow.compat.v1 was in tensorflow==2.15."*
- Issue #13599 (PR open 2026-01-11, sin merge): *"Currently, training fails with a cryptic ImportError when using TensorFlow >= 2.16 due to removal of Estimator support."*
- Issue #9835 (open 2021-03-25): *"The graph_rewriter is not handled in tf2."* → QAT placebo en TF2.
- Issue #11168 (open 2024-02-23): protos no compilados al instalar desde PyPI directamente — workaround: clonar repo + `protoc + setup.py`.
- DeepWiki verbatim: *"The research/object_detection directory is no longer actively maintained for compatibility with new external dependencies. Users are encouraged to consider TF-Vision or scenic."*

**A3. TFLite forward-compatibility schema:** `TFLITE_SCHEMA_VERSION = 3` constante entre TF 2.5 y TF 2.21. La compatibilidad real depende de versiones de operador individuales (op versioning), no del schema container. Para ops INT8 maduras (CONV_2D, DEPTHWISE_CONV_2D, ADD, MUL, RESHAPE) introducidas en TF ≤ 2.5, el riesgo de forward-compat es **bajo**. Validar empíricamente con `flatbuffer_utils.py` post-export pero antes de transferir al Nano.

**A4. MediaPipe Model Maker — restricciones:**

- `requirements.txt` verbatim: `tensorflow>=2.10,<2.16`, `tf-models-official>=2.13.2,<2.16.0`, `tensorflow-model-optimization<0.8.0`.
- `setup.py` Python classifiers: 3.8, 3.9, 3.10 (no 3.11/3.12).
- `MOBILENET_V2_I320` usa checkpoint `gs://tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i320_ckpt.tar.gz` — QAT preintegrado.
- **NMS omitido en TFLite export:** `tflite_post_processing=configs.common.TFLitePostProcessingConfig(omit_nms=True)` → modelo `.tflite` no contiene `TFLite_Detection_PostProcess`. Requiere decoder custom (decoding de anchors + NMS) en Nano.
- Dataset solo COCO/PASCAL VOC. Roboflow exporta ambos formatos → conversión trivial desde TFRecord.

**A5. TFMOT — sin soporte para SSD FPNLite:** `tfmot.quantization.keras.quantize_model()` opera por capa Keras y no entiende arquitecturas compuestas como `WeightSharedConvolutionalBoxPredictor` del SSD. No hay receta oficial.

### B. Benchmarks INT8 y comparativa de modelos (agent `research-academic`)

**B1. SSD MV2 plain vs FPNLite en Jetson Nano:**

| Modelo | Resolución | Precisión | Latencia (Nano TRT GPU) | mAP COCO | mAP@50 |
|--------|-----------|-----------|-------------------------|----------|--------|
| SSD MV2 plain | 320×320 | FP32 | 26.96 ms | 20.18 | 34.74 |
| SSD MV2 plain | 320×320 | FP16 | 24.31 ms | 20.18 | 34.74 |
| SSD MV2 FPNLite | 320×320 | FP32 | 39.54 ms | 21.97 | 36.95 |
| SSD MV2 FPNLite | 320×320 | FP16 | 37.99 ms | 21.97 | 36.95 |

Fuente: `NobuoTsukamoto/benchmarks` (Jetson Nano B01 JetPack 4.6.1 + TRT). Gap mAP@50: **+2.21 pp para FPNLite**. Cost: **47% más lento**.

**B2. QAT vs PTQ — gap mAP esperado:**

- Jacob et al. 2018 (arXiv:1712.05877): *"MobileNet SSD INT8 con QAT achieve accuracy within 1.5% of floating-point."* → QAT mantiene caída < 1.5 pp.
- Karimov et al. 2025 (arXiv:2508.19600): *"Static INT8 TensorRT engines offer substantial speedups (~1.5-3.3x) with a moderate accuracy drop (~3-7% mAP50-95) on clean data."* → PTQ INT8 cae 3-7 pp.
- Diferencia neta esperada **PTQ - QAT ≈ 2-5 pp** en favor de QAT.

**B3. Alternativas para CPU TFLite ARM Cortex-A57 (umbral ≥ 10 FPS):**

| Modelo | Resolución | INT8 | FPS estimado Nano CPU | mAP@50 | QAT real | Verdict |
|--------|-----------|------|----------------------|--------|----------|---------|
| SSD MV2 plain | 320×320 | ✅ | 14-18 FPS | 34.7 | ✅ vía TFMOT/Model Garden ckpt | ✅ candidato principal |
| SSD MV2 FPNLite | 320×320 | ✅ | 8-10 FPS | 36.9 | ⚠️ checkpoint QAT existe pero sin receta TFMOT | ⚠️ margen ajustado |
| EfficientDet-Lite0 | 320×320 | ✅ | 10-11 FPS | 39-41 | ❌ solo PTQ | ⚠️ margen ajustado, sin QAT nativo |
| YOLOv8n TFLite | 320×320 | ✅ | 1-2 FPS | 52 | ❌ | ❌ no viable CPU |
| NanoDet-plus-m | 320×320 | ⚠️ | gap evidencia | 52 | ❌ TFLite support no documentado | ❌ riesgo alto |

Confirmación explícita Zagitov et al. 2024 (doi:10.18287/2412-6179-CO-1343): *"We recommend EfficientDet Lite 320×320 quantized or SSD Mobilenet V2 320×320 for tasks with over 10 FPS."*

**B4. Caso dominio específico (no COCO):** Trisuwita et al. 2024 reportan en detección de cascos que SSD MV2 plain (80.12% mAP) supera al FPNLite (71.59% mAP) — gap invertido respecto a COCO. **Implicación para waste-3class:** en datasets pequeños y balanceados con clases visualmente distintas, FPN puede no aportar; plain es competitivo o superior.

### C. APIs de plataforma — secrets, mount, quotas (agent `research-web`)

**C1. Patrón canónico Colab Secrets (sin breaking changes):**

```python
from google.colab import userdata
try:
    api_key = userdata.get('ROBOFLOW_API_KEY')
except userdata.SecretNotFoundError:
    raise RuntimeError("Configura el secret ROBOFLOW_API_KEY en Colab UI")
except userdata.NotebookAccessError:
    raise RuntimeError("Habilita 'Notebook access' para este secret")
```

**C2. Patrón canónico Kaggle Secrets (sin breaking changes):**

```python
from kaggle_secrets import UserSecretsClient
secret = UserSecretsClient().get_secret('ROBOFLOW_API_KEY')
```

Recordar: en Kaggle, los secrets requieren toggle "Attach Secret" en la UI del notebook editor; al hacer fork, los valores NO se transfieren.

**C3. Quotas:**

| Plataforma | Sesión máx | Cuota semanal | RAM | GPU típica |
|------------|-----------|---------------|-----|-----------|
| Colab Free | 12 h | adaptativa, no publicada | ~12 GB | T4 / K80 compartida |
| Colab Pro | 12 h | mejor disponibilidad | High-RAM | T4/V100 prioritaria |
| Kaggle GPU | 12 h | **30 h/sem** | 29 GB | T4×2 o P100 |

→ Diseñar epochs con checkpoint cada 2-3 epochs, training cap < 11 h por sesión para evitar disconnect en hora 12.

### D. Roboflow SDK bug `location` (discoveries 5 + agent `research-web`)

**D1. Causa raíz confirmada en código fuente** (`roboflow/core/version.py`):

```python
def download(self, model_format=None, location=None, overwrite: bool = False):
    if location is None:
        location = self.__get_download_location()  # ← devuelve path RELATIVO al CWD
```

**D2. Bug residual cuando `location` SÍ se pasa:** el `data.yaml` generado contiene paths relativos que asumen ubicación distinta — issue #240 verbatim del síntoma:

> *"RuntimeError: Dataset 'TennisBallTracker-9/data.yaml' error [...] missing path '/.../datasets/TennisBallTracker-9/TennisBallTracker-9/valid/images'"*

**D3. Issue #88 (open desde 2022-12-21, sin fix):** clases fantasma en `data.yaml` cuando se eliminaron clases sin regenerar versión. Workaround: validación manual post-download de `nc:` y `names:`.

**D4. Fix aplicado en ambos notebooks** (cell-10 Track B, cell-12 Track A):

1. `os.chdir(WORK_DIR)` antes de `version.download()` para forzar CWD conocido.
2. Búsqueda en cascada del archivo target (`data.yaml` para B, `_label_map.pbtxt` para A): primero en `ds.location` retornado por SDK, luego en `WORK_DIR`, fallback global con heurística "glass + paper + plastic" en contenido.
3. Si se encuentra fuera de `WORK_DIR`, copia/mueve a la ubicación esperada antes de continuar.
4. Set `os.environ["DATASET_DIRECTORY"]` ANTES de instanciar `Roboflow()`.

### E. ONNX opset y onnxsim (discovery 4 + agent `research-web`)

**E1. Ultralytics 8.4.x `best_onnx_opset`:** default 20 con torch 2.9+, hay que forzar `opset=11` para TRT 8.2.

**E2. Ops del head YOLOv8n en opset 11 soportados por TRT 8.2:** `Concat`, `Reshape`, `Transpose`, `MatMul`, `Sigmoid`, `Sub`, `Mul`, `Add`, `Upsample`, `Resize`. **`NonMaxSuppression` está marcado EXPERIMENTAL** → exportar sin NMS embebido y hacer NMS en CPU NumPy en Nano (confirma decisión previa del proyecto).

**E3. onnxsim pin recomendado:** `>=0.6.2,<0.7`. Versiones 0.4.x no compilan en Python 3.12 (issue #334). Breaking changes desde 0.5.0:
- `--dynamic-input-shape` removido
- `--input-shape` → `--overwrite-input-shape`
- `--enable-fuse-bn` removido (default)

---

## Gates de validación pre-export y post-export

### Track A

**Pre-train:**

- [ ] `_label_map.pbtxt` contiene exactamente 3 clases: glass, paper, plastic, con IDs 1, 2, 3.
- [ ] `pipeline.config` apunta a `train.record` y `val.record` existentes con `num_classes: 3`.
- [ ] `fine_tune_checkpoint_type: "detection"` y `fine_tune_checkpoint` apunta al checkpoint pretrained de `ssd_mobilenet_v2_320x320_coco17_tpu-8`.

**Post-train (antes de export):**

- [ ] `total_loss < 2.0` empírico en `train_metrics`.
- [ ] `DetectionBoxes_Precision/mAP@.50IOU > 0.7` en eval split.
- [ ] Checkpoint reciente disponible en `model_dir`.

**Pre-export TFLite:**

```python
# Validar SavedModel signatures
import tensorflow as tf
sm = tf.saved_model.load(EXPORTED_SAVED_MODEL_DIR)
sig = sm.signatures['serving_default']
assert 'image_tensor' in sig.structured_input_signature[1] or list(sig.structured_input_signature[1].keys())[0]
print(f"Inputs: {sig.structured_input_signature}")
print(f"Outputs: {sig.structured_outputs}")
```

**Post-export TFLite:**

```python
# Cargar TFLite y correr inferencia en imagen test
import tensorflow as tf
import numpy as np
from PIL import Image

interp = tf.lite.Interpreter(model_path='model_int8.tflite')
interp.allocate_tensors()
in_d = interp.get_input_details()
out_d = interp.get_output_details()
print(f"Input dtype: {in_d[0]['dtype']}, shape: {in_d[0]['shape']}")
# Si INT8: in_d[0]['dtype'] debe ser np.uint8, no np.float32

# Validar que TFLite_Detection_PostProcess está embebido
# Outputs esperados: 4 tensores (boxes, classes, scores, num_detections)
assert len(out_d) == 4, f"PTQ TFLite con NMS embebido debe tener 4 outputs, got {len(out_d)}"

# Inferencia sobre imagen test
img = np.array(Image.open('test_image.jpg').resize((320, 320)))
in_data = np.expand_dims(img, 0).astype(in_d[0]['dtype'])
interp.set_tensor(in_d[0]['index'], in_data)
interp.invoke()
boxes = interp.get_tensor(out_d[0]['index'])
print(f"Boxes shape: {boxes.shape}")
```

**Pre-deploy al Nano:**

- [ ] Inspeccionar `op_version` con `flatbuffer_utils.py` (TF tools) — verificar que todas las ops están en versión ≤ 5 (introducidas en TF ≤ 2.5).
- [ ] Validar que `TFLite_Detection_PostProcess` aparece en el listado de custom ops del modelo.

### Track B

**Pre-train:**

- [ ] `data.yaml` post-Roboflow contiene `nc: 3` y `names: [glass, paper, plastic]`.
- [ ] Splits train/val/test: cada uno con ≥ 10 imágenes por clase.

**Post-train:**

- [ ] `mAP50 > 0.7` en val split tras 50 epochs.
- [ ] `model.fuse()` no genera error.

**Post-export ONNX:**

```python
import onnx, onnxruntime as ort, numpy as np

m = onnx.load('best.onnx')
assert m.opset_import[0].version == 11, f"Opset esperado 11, got {m.opset_import[0].version}"
assert m.ir_version <= 10, f"IR version > 10 puede romper onnxsim 0.6.x"

# Simplificar
import subprocess
subprocess.run(['python', '-m', 'onnxsim', 'best.onnx', 'best_sim.onnx'], check=True)

# Inferencia ORT (sanity check)
sess = ort.InferenceSession('best_sim.onnx', providers=['CPUExecutionProvider'])
in_name = sess.get_inputs()[0].name
in_shape = sess.get_inputs()[0].shape
print(f"Input: {in_name}, shape: {in_shape}")  # esperar [1, 3, 416, 416]
dummy = np.random.randn(1, 3, 416, 416).astype(np.float32)
out = sess.run(None, {in_name: dummy})
print(f"Output shapes: {[o.shape for o in out]}")
```

**Build TRT engine en Nano (no en Colab/Kaggle):**

```bash
# Solo en el Jetson Nano:
trtexec --onnx=best_sim.onnx \
        --fp16 \
        --workspace=1024 \
        --saveEngine=best.engine \
        --verbose 2>&1 | tee trt_build.log

# Validar engine tras build
trtexec --loadEngine=best.engine --shapes=images:1x3x416x416 --iterations=100
# Esperar latencia ~30-50 ms/iteration = 20-33 FPS
```

---

## Gotchas conocidos (acumulado)

| # | Plataforma / Componente | Gotcha | Mitigación |
|---|------------------------|--------|------------|
| 1 | Colab Py 3.12 + TF 2.13/2.14/2.15 | No hay wheels cp312 | Usar `condacolab` para bajar a Py 3.10 |
| 2 | TF OD API + TF 2.16+ | `tf.estimator` removido (#13575, #13599) | Pin `tensorflow==2.15.0` |
| 3 | TF OD API + `graph_rewriter` | QAT silently broken (#9835) | Usar PTQ post-train; o pivotar a MediaPipe Model Maker |
| 4 | TFMOT + SSD FPN | No soporta arquitecturas compuestas | No usar para Track A |
| 5 | MediaPipe Model Maker | NMS omitido en export TFLite | Solo Plan B, requiere decoder custom Nano |
| 6 | Roboflow `version.download(location=X)` | Path relativo a CWD si `location=None` | `os.chdir(WORK_DIR)` + `os.environ["DATASET_DIRECTORY"]` + cascada de búsqueda |
| 7 | Roboflow `data.yaml` clases fantasma (#88) | Backend bug sin fix | Validar `nc:` y `names:` manualmente |
| 8 | EfficientNMS_TRT en Maxwell | Plugin roto (#1538) | NMS en CPU NumPy |
| 9 | onnxsim 0.4.36 en Py 3.12 | No compila (#334) | Pin `>=0.6.2` |
| 10 | Ultralytics `best_onnx_opset` | Default 20 con torch 2.9+ | Forzar `opset=11` explícito |
| 11 | onnxsim breaking changes 0.5.0+ | `--dynamic-input-shape`, `--input-shape` deprecados | Usar nueva sintaxis |
| 12 | Kaggle GPU quota 30 h/sem | Sesión cap 12 h | Checkpoints cada 2-3 epochs; planear chunks |
| 13 | Colab idle disconnect | 30-90 min sin actividad | Heartbeat cells obligatorios |
| 14 | `DATASET_DIRECTORY` env var | Debe estar set ANTES de `Roboflow()` | Set en cell-8 de deps |
| 15 | Pillow 12 + OD API legacy | `Image.fromarray` stricter type checking | Pin `Pillow==10.4.0` |
| 16 | `protobuf` con OD API | conflicto si versión > 3.20 | Pin `protobuf==3.20.3` |
| 17 | TFLite forward-compat TF 2.15 → 2.5 | op_version puede ser nuevo | Validar con `flatbuffer_utils.py` pre-deploy |
| 18 | Track B Roboflow Kaggle secret | Toggle "Attach Secret" requerido | UI step manual |
| 19 | YOLOv8n CPU TFLite en Nano | < 10 FPS, no viable | Solo TRT FP16 GPU |
| 20 | TRT engine OOM Jetson Nano (#14751) | TRT build necesita workspace | `trtexec --workspace=1024` |

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Discovery / Agent |
|---|--------|-----|------|--------------------|
| 1 | Colab Runtime Version FAQ | https://research.google.com/colaboratory/runtime-version-faq.html | Doc oficial | research-web |
| 2 | Kaggle Notebook Specs | https://www.kaggle.com/docs/notebooks | Doc oficial | research-web |
| 3 | TensorRT 8.2 ops support | https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md | Doc oficial NVIDIA | research-web |
| 4 | PyTorch Version Compatibility | https://github.com/pytorch/pytorch/wiki/PyTorch-Versions | Doc oficial | research-web |
| 5 | Ultralytics Jetson Guide | https://docs.ultralytics.com/guides/nvidia-jetson/ | Doc oficial | yolov8-trt |
| 6 | MediaPipe Model Maker (Object Detector) | https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector | Doc oficial | mediapipe-mm |
| 7 | TF blog 2024-03 Python 3.12 support | https://blog.tensorflow.org/2024/03/whats-new-in-tensorflow-216.html | Blog oficial | research-code |
| 8 | TF blog 2022-06 QAT Model Garden | https://blog.tensorflow.org/2022/06/Adding-Quantization-aware-Training-and-Pruning-to-the-TensorFlow-Model-Garden.html | Blog oficial | research-academic |
| 9 | DeepWiki tensorflow/models | https://deepwiki.com/tensorflow/models | Doc generada | research-code |
| 10 | DeepWiki google-ai-edge/mediapipe | https://deepwiki.com/google-ai-edge/mediapipe | Doc generada | research-code |
| 11 | DeepWiki roboflow/roboflow-python | https://deepwiki.com/roboflow/roboflow-python/4.2-dataset-download | Doc generada | research-web |
| 12 | googlecolab/colabtools #5483 Py 3.12 | https://github.com/googlecolab/colabtools/issues/5483 | Foro GitHub | colab-tfod |
| 13 | tensorflow/models #13575 estimator | https://github.com/tensorflow/models/issues/13575 | Foro GitHub | research-code |
| 14 | tensorflow/models #13599 PR guard TF 2.16 | https://github.com/tensorflow/models/issues/13599 | Foro GitHub | research-code |
| 15 | tensorflow/models #9835 graph_rewriter | https://github.com/tensorflow/models/issues/9835 | Foro GitHub | research-code |
| 16 | tensorflow/models #11168 eval_pb2 | https://github.com/tensorflow/models/issues/11168 | Foro GitHub | research-code |
| 17 | tensorflow/tensorflow #62003 Py3.12 | (referenced) | Foro GitHub | research-code |
| 18 | NVIDIA/TensorRT #1538 EfficientNMS Maxwell | (referenced) | Foro GitHub | proyecto previo |
| 19 | ultralytics/ultralytics #14751 Nano OOM | https://github.com/ultralytics/ultralytics/issues/14751 | Foro GitHub | yolov8-trt |
| 20 | ultralytics/ultralytics #10298 postprocess Nano | https://github.com/ultralytics/ultralytics/issues/10298 | Foro GitHub | yolov8-trt |
| 21 | ultralytics/ultralytics #7222 FP16 TRT | https://github.com/ultralytics/ultralytics/issues/7222 | Foro GitHub | yolov8-trt |
| 22 | ultralytics/ultralytics #19498 upsample opset8 | https://github.com/ultralytics/ultralytics/issues/19498 | Foro GitHub | research-web |
| 23 | roboflow/roboflow-python #125 paths inconsistentes | https://github.com/roboflow/roboflow-python/issues/125 | Foro GitHub | roboflow-bug |
| 24 | roboflow/roboflow-python #240 Incorrect Data Path | https://github.com/roboflow/roboflow-python/issues/240 | Foro GitHub | roboflow-bug |
| 25 | roboflow/roboflow-python #88 Wrong classes data.yaml | https://github.com/roboflow/roboflow-python/issues/88 | Foro GitHub | research-web |
| 26 | roboflow/notebooks #69 FileNotFoundError | https://github.com/roboflow/notebooks/issues/69 | Foro GitHub | roboflow-bug |
| 27 | roboflow/roboflow-python #108 redownload | https://github.com/roboflow/roboflow-python/issues/108 | Foro GitHub | roboflow-bug |
| 28 | daquexian/onnx-simplifier #334 Py 3.12 wheel | https://github.com/daquexian/onnx-simplifier/issues/334 | Foro GitHub | research-web |
| 29 | daquexian/onnx-simplifier #367 ir_version | https://github.com/daquexian/onnx-simplifier/issues/367 | Foro GitHub | research-web |
| 30 | discuss.ai.google.dev MediaPipe QAT accuracy | https://discuss.ai.google.dev/t/mediapipe-massive-accuracy-loss-with-quantization-aware-training/23177 | Foro Google | mediapipe-mm |
| 31 | discuss.ai.google.dev TFLite custom detector | https://discuss.ai.google.dev/t/decoding-of-tflite-custom-object-detector-output-from-model-trained-with-mediapipe-mobilenetv2/32206 | Foro Google | mediapipe-mm |
| 32 | Qengineering YoloV8 TensorRT Jetson | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo | yolov8-trt |
| 33 | the0807 YOLOv8 ONNX TensorRT | https://github.com/the0807/YOLOv8-ONNX-TensorRT | Repo | yolov8-trt |
| 34 | Qengineering TensorFlow JetsonNano (wheels) | https://github.com/Qengineering/TensorFlow-JetsonNano | Repo | jetpack-tflite |
| 35 | google-coral.github.io tflite_runtime wheels | https://google-coral.github.io | Doc oficial | jetpack-tflite |
| 36 | NobuoTsukamoto/benchmarks (Jetson Nano) | https://github.com/NobuoTsukamoto/benchmarks | Repo benchmarks | research-academic |
| 37 | NobuoTsukamoto/tensorrt-examples | https://github.com/NobuoTsukamoto/tensorrt-examples | Repo | research-academic |
| 38 | Jacob et al. 2018 — QAT integer-only inference | https://arxiv.org/abs/1712.05877 | Paper CVPR | research-academic |
| 39 | Karimov et al. 2025 — Quantization robustness | https://arxiv.org/abs/2508.19600 | Paper arXiv | research-academic |
| 40 | Zagitov et al. 2024 — Edge object detection | https://doi.org/10.18287/2412-6179-CO-1343 | Paper Computer Optics | research-academic |
| 41 | Alqahtani et al. 2024 — Edge DL benchmarks | https://arxiv.org/abs/2409.16808 | Paper arXiv | research-academic |
| 42 | Swaminathan et al. 2024 — Jetson Nano DL | https://arxiv.org/abs/2406.17749 | Paper arXiv | research-academic |
| 43 | Lazarevich et al. 2023 — YOLOBench | https://arxiv.org/abs/2307.13901 | Paper ICCVW | research-academic |
| 44 | Trisuwita et al. 2024 — SSD MV2 helmet detection | https://doi.org/10.34010/komputika.v13i1.10333 | Paper Komputika | research-academic |
| 45 | TF2 Detection Model Zoo | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md | Doc oficial | research-academic |
| 46 | Colab vscode #215 userdata timeout | https://github.com/googlecolab/colab-vscode/issues/215 | Foro GitHub | research-web |
| 47 | Kaggle docker-python kaggle_secrets.py | https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py | Código fuente | research-web |
| 48 | Stanford Library Colab API auth guide | https://guides.library.stanford.edu/api_auth/colab | Doc institucional | research-web |
| 49 | Kaggle Feature Launch User Secrets | https://www.kaggle.com/product-feedback/114053 | Doc oficial | research-web |
| 50 | Kaggle blog GPU hours con Colab Pro | https://www.kaggle.com/blog/level-up-your-compute-more-gpu-hours-on-kaggle-wit | Blog oficial | research-web |
| 51 | mediapipe github #5003 tfrecord support | (referenced) | Foro GitHub | research-academic |
| 52 | mediapipe github #4744 RPi tflite inference | https://github.com/google-ai-edge/mediapipe/issues/4744 | Foro GitHub | mediapipe-mm |
| 53 | mediapipe github #4836 mobilenetv2 inference | https://github.com/google-ai-edge/mediapipe/issues/4836 | Foro GitHub | mediapipe-mm |
| 54 | TF OD API github tutorial deprecation | README de research/object_detection | Doc oficial | colab-tfod |
| 55 | Ultralytics TensorRT integration blog | https://www.ultralytics.com/blog/optimizing-ultralytics-yolo-models-with-the-tensorrt-integration | Blog oficial | yolov8-trt |
| 56 | jetson-ai-lab Ultralytics tutorial | https://jetson-ai-lab.com/tutorial_ultralytics.html | Tutorial | yolov8-trt |
| 57 | trtutils YOLOv8 tutorial | https://trtutils.readthedocs.io/en/stable/tutorials/yolo/yolov8.html | Doc oficial | yolov8-trt |

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco | Output |
|-------|-------|-------------|------|--------|
| 1 | 2026-05-12 | Alto | Compatibilidad stack training Colab/Kaggle vs runtime Jetson Nano JetPack 4.6.1 + bug Roboflow `location` + decisión QAT Track A | Este documento (530+ líneas), 5 discoveries, 3 agents background, fixes aplicados a ambos notebooks |
| 2 | 2026-05-12 | Alto | Validación stack Track A post-fix Ronda 1 — condacolab `python_version`, pin tensorflow/models, conflicto protobuf 5.x, output format `model_main_tf2.py` | 3 discoveries adicionales (6-8), 2 agentes background, fixes a celdas 8 y 10 (install_from_url + Miniforge 23.11.0 + SHA pin master 9cafa3d150 + grpcio-tools 1.64.1 + PROTOCOL_BUFFERS env var + ANTIALIAS shim) |
| 3 | 2026-05-12 | Alto | Validación stack Track B YOLOv8 — regresiones Ultralytics 8.4/8.5, Stack Kaggle 2026 (Py 3.12 + PyTorch 2.10 + NumPy 2.4), validación ONNX→TRT 8.2.1 Maxwell, Roboflow SDK 1.3.9, compat consumidor Jetson | 4 discoveries (9-12), 3 agentes background (research-code + research-web + research-video MCP youtube), fixes a celdas 8 y 20 (numpy<2.0 pin + onnxsim→onnxslim + validación ops problematicos TRT 8.2 + assert opset 11/IR 10) |
| 4 | 2026-05-12 | Alto | Migración Vast.ai dual-stack (TF 2.15 + Ultralytics) + persistencia HF Hub + validación end-to-end Jetson Nano B01. Tras fallos sucesivos Colab condacolab/mamba | 5 discoveries (13-17), 4 agentes (research-code + research-web + research-academic + research-video), decisiones vinculantes: container vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 + dual virtualenv, RTX 4090 on-demand, repo HF privado mitgar14/embebidos-3-models, script run.sh headless con jupytext+papermill, checklist validación pre-deploy .tflite (flatbuffer_utils op_versions) y .onnx (polygraphy + script ops blacklist), engine TRT compilado en Nano siempre |

---

## Próximos pasos

1. ✅ **Ronda 1 — Aplicar fix deps a celda 8 Track A** — `condacolab` + TF 2.15 + tf-models-official 2.15 + protobuf 3.20.3 + Pillow 10.4. Reescribir manifest con stack JetPack 4.6.1 documentado.
2. ✅ **Ronda 1 — Aplicar fix deps a celda 8 Track B** — Ultralytics 8.4.46+, onnxsim 0.6.2+, opset 11 explícito en export, set `DATASET_DIRECTORY` env var.
3. ✅ **Ronda 2 — Corregir `condacolab.install(python_version=...)`** (no existe) por `install_from_url` apuntando a Miniforge 23.11.0-0.
4. ✅ **Ronda 2 — Corregir pin `tensorflow/models` tag `v2.15.0`** (no contiene `research/`) por commit SHA `9cafa3d150` de master.
5. ✅ **Ronda 2 — Compilar protos con `grpcio-tools==1.64.1`** para evitar `runtime_version` import en `*_pb2.py`.
6. ✅ **Ronda 2 — Red de seguridad `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`** + shim `PIL.Image.ANTIALIAS = LANCZOS`.
7. ❌ **Ronda 4 — Abandonar Colab para Track A** (condacolab + mamba install python=3.10 falla por google-colab pin a 3.12 en conda env). Pivote a Vast.ai.
8. ✅ **Ronda 4 — Plan migratorio a Vast.ai** definido: container, GPU, repo HF, workflow headless, validación pre-deploy, auto-destroy.
9. ⏸️ **Implementar `train_track_a.py`** (jupytext convert + adaptaciones para headless, eliminar bootstrap defensivo, agregar `CommitScheduler` HF Hub).
10. ⏸️ **Implementar `train_track_b.py`** (jupytext convert + W&B nativo + numpy<2.0 pin + Roboflow cascada conservada).
11. ⏸️ **Implementar `run.sh`** template del agente research-code con clone + uv venvs + train + upload + auto-destroy.
12. ⏸️ **Crear repo `mitgar14/embebidos-3-models`** privado en HF Hub con `track_a/` y `track_b/`.
13. ⏸️ **Implementar checklist validación pre-deploy** (script Python para flatbuffer_utils + polygraphy + TRT82_UNSUPPORTED ops blacklist).
14. ⏸️ **Pilot run Vast.ai Track A** (RTX 4090 on-demand, 1-2 h, end-to-end).
15. ⏸️ **Pilot run Vast.ai Track B** (RTX 4090 spot si pilot A funciona, 30-60 min).
16. ⏸️ **Deploy en Nano** + golden test set + medición mAP cross-architecture (verificar gap empírico documentado: Karimov 2025 es sm_75+, no Maxwell sm_53).
17. ⏸️ **Si drop INT8 en Maxwell excede umbral** (>10 pp), pivote a FP16 only en Track A (sacrificar latencia por accuracy).

---

## Notas y gaps de evidencia

1. **Forward-compat .tflite TF 2.15 → TF 2.5 op_version:** no existe tabla pública exhaustiva mapeando versión de operador por TF release. Verificación obligatoria con `flatbuffer_utils.py` antes de deploy al Nano.
2. **MediaPipe Model Maker MOBILENET_V2_I320 arquitectura:** conflicto entre fuentes — research-code dice "RetinaNet backbone MobileNetV2" citando `build_qat_retinanet`; research-academic dice "SSD MobileNetV2 plain" citando path GCS `mobilenetv2_ssd_coco/mobilenetv2_ssd_i320_ckpt.tar.gz`. La verdad probable: el wrapper `build_qat_retinanet` aplica QAT al checkpoint SSD; el nombre "retinanet" es del wrapper, no de la arquitectura subyacente. No bloqueante para la decisión (es Plan B).
3. **FPS estimados CPU Cortex-A57 Nano son extrapolaciones de Pi4 Cortex-A72** ajustadas por frecuencia de reloj. Margen estimado ±3 FPS. Medición empírica requerida tras primer deploy.
4. **TF 2.21.0 wheel cp312** no verificado directamente en PyPI; inferido de release notes sin breaking changes Python compat 2.19→2.21.
5. **Roboflow v1.3.8 (2026-05-06):** no se confirmó si el bug `location` está parcheado en esta versión. Aplicamos el fix defensivo independientemente.
6. **Kaggle PyTorch version v168 (mar 2026):** no se pudo confirmar exactamente; `Dockerfile.tmpl` no fija versión. Verificar en runtime con `torch.__version__`.

---

## Ronda 2 — 2026-05-12 (alta)

### Disparador

Tras aplicar el fix Ronda 1 al Track A, una nueva ejecución en Colab reveló dos fallos encadenados:

1. **condacolab no bajó Python a 3.10** — el kernel reinició pero `sys.version_info` seguía mostrando `3.12.13`, y `pip install tensorflow==2.15.0` falló con "Could not find version".
2. **Pillow 11.3.0 quedó instalado tras `pip install tf-models-official==2.15.0`** y `import google.protobuf` falló con `cannot import name 'runtime_version' from 'google.protobuf'`.

Pregunta de la ronda: ¿qué otras fallas latentes hay en `train_track_a_ssd.ipynb` que el fix Ronda 1 no cubre?

### Track A (agentes)

**research-code (TFOD API stack TF 2.15) — sintesis:**

- **`tf-models-official 2.15.0` declara `Pillow` SIN pin** en su `requires_dist` (PyPI JSON, upload 2023-11-15). Idem `research/object_detection/packages/tf2/setup.py` (commit `152a0ce`, 2022-09-13): `'pillow'` sin version constraint. → `pip install -e research/` puede subir Pillow a 11.x/12.x sin avisar. Solucion: `--no-deps` en el `pip install -e` + `--force-reinstall --no-deps Pillow==10.4.0` al final.
- **TF 2.15.0 declara `protobuf (!=4.21.x,<5.0.0dev,>=3.20.3)`** — acepta 3.20.x a 4.25.x, NO acepta 5.x. Pinear 3.20.3 evita conflictos con deps transitivas (apache-beam, google-api-core, google-cloud-*) que pueden traer protobuf 5.x.
- **`runtime_version.py` se introdujo en protobuf 5.26.0** (commit `554a00c`, 2024-03-06; lanzamiento PyPI 2024-03-13). Archivos `*_pb2.py` generados con `protoc 5.x+` incluyen `from google.protobuf import runtime_version as _runtime_version`, que no existe en protobuf 3.20.x ni 4.x. → Si TFMOT 0.7.5 wheel o el wheel de `tf-models-official` tienen `*_pb2.py` regenerados con protoc 5.x, el import explota con protobuf 3.20.3 pineado. **Workaround robusto: `os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"`** (parser puro Python, 5-10x mas lento, negligible para training overhead).
- **`condacolab.install_miniforge` NO acepta `python_version`** — verificado en `condacolab.py` linea 233 (rama main, file SHA `7283febc5...`). Firma real: `install_miniforge(prefix, env, run_checks, restart_kernel)`. El parametro `python_version="3.10"` que aplicamos en Ronda 1 era inexistente y silenciosamente ignorado o causaba TypeError.
- **`sys.executable` post-condacolab NO cambia** — sigue apuntando al path original (`/usr/bin/python3`), pero ese path es ahora un wrapper shell que hace `exec /opt/conda/bin/python`. Por tanto subprocess con `sys.executable` corre correctamente Python 3.10 del conda env. Mecanismo verificado en `condacolab.py` lineas 190-206.
- **Issue #11168 `eval_pb2`** no tiene workaround oficial pero la causa es bien conocida: protos no compilados. Fix: clonar repo + `protoc object_detection/protos/*.proto --python_out=.` antes de `pip install -e`. El `protoc` debe ser <= 4.x para evitar el import de `runtime_version`. Solucion: `pip install grpcio-tools==1.64.1` y compilar con `python -m grpc_tools.protoc`.
- **Tag `v2.15.0` de tensorflow/models NO contiene `research/`** — la release note verbatim: "Note that Research/tutorial/sample models have been removed." Solo contiene `official/` + `orbit/`. → Pin a tag falla con `assert (research_dir / "object_detection").exists()`.
- **Ultimo commit estable de master pre-Pillow12-patch:** `9cafa3d150` (2026-03-17, "Merge PR #13619 ai-gsutil-migration"). El patch que rompe el pin Pillow 10.4 es `971ded9e16` (2026-04-29, "Support Pillow 12's stricter type checking in Image.fromarray"). Pin estable adicional verificado: `f9fdc4faef47af76351204b6d8df576f0e79baab` (2024-06-07, pre-NumPy-2.0).
- **Output de `model_lib_v2.py`:** dos lineas por iteracion (`LOG_EVERY=100`): `Step {} per-step time {:.3f}s` y `pprint.pformat(logged_dict_np, width=40)`. Las regex actuales del notebook `Step\\s+(\\d+)\\s+per-step` y `Loss/total_loss[:\\s=]+([\\d.]+)` matchean correctamente el formato real (`absl-py` prefija con timestamp `IYYYYMMDD HH:MM:SS.usec`).

**research-web (pin strategy Pillow + protobuf):**

- **Constraint file de pip** es el mecanismo canonico documentado (pip docs v26.1.1) para forzar pin sobre deps transitivas: `pip install <pkg> -c constraints.txt`. Workaround mas pragmatico: `pip install --force-reinstall --no-deps Pillow==10.4.0` AL FINAL, despues de todos los installs. Patron con 327 votos en SO #19548957.
- **`Image.ANTIALIAS` removido en Pillow 10.0** — codigo de `research/object_detection/utils/visualization_utils.py` puede invocarlo. Shim: `if not hasattr(PIL.Image, "ANTIALIAS"): PIL.Image.ANTIALIAS = PIL.Image.LANCZOS`. Confirmado en `tensorflow/hub` commit `a01d02f9...` (jul 2023).
- **Workarounds runtime_version:** (A) env var `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` — robusto, lento. (B) Copiar `runtime_version.py` de protobuf 5.x a la instalacion de 3.20.x — quirky. (C) Recompilar protos con `grpcio-tools==1.64.1` — fix correcto pero requiere clone del repo.
- **Tutoriales 2024-2026:** EdjeElectronics, Roboflow, RTD tensorflow-object-detection-api-tutorial — todos desactualizados para el stack TF 2.15 + condacolab + Py 3.10 + protobuf 3.20.3. La comunidad migro a YOLOv8/v11 o al Official Model Garden. No hay tutorial publico que cubra este stack exacto. → El notebook es esencialmente un trabajo de arqueologia: TF OD API legacy en runtime moderno.

### Track A (discoveries auxiliares)

- **discover-6** (54 resultados): pinning Pillow + protobuf en TF OD API. Confirma issue #22726 Ultralytics ("Pillow new version breaking ultralytics import in Google Colab", nov 2025) — mismo patron de bug en otro proyecto.
- **discover-7** (52 resultados): formato output de `model_main_tf2.py`. Las regex actuales del notebook funcionan; no requiere cambio.
- **discover-8** (56 resultados): condacolab + subprocess. Confirma que `sys.executable` apunta al wrapper, no hay bug en subprocess. Issue #21 (resuelto en PR #31, oct 2022) es historico.

### Track B (sin actividad esta ronda)

Track B (YOLOv8 Kaggle) ya quedo estable en Ronda 1. No se descubrieron fallos nuevos.

### Fixes aplicados a `train_track_a_ssd.ipynb` (Ronda 2)

**Celda 8 (Dependencias):**
- Cambiado `condacolab.install(python_version="3.10")` (parametro inexistente) por `condacolab.install_from_url(MINIFORGE_URL)` con URL fija a Miniforge `23.11.0-0` (ultima version con Python 3.10 base, antes del cambio a Python 3.12 en Miniforge 24.5.0).
- Anadida red de seguridad post-restart: `os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"` antes de cualquier `pip install`.
- Anadido re-pin defensivo al final de instalaciones: `pip install --force-reinstall --no-deps Pillow==10.4.0 protobuf==3.20.3`.
- Actualizado mensaje de error de validacion con guia clara (workaround manual `mamba install python=3.10` + plan B TF 2.19).

**Celda 10 (TF Object Detection API):**
- Cambiado pin de tag `v2.15.0` (no contiene `research/`) por pin a commit SHA `9cafa3d150` de master (2026-03-17, pre-Pillow12-patch).
- Cambiado `git clone --depth 1 --branch v2.15.0` por `git clone --filter=blob:none --no-checkout` + `git checkout SHA` (necesario porque git no soporta shallow clone con SHA arbitrario).
- Anadido assert `(research_dir / "object_detection").exists()` como proteccion contra checkout fallido.
- Anadido `pip install grpcio-tools==1.64.1` + compilacion de protos con `python -m grpc_tools.protoc` (forzar protoc 4.x para evitar `runtime_version` import).
- Mantenido `pip install --no-deps -e research/` (evita que el setup.py traiga apache-beam + google-api-core con protobuf 5.x).
- Mantenido re-pin defensivo Pillow 10.4 + protobuf 3.20.3.
- Anadido shim `PIL.Image.ANTIALIAS = LANCZOS` como red de seguridad para codigo legacy.

### Fuentes consultadas (Ronda 2, acumulativas en tabla principal)

| # | Titulo | URL | Tipo | Track |
|---|--------|-----|------|-------|
| 58 | tf-models-official 2.15.0 PyPI JSON | https://pypi.org/pypi/tf-models-official/2.15.0/json | PyPI metadata | research-code |
| 59 | tensorflow 2.15.0 PyPI JSON | https://pypi.org/pypi/tensorflow/2.15.0/json | PyPI metadata | research-code |
| 60 | tensorflow-model-optimization 0.7.5 PyPI JSON | https://pypi.org/pypi/tensorflow-model-optimization/0.7.5/json | PyPI metadata | research-code |
| 61 | research/object_detection/packages/tf2/setup.py (master) | https://github.com/tensorflow/models/blob/master/research/object_detection/packages/tf2/setup.py | Codigo fuente | research-code |
| 62 | condacolab.py (main, v0.1.4) | https://github.com/conda-incubator/condacolab/blob/main/condacolab.py | Codigo fuente | research-code |
| 63 | Miniforge 24.5.0 release notes ("Base version of python is now 3.12") | https://github.com/conda-forge/miniforge/releases/tag/24.5.0-0 | Release notes | research-web |
| 64 | Miniforge 23.11.0-0 release | https://github.com/conda-forge/miniforge/releases/tag/23.11.0-0 | Release notes | research-web |
| 65 | issue tensorflow/models #11168 (eval_pb2 not found) | https://github.com/tensorflow/models/issues/11168 | GitHub Issue | research-code |
| 66 | issue tensorflow/models #11085 (workaround @eldivategar) | https://github.com/tensorflow/models/issues/11085 | GitHub Issue | research-code |
| 67 | issue tensorflow/models #11192 (runtime_version, @namasSinjali fix) | https://github.com/tensorflow/models/issues/11192 | GitHub Issue | research-web |
| 68 | issue tensorflow/models #13599 (TF 2.16+ incompatibility PR, ene 2026) | https://github.com/tensorflow/models/issues/13599 | GitHub Issue/PR | research-web |
| 69 | issue ultralytics/ultralytics #22726 (Pillow 12 breaking Colab) | https://github.com/ultralytics/ultralytics/issues/22726 | GitHub Issue | research-web |
| 70 | commit tensorflow/models 971ded9e16 (Pillow 12 patch) | https://github.com/tensorflow/models/commit/971ded9e16 | Commit | research-code |
| 71 | commit tensorflow/models 9cafa3d150 (pre-Pillow12 stable) | https://github.com/tensorflow/models/commit/9cafa3d150 | Commit | research-code |
| 72 | protocolbuffers/protobuf commit 554a00c (introduce runtime_version) | https://github.com/protocolbuffers/protobuf/commit/554a00c | Commit | research-code |
| 73 | SO #78671850 (recompile protos with grpcio-tools 1.64.1) | https://stackoverflow.com/questions/78671850 | Stack Overflow | research-web |
| 74 | SO #19548957 (pip --force-reinstall --no-deps) | https://stackoverflow.com/questions/19548957 | Stack Overflow | research-web |
| 75 | foro Google AI Developers — runtime_version downgrade workaround | https://discuss.ai.google.dev/t/importerror-cannot-import-name-runtime-version-from-google-protobuf/22770 | Foro oficial | research-web |
| 76 | pip docs v26.1.1 — constraint files | https://pip.pypa.io/en/stable/topics/dependency-resolution/ | Doc oficial | research-web |
| 77 | tensorflow/hub commit a01d02f9 (ANTIALIAS -> LANCZOS) | https://github.com/tensorflow/hub/commit/a01d02f9 | Commit | research-web |
| 78 | discover-6 — TFOD API + Pillow + protobuf | investigaciones/2026-05-12/discover-6-tfod-api-pillow-protobuf.md | Discovery local | discover.py |
| 79 | discover-7 — model_main_tf2 output format | investigaciones/2026-05-12/discover-7-model-main-tf2-output.md | Discovery local | discover.py |
| 80 | discover-8 — condacolab + subprocess | investigaciones/2026-05-12/discover-8-condacolab-subprocess.md | Discovery local | discover.py |

### Gotchas nuevos (Ronda 2)

1. **`condacolab.install()` y `install_miniforge()` NO aceptan `python_version`**. La API actual solo permite pasar un URL custom via `install_from_url`. Para Python 3.10 hay que apuntar a Miniforge 23.11.0 o anterior.
2. **Tag `v2.15.0` de `tensorflow/models` NO incluye `research/`**. Solo `official/` + `orbit/`. Para clonar OD API hay que usar master + checkout a SHA pre-Pillow12-patch.
3. **`pip install --no-deps -e research/`** es obligatorio para evitar que `apache-beam`, `google-api-core` y `google-cloud-*` traigan protobuf 5.x via deps transitivas.
4. **Re-pin defensivo Pillow + protobuf al final** (`--force-reinstall --no-deps`) garantiza supervivencia del pin tras secuencia de installs largos.
5. **`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`** es la red de seguridad mas robusta contra el bug `runtime_version`. Pequeno overhead, alta robustez.
6. **`grpcio-tools==1.64.1`** genera `*_pb2.py` compatibles con protobuf 3.20.x (sin import de `runtime_version`). Versiones >= 1.66.0 son incompatibles con el pin 3.20.3.
7. **`Image.ANTIALIAS` deprecado en Pillow 9.1, removido en Pillow 10.0**. Shim `PIL.Image.ANTIALIAS = LANCZOS` defensivo.
8. **`condacolab.check()` lanza `AssertionError`, NO devuelve `False`**. Source verbatim (`condacolab.py` linea 320): `assert find_executable("conda"), "Conda not found!"`. El patron `if not condacolab.check():` rompe con `AssertionError: Conda not found!` en el primer run cuando conda aun no esta instalado. Solucion: llamar `install_from_url(...)` directamente — ya hace `try: check(); except AssertionError: pass` internamente. Confirmado en `install_from_url` lineas 132-136. Observado 2026-05-12 en Colab tras aplicar fix Ronda 2.
9. **Miniforge 23.11.0-0 tambien trae Python 3.12 en Colab** (observado empiricamente 2026-05-12: tras `condacolab.install_from_url("23.11.0-0/...")` y kernel restart, `sys.version_info` sigue en 3.12.13). Release notes 23.11.0-0 NO especifican version de Python explicitamente. Hipotesis: o (a) el installer fue rebuild con Python 3.12 retroactivamente, o (b) el wrapper de condacolab no exec el conda Python correctamente en Colab moderno. **Solucion robusta: forzar downgrade con `mamba install python=3.10` post-restart** (flujo de 2 restarts: instala condacolab -> restart -> detecta Python != 3.10 -> mamba downgrade -> restart -> Python 3.10).
10. **`install_from_url` llama `do_shutdown(True)` asincrono**. El codigo siguiente al `install_from_url` SI se ejecuta antes de que el kernel muera. Si la celda continua con un check de Python version inmediato, va a leer el Python OLD (no el del kernel post-restart). Solucion: `sys.exit(0)` justo despues de `install_from_url` para terminar la celda limpiamente.
11. **Tras kernel restart en Colab, NO se mantienen las definiciones de celdas previas**. Si la celda 8 usa `banner()`, `t_ts()`, `IS_COLAB` definidas en celdas anteriores (e.g., celda 2), tras un `do_shutdown(True)` esos symbols quedan undefined. Re-ejecutar solo la celda 8 falla con `NameError: name 'banner' is not defined`. Solucion: bootstrap defensivo en celda 8 que define fallbacks idempotentes con `if "banner" not in dir(): def banner(s): ...`. Permite re-ejecutar la celda aislada en el flujo de 2 restarts.

### Gaps de evidencia (Ronda 2)

1. **TFMOT 0.7.5 `*_pb2.py` internos** — no verificado con que `protoc` se compilaron. Si fueron generados con protoc 5.x, el import de TFMOT explota con protobuf 3.20.3 a menos que `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` este activo. La env var ya esta aplicada en celda 8 como red de seguridad, mitiga.
2. **Miniforge 23.11.0 sigue disponible en GitHub releases** — verificado con `curl -sI` (HTTP 302 redirect a S3). Riesgo bajo de removal, pero anadir alternativa de fallback en docs si conda-forge limpia releases antiguos.
3. **Si `condacolab` lanza al cambiar el SHA del installer URL** (mecanismo de validacion interno), no se ha verificado. Probable que funcione porque `install_from_url` es un metodo publico documentado.

### Decision tecnica Ronda 2

- Pin de `tensorflow/models` corregido: **master SHA `9cafa3d150`** (no tag `v2.15.0`).
- Pin de Python: **condacolab + Miniforge 23.11.0-0** (no parametro `python_version`).
- Defensa de protobuf: **env var `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`** post-condacolab restart.
- Compilacion de protos: **`grpcio-tools==1.64.1`** (no `protoc` del sistema).
- Re-pin Pillow + protobuf: **`--force-reinstall --no-deps`** al final de cada bloque de installs.

Vinculante para el notebook actualizado. Si el siguiente piloto Colab falla, escalar a Plan B (TF 2.19 + tf-keras + `official/projects/qat/vision`) sin volver a depender de `condacolab`.


---

## Ronda 3 — 2026-05-12 (alta) — Track B YOLOv8

### Disparador

Tras estabilizar Track A en Ronda 2, foco a Track B `train_track_b_yolov8.ipynb`. El notebook pineaba `ultralytics>=8.4.46,<8.5`, `onnxsim>=0.6.2,<0.7`, `opset=11` explicito en export, y `nms=False`. Ronda 1 no profundizo en regresiones recientes del exporter Ultralytics ni en el stack actual de Kaggle (Python 3.12, PyTorch 2.10, NumPy 2.4 default).

Pregunta de la ronda: que regresiones latentes hay entre lo que el notebook asume y la realidad del stack en mayo 2026?

### Track A (3 agentes background)

**research-code (Ultralytics 8.4/8.5 + Roboflow SDK source + ONNX export):**

- **`onnxsim` ya NO es el simplificador de Ultralytics** — desde 8.3+ migro a `onnxslim>=0.1.82`. Verificado leyendo source de `ultralytics/engine/exporter.py` y `pyproject.toml` rama main 2026-05-12. El flag `simplify=True` llama `onnxslim.slim(model_onnx)`, NO `onnxsim`. Pin `onnxsim>=0.6.2,<0.7` en notebook era irrelevante para el exporter.
- **PR #24028 (INT8 calib no-square imgsz) merged 2026-03-28 en v8.4.31**. Pin `>=8.4.46` correcto. v8.4.48 (2026-05-08) es latest. NO existe 8.5 a 2026-05-12.
- **`best_onnx_opset` con torch 2.9+: cap a opset 20** (legacy TorchScript). Si notebook omitiera `opset=11`, exportaria opset 20-22 -> incompatible con TRT 8.2.1 (parser hasta opset 13). Pin explicito `opset=11` es CRITICO.
- **Roboflow SDK v1.3.9 (2026-05-07, SHA `1e4cbc04`) SIN fix** para bug `dataset.location` vacio. Releases 1.3.7-1.3.9 enfocados en soft-delete/device-management. Workaround cascada cell-10 sigue obligatorio.
- **Ultralytics `pyproject.toml` declara `numpy<2.0.0`** (comentario "TF 2.20 compatibility"). Heredado automaticamente, pero en Kaggle deps transitivas pueden dejar 2.x ya instalado.
- **Ops YOLOv8n + opset 11 + TRT 8.2.1 + FP32 estatico sin NMS = todos soportados** (verificado contra `onnx-tensorrt/docs/operators.md` ref `release/8.2-GA`). Riesgo minimo.
- **Ops problematicos en TRT 8.2** (a evitar): `NonMaxSuppression` (EXPERIMENTAL FP32 only — solo si nms=True), `NonZero` (no soportado), `RoiAlign` (no — solo segmentacion), `QLinearConv/MatMul` (no — solo INT8 ONNX, no aplica), `Reciprocal` (no soportado, improbable en YOLOv8n detect FP32). El notebook con `nms=False` + `dynamic=False` + FP32 evita todos.
- **`Gather` rank-0 bug en TRT 8.x con opsets >=17** (issue NVIDIA/TensorRT #4383). Opset 11 lo evita.
- **PR #23808 ultralytics** (mar 2026): anade "safer ONNX opset cap for Torch 2.9+ exports" en el exporter interno — proteccion adicional al pin `opset=11` del notebook.

**research-web (Stack Kaggle 2026 + YOLOv8/TRT issues):**

- **Stack Kaggle GPU v168 (mar 2026)** — verificado leyendo Dockerfile.tmpl y releases v167/v168 del repo `Kaggle/docker-python`:

| Componente | Version mayo 2026 |
|------------|-------------------|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| CUDA host | 12.8.1 |
| cuDNN | 9.8.0.87 |
| NumPy | **2.4.x (default sin flag para mantener 1.x)** |
| NCCL | 2.25.1+cuda12.8 |

- **NumPy 2.x rompe ultralytics en Kaggle** — issue ultralytics/ultralytics #22346 ("NumPy 2.2.6 import errors when running on Kaggle T4x2", abierto). Pin explicito `numpy<2.0` ANTES de `pip install ultralytics` es defensa robusta.
- **PR ultralytics #23807** (2026-03-05): Docker base actualizado a `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`. Confirma torch 2.10 como runtime moderno.
- **Issue Ultralytics #22336** "dependency conflicts when install ultralytics" en Kaggle Python 3.11.13 — confirma fragilidad del stack pip default Kaggle.
- **EfficientNMS_TRT roto en Maxwell con TRT 8.x** — issue NVIDIA/TensorRT #1538 (2021). No re-abierto/fixed para Maxwell. NMS CPU NumPy es solucion establecida (alinea con memoria del proyecto).
- **Polygraphy NO funcional en JetPack 4.6.1** (Python 3.6.9 incompatible con versiones recientes que requieren Py 3.8+). Validacion pre-deploy debe usar `trtexec` directo.
- **ONNX Runtime + TRT EP NO viable en Nano JP4.6.1** (CUDA 10.2 vs ORT 1.11 requiere CUDA 11.4). Ruta recomendada: TRT runtime Python bindings + `cuda-python`.
- **TRT 8.2 INT8 con plugins fragil** — issue NVIDIA forums 349598 (oct 2025, Jetson Nano TRT 8.2.1.8): Polygraphy + INT8 calibration falla. Workaround: `IInt8EntropyCalibrator2` Python script + `trtexec --int8 --calib=<cache>`.

**research-video (YouTube MCP — Dustin Franklin, jetson-ai-lab, Ultralytics official):**

- **Confirma issue #1538 EfficientNMS_TRT Maxwell** — solucion estandar: NMS CPU o `BatchedNMS_TRT` con `--legacy_plugins`.
- **Confirma opset 11 canonico para TRT 8.2** — fuentes: `triple-Mu/YOLOv8-TensorRT` (`--opset 11` default en `export-det.py`), `Qengineering/YoloV8-TensorRT-Jetson_Nano` (`yolo export model=yolov8s.pt format=onnx opset=11 simplify=True`).
- **Confirma migracion onnxsim -> onnxslim** en Ultralytics 8.3+.
- **Benchmark FPS Qengineering (YOLOv8n FP16 @ 640x640, Maxwell sm_53):** 19 FPS solo inferencia. Para @416 + NMS CPU: 10-15 FPS pipeline completo. Confirma estimacion del notebook.
- **Gap de cobertura YouTube:** ningun video 2024-2026 cubre stack JetPack 4.6.1 + Ultralytics 8.4 + opset 11 + TRT 8.2.1 + Nano B01. Todo el material reciente apunta a JetPack 5/6 + Orin Nano (Ampere sm_87, ~40x mas rapido). Este proyecto es arqueologia confirmada en video.
- **Discrepancia descartada (low-confidence en research-video):** sugirio `ultralytics<=8.0.200` para JetPack 4.6.1 — incorrecto. Notebook corre Ultralytics SOLO en Kaggle, no en Nano. En Nano corre TRT runtime con `.engine` compilado de `.onnx`. Pin antiguo no aplica.

### Track B (4 discoveries + verificacion gh CLI)

- **discover-9** (28 resultados): releases Ultralytics 8.4.x, issue ONNX IR Version 8 (#19498), opset issues YOLOv11 (#16839), zenodo records con changelogs.
- **discover-10** (27 resultados): PR #23807 (PyTorch 2.10 + CUDA 12.8 base), PR #23808 (safer opset cap), issue #22346 (NumPy 2.2.6 Kaggle), issue colabtools #5801 (torch 2.10 release 2026-03-24).
- **discover-11** (30 resultados): issue #1538 NVIDIA/TensorRT (EfficientNMS Maxwell), issue Linaom1214 #112 (FP16 detection duplicates TRT 8.2.1.8), Qengineering repo (branch `tensorrt8`), issue #2821 (INT64 weights), issue #14751 (OOM exporting yolov8n on Nano).
- **discover-12** (24 resultados): issue roboflow-python #240 (Incorrect Data Path), issue notebooks #306 (`location` vacio), workaround issue #183 (reemplazar `dataset.location`), issue #333 (paths relativos data.yaml).
- **gh CLI verificacion**: `gh api repos/ultralytics/ultralytics/releases?per_page=10` confirma v8.4.48 (2026-05-08) como latest, NO 8.4.45 que sugirio research-web (inconsistencia resuelta).

### Fixes aplicados a `train_track_b_yolov8.ipynb` (Ronda 3)

**Celda 8 (Dependencias):**
- **Anadido `pip install "numpy<2.0"` ANTES de `pip install ultralytics`** — defensa contra NumPy 2.x default Kaggle 2026 (issue #22346).
- **Cambiado pin `onnxsim>=0.6.2,<0.7`** -> **`onnxslim>=0.1.82`** (el simplificador real que usa Ultralytics 8.3+, verificado en source de `exporter.py`).
- **Smoke check actualizado**: ahora valida `numpy<2.0` y `onnxslim` instalado en lugar de `onnxsim`.
- **Validacion dura post-install**: si `numpy.__version__.split(".")[0] >= 2`, abortar con mensaje claro citando issue #22346.
- **Actualizado header con stack Kaggle 2026 verificado** (Py 3.12, torch 2.10.0+cu128, NumPy 2.4.x, CUDA host 12.8) y referencias a PR #24028 + PR #23808.
- Mantenido `roboflow>=1.3.6,<1.4` con nota: v1.3.9 (mayo 2026) SIN fix `location` bug, workaround cascada cell-10 sigue obligatorio.

**Celda 20 (Export ONNX):**
- Actualizado comentario: `onnxsim 0.6.x integrado` -> `onnxslim integrado (Ultralytics 8.3+ migro de onnxsim)`.
- Anadida validacion adicional pre-deploy: detectar ops problematicos para TRT 8.2.1 (`Reciprocal`, `NonZero`, `RoiAlign`, `QLinearConv`, `QLinearMatMul`). Si aparece alguno, warn pero no abort (puede ser falso positivo si Ultralytics regresion futura los inyecta).
- Documentada justificacion explicita de cada arg con referencia a issues: opset=11 (issue #4383 Gather rank-0), nms=False (issue #1538 EfficientNMS Maxwell), simplify=True (onnxslim).

### Fuentes consultadas (Ronda 3)

| # | Titulo | URL | Tipo | Track |
|---|--------|-----|------|-------|
| 81 | ultralytics/ultralytics PR #24028 (INT8 calib no-square) | https://github.com/ultralytics/ultralytics/pull/24028 | GitHub PR | research-code |
| 82 | ultralytics/ultralytics PR #23807 (Docker pytorch 2.10) | https://github.com/ultralytics/ultralytics/pull/23807 | GitHub PR | research-web |
| 83 | ultralytics/ultralytics PR #23808 (safer opset cap torch 2.9+) | https://github.com/ultralytics/ultralytics/pull/23808 | GitHub PR | research-web |
| 84 | ultralytics/ultralytics issue #22346 (NumPy 2.2.6 Kaggle import error) | https://github.com/ultralytics/ultralytics/issues/22346 | GitHub Issue | research-web |
| 85 | ultralytics/ultralytics issue #22336 (Kaggle dep conflicts) | https://github.com/ultralytics/ultralytics/issues/22336 | GitHub Issue | research-web |
| 86 | ultralytics/ultralytics issue #19498 (ONNX IR Version 8) | https://github.com/ultralytics/ultralytics/issues/19498 | GitHub Issue | research-code |
| 87 | ultralytics/ultralytics issue #23436 (PyTorch 2.10 support) | https://github.com/ultralytics/ultralytics/issues/23436 | GitHub Issue | discover-10 |
| 88 | googlecolab/colabtools issue #5801 (torch 2.10.0 release 2026-03-24) | https://github.com/googlecolab/colabtools/issues/5801 | GitHub Issue | discover-10 |
| 89 | Kaggle/docker-python Dockerfile.tmpl + releases v167/v168 | https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl | Repo oficial | research-web |
| 90 | ultralytics/ultralytics releases (v8.4.31 -> v8.4.48 timeline) | https://github.com/ultralytics/ultralytics/releases | Repo oficial | research-code |
| 91 | ultralytics/ultralytics source engine/exporter.py + pyproject.toml rama main | https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py | Codigo fuente | research-code |
| 92 | NVIDIA/TensorRT issue #1538 (EfficientNMS Maxwell Nano TRT 8.0/8.2) | https://github.com/NVIDIA/TensorRT/issues/1538 | GitHub Issue | research-web |
| 93 | NVIDIA/TensorRT issue #4383 (Gather rank-0 opset 19 TRT 8.6) | https://github.com/NVIDIA/TensorRT/issues/4383 | GitHub Issue | research-code |
| 94 | NVIDIA developer forum 349598 (Polygraphy + INT8 Nano TRT 8.2.1.8) | https://forums.developer.nvidia.com/t/how-to-generate-and-verify-an-int8-calibration-cache-cache-for-trtexec-on-on-jetson-nano-tensorrt-8-2-1-8-polygraphy-failing-on-device/349598 | Foro oficial | research-web |
| 95 | NVIDIA developer forum 331356 (TRT INT8 Ultralytics assertion error Orin) | https://forums.developer.nvidia.com/t/tensorrt-int8-conversion-fails-with-assertion-error-using-ultralytics/331356 | Foro oficial | research-web |
| 96 | onnx/onnx-tensorrt operators.md ref release/8.2-GA | https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md | Doc oficial | research-code |
| 97 | Qengineering/YoloV8-TensorRT-Jetson_Nano (branch tensorrt8, opset 11) | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo comunidad | research-video |
| 98 | triple-Mu/YOLOv8-TensorRT (--opset 11 default) | https://github.com/triple-mu/YOLOv8-TensorRT | Repo comunidad | research-video |
| 99 | Linaom1214/TensorRT-For-YOLO-Series issue #112 (FP16 detection duplicates Nano 8.2.1.8) | https://github.com/Linaom1214/TensorRT-For-YOLO-Series/issues/112 | GitHub Issue | discover-11 |
| 100 | ONNX Runtime TRT EP requirements matrix | https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html | Doc oficial | research-web |
| 101 | roboflow/roboflow-python v1.3.9 source (version.py + dataset.py) | https://github.com/roboflow/roboflow-python/blob/main/roboflow/core/version.py | Codigo fuente | research-code |
| 102 | roboflow/roboflow-python issue #240 (Incorrect Data Path YOLOv8) | https://github.com/roboflow/roboflow-python/issues/240 | GitHub Issue | discover-12 |
| 103 | roboflow/notebooks issue #306 (dataset.location empty) | https://github.com/roboflow/notebooks/issues/306 | GitHub Issue | discover-12 |
| 104 | Ultralytics video "DeepStream Jetson Nano YOLOv8 Ep.82" (sep 2024) | https://youtu.be/wWmXKIteRLA | Video | research-video |
| 105 | Ultralytics Live Session 6 (Jetson edge, mar 2023) | https://youtu.be/QGeP-Y6KMLM | Video | research-video |
| 106 | discover-9 — Ultralytics 8.4/8.5 ONNX export | investigaciones/2026-05-12/discover-9-ultralytics-onnx.md | Discovery local | discover.py |
| 107 | discover-10 — Kaggle stack 2026 | investigaciones/2026-05-12/discover-10-kaggle-stack.md | Discovery local | discover.py |
| 108 | discover-11 — ONNX TRT Jetson Maxwell | investigaciones/2026-05-12/discover-11-onnx-trt-nano.md | Discovery local | discover.py |
| 109 | discover-12 — Roboflow SDK 1.3.8 | investigaciones/2026-05-12/discover-12-roboflow-sdk.md | Discovery local | discover.py |

### Gotchas nuevos (Ronda 3)

1. **Ultralytics 8.3+ migro de `onnxsim` a `onnxslim`**. Pin `onnxsim` en cualquier proyecto con ultralytics 8.3+ es irrelevante para `model.export(simplify=True)`. Verificado leyendo source.
2. **`best_onnx_opset` con torch 2.9+: cap a opset 20** (no 22 como dice docstring antiguo). Si se omite `opset=`, exporta opset 20 -> incompatible con TRT 8.2.
3. **Kaggle 2026 default NumPy 2.4.x rompe ultralytics en runtime** aunque pyproject.toml declare `numpy<2.0`. Deps transitivas pueden dejar 2.x. Pin explicito `numpy<2.0` ANTES de ultralytics es defensa.
4. **Roboflow SDK 1.3.9 (mayo 2026) SIN fix bug `dataset.location`**. Workaround cascada obligatorio.
5. **`Reciprocal`, `NonZero`, `RoiAlign` NO soportados en TRT 8.2** — improbable en YOLOv8n FP32 estatico pero validar grafo antes de trtexec.
6. **Polygraphy NO funcional en JetPack 4.6.1** (Python 3.6.9). Validacion pre-engine con `trtexec` directo.
7. **ONNX Runtime + TRT EP NO viable en Nano JP4.6.1** (ORT 1.11+ requiere CUDA 11.4, Nano tiene CUDA 10.2). Ruta: `tensorrt` Python bindings + `cuda-python` 11.0.
8. **EfficientNMS_TRT roto en Maxwell con TRT 8.x** (issue #1538, no fixed). NMS CPU NumPy obligatorio.

### Gaps de evidencia (Ronda 3)

1. **Pipeline end-to-end ultralytics 8.4.x + opset 11 + TRT 8.2.1.8 + Maxwell sm_53 + Nano B01**: ningun test publico, ningun video, ningun blog 2024-2026 documenta exactamente esta cadena. Inferencia validada por ops soportados (research-code) y issue tracking (research-web). Riesgo residual: bajo pero presente.
2. **`Reciprocal` op real en ONNX YOLOv8n FP32**: no verificado contra grafo exportado. Cell-20 ahora detecta y warnea.
3. **NumPy 1.x en Kaggle**: confirma que cuando `pyproject.toml` declara `numpy<2.0`, pip suele resolver 1.26.x. Pero la cadena `kaggle base -> numpy 2.4 -> pip install ultralytics -> numpy <2` puede o no funcionar dependiendo de orden de deps transitivas. Pin explicito previo es robusto.
4. **Roboflow v1.3.9 changelog publico**: no disponible, verificado solo el source diff entre 1.3.6 y 1.3.9. Posible fix silencioso entre 1.1.x y 1.2.x (no documentado).
5. **`onnxslim` vs `onnxsim` diferencia de output**: ambos simplifican ONNX, pero pueden producir grafos ligeramente distintos. `onnxslim` es mas nuevo y aceptado por Ultralytics; `onnxsim` esta en mantenimiento de comunidad. No verificado si trt parser acepta uno y no el otro para el mismo modelo.

### Decision tecnica Ronda 3

- Pin de NumPy: **`numpy<2.0` EXPLICITO antes de ultralytics** (no confiar en herencia de pyproject.toml).
- Pin de simplificador: **`onnxslim>=0.1.82`** (el real). Quitar `onnxsim` o dejarlo solo si se usa en cell custom.
- Pin de Ultralytics: **`>=8.4.46,<8.5`** confirmado correcto. Captura v8.4.46 -> v8.4.48 (latest).
- Export ONNX: **`opset=11` explicito + `nms=False` + `dynamic=False` + `simplify=True` + `device="cpu"`** (cada uno justificado por issue/PR documentado).
- Validacion pre-deploy: **`onnx.checker.check_model` + assert opset==11 + assert ir_version<=10 + detect ops problematicos**.
- Validacion en Nano: **`trtexec --onnx=... --fp16 --workspace=4096`** directamente (no Polygraphy, no ORT+TRT EP).
- NMS: **CPU NumPy** obligatorio (EfficientNMS_TRT roto Maxwell).
- Roboflow: **mantener cascada de busqueda en cell-10** (v1.3.9 sin fix).

Vinculante para Track B. Pipeline end-to-end (Kaggle export -> Nano deploy) ahora cubierto por defensa en cada capa: pin numpy, opset 11, nms=False, validacion grafica pre-deploy, trtexec en Nano.

---

## Ronda 4 — 2026-05-12 (alta)

### Disparador

Tras los fallos sucesivos en Colab (Ronda 2 — condacolab `python_version` inexistente, Miniforge 23.11.0 trae Python 3.12, `mamba install python=3.10` falla en conda env de Colab por google-colab pin), el usuario decidio migrar **ambos tracks a Vast.ai** con persistencia integral en HF Hub. Vacios criticos identificados antes de implementar: (1) container Vast.ai dual-stack TF 2.15 + Ultralytics; (3) workflow notebook -> script Python headless; (4-8) HF Hub completo (repos, streaming logs, W&B vs heartbeat, validacion pre-deploy, checkpoints); (extra) compatibilidad de frameworks y outputs `.tflite`/`.onnx` con Jetson Nano B01 sm_53.

### Decisiones tecnicas vinculantes Ronda 4

**Container Vast.ai:**
- Tag recomendado: **`vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`** (last_pushed 2026-03-26, tag_status active, cuDNN 8.9+ via variante `cudnn-devel`).
- Justificacion: TF 2.15.0 oficial pide CUDA 12.2/cuDNN 8.9 (`tensorflow.org/install/source`), y CUDA 12.4 satisface minor compat con drivers 12.x. Wheel `tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (PyPI subido 2023-11-14) confirmado existente.
- Estructura: **UN container, DOS virtualenvs**: `/opt/venv/tracka` (TF 2.15 + tf-models-official + Pillow 10.4 + protobuf 3.20.3 + grpcio-tools 1.64.1) y `/opt/venv/trackb` (PyTorch 2.x cu124 + Ultralytics 8.4.46 + onnxslim).
- Alternativa robusta: imagen oficial `tensorflow/tensorflow:2.15.0-gpu` (preinstalada) para Track A si la build custom da problemas.

**GPU recomendada: RTX 4090 on-demand** (mediana 0,40 USD/h Vast.ai mayo 2026, spot 0,14-0,31 USD/h). Ada Lovelace sm_89, 24 GB VRAM, 1008 TFLOPS FP16. A100/H100 son desperdicio para SSD MV2 320 + YOLOv8n 416 (modelos pequenos, batch 32-64 entra holgado). Con 1,72 USD del usuario: 4-12 h disponibles, holgado para 1-3 h de training.

**Estructura repo HF Hub: `mitgar14/embebidos-3-models`** (PRIVADO durante desarrollo, public antes de entrega 2026-05-26):
- `track_a/checkpoints/step_NNNNNN/` (TF OD API ckpt-*)
- `track_a/tensorboard_logs/` (tfevents para TensorBoard hosted gratuito)
- `track_a/final/detect_int8.tflite` + `manifest.json` + `pipeline.config`
- `track_b/checkpoints/best.pt`, `last.pt`
- `track_b/final/yolov8n.onnx` + `manifest.json`
- `track_b/wandb_runs/` (logs W&B mirror si se elige stack avanzado)

HF Hub free tier 2026: 100 GB privado total, sin limite de repos privados, 500 GB max por archivo, sin cap de bandwidth. Artefactos totales del proyecto ~250-450 MB, holgado. Xet storage por default desde mayo 2025 (dedup chunks).

**Logging stack:**
- Track A (TF OD API): TensorBoard local en `model_dir/` + `CommitScheduler(every=5)` para push automatico al repo HF. Hub detecta `tfevents` y monta TensorBoard hosted gratis.
- Track B (Ultralytics): `yolo train wandb=True` (W&B nativo, integrado). W&B free 2026 sin cap conocido en runs.
- Heartbeat custom de los notebooks actuales se conserva (logs cada N segundos a stdout).

**Workflow ejecucion: script `run.sh` headless** (no notebooks .ipynb en Vast.ai). Plantilla del agente research-code:

```bash
#!/usr/bin/env bash
set -euo pipefail
REPO_URL="https://github.com/mitgar14/embebidos-3.git"
TRACK="${1:-A}"
export HF_TOKEN="${HF_TOKEN:?requerido}"

git clone --depth=1 "$REPO_URL" /workspace/embebidos-3
cd /workspace/embebidos-3
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [ "$TRACK" = "A" ]; then
  uv venv /opt/venv/tracka --python 3.10
  source /opt/venv/tracka/bin/activate
  uv pip install tensorflow==2.15.0 tf-models-official==2.15.0 \
    Pillow==10.4.0 protobuf==3.20.3 grpcio-tools==1.64.1 huggingface_hub
  git clone https://github.com/tensorflow/models.git /workspace/tf_models
  cd /workspace/tf_models && git checkout 9cafa3d150
  cd /workspace/embebidos-3
  PYTHONPATH=/workspace/tf_models python train_track_a.py
else
  uv venv /opt/venv/trackb --python 3.10
  source /opt/venv/trackb/bin/activate
  uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
  uv pip install ultralytics==8.4.46 huggingface_hub wandb numpy\<2.0
  python train_track_b.py
fi

# Auto-destroy tras upload exitoso
vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY
```

**Conversion notebooks -> scripts**: usar `jupytext --to py train_track_X.ipynb` + papermill para inyectar parametros. Eliminar el bootstrap defensivo de `banner/t_ts/IS_COLAB` (no aplica en headless). Timeout: `papermill --execution-timeout=-1` o `nbconvert --ExecutePreprocessor.timeout=-1` para training largos.

**Persistencia HF Hub - patrones de codigo**:

```python
from huggingface_hub import HfApi, CommitScheduler

api = HfApi(token=os.environ["HF_TOKEN"])
REPO_ID = "mitgar14/embebidos-3-models"

# A) Checkpoint upload non-blocking durante training
def upload_checkpoint(step, ckpt_dir, track="a"):
    api.upload_folder(
        folder_path=ckpt_dir,
        repo_id=REPO_ID,
        path_in_repo=f"track_{track}/checkpoints/step_{step:06d}",
        repo_type="model",
        run_as_future=True,  # NON-BLOCKING
        commit_message=f"Checkpoint step {step}",
    )

# B) TensorBoard logs append-only en background thread
scheduler = CommitScheduler(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path="/workspace/embebidos-3/logs",
    path_in_repo="track_a/tensorboard_logs",
    every=5,  # cada 5 min
)

# C) Resume-friendly para training largos (idempotente, multi-thread)
api.upload_large_folder(
    folder_path="/workspace/model_dir",
    repo_id=REPO_ID,
    repo_type="model",
    cache_dir="./.hf_cache",  # cachea para resume
)
```

### Track A (agentes findings, sintesis Ronda 4)

**research-code (a706e8b2... — Vast.ai dual-stack + HF Hub Python API):**
- Container tag verificado en Docker Hub API: `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` activo 2026-03-26.
- HF Hub patterns: `run_as_future=True`, `CommitScheduler`, `upload_large_folder` (resumible).
- Validacion `.tflite`: `tensorflow.lite.tools.flatbuffer_utils.read_model()` lee op_versions sin ejecutar el modelo. Cross-ref con TF 2.5 limits (CONV_2D max v3, DEPTHWISE_CONV_2D max v3).
- Validacion `.onnx`: `polygraphy run model.onnx --trt --onnxrt --tol 1e-3` compara salidas numericamente. Proxy en cloud con `nvcr.io/nvidia/tensorrt:22.06-py3` (TRT 8.4) — engine final SIEMPRE en Nano.

**research-academic (a512c1db... — validacion cross-platform sm_89/80/90 -> sm_53):**
- TFLite forward compat issues confirmadas: `Cast v2` requiere TF 2.7.0+ (kTfLiteError en 2.5), `BatchMatMul v5+` requiere TF 2.6+. Custom op `TFLite_Detection_PostProcess` requiere `resolver.AddCustom(...)` registro manual en runtime de destino.
- TRT engines NO portables Maxwell: confirmado en docs NVIDIA/TensorRT — "engines are specific to both GPU architecture and TensorRT version". `--versionCompatible` solo TRT 8.6+; `--hardwareCompatibilityLevel=ampere+` excluye matematicamente sm_53. **Engine debe compilarse en el Nano, no en Vast.ai.**
- Magnitud drop PTQ INT8 (con evidencia empirica):
  - YOLOv8n FP16 TRT: 0,0003 absoluto vs FP32 (negligible) — Karimov et al. 2025 arXiv:2508.19600
  - YOLOv8n Static INT8 TRT: 0,0722 absoluto (7,2 pp drop) — mismo paper
  - YOLOv8s Static INT8 TRT: 0,0649 (6,5 pp)
  - SSD MV2 QAT INT8: < 1,5 pp — Jacob et al. 2018 arXiv:1712.05877
  - SSD MV2 PTQ INT8 TFLite Jetson Nano: < 2% accuracy — Rey et al. 2025 arXiv:2502.15737
- Maxwell sm_53 SIN Tensor Cores INT8: TRT 8.2.1 compila INT8 sobre CUDA cores. Speedup FP16/FP32 ~1,2-1,5x (vs 2x en Turing/Ampere). Comportamiento de redondeo puede diferir del baseline GPU moderna.

**research-web (a187456... — Vast.ai + HF Hub + neoclouds comparativa):**
- Vast.ai persistencia: `stop` NO ahorra costo (disco sigue facturando). Volumes son host-local. Solucion: subir todo a HF Hub + `destroy`.
- Auto-destroy desde container: `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` (vars de env inyectadas automaticamente por Vast).
- Workflow tmux por default — `Ctrl+b, d` detach, sesion persistente ante SSH disconnect.
- Comparativa neoclouds (mayo 2026): Vast.ai (recomendado, RTX 4090 0,35-0,50 USD/h on-demand), RunPod (alternativa, Secure Cloud mas estable), Lambda Cloud (solo si presupuesto generoso, no RTX 4090), Paperspace (NO recomendado post-DigitalOcean degradado).
- TFLite issue documentada: `tensorflow/tensorflow#58651` — TFLite 2.10 causa Segfault en runtime 2.5 ARM. Mitigacion: `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]` + `converter.experimental_new_converter = False`.

**research-video (af178c8... — YouTube MCP talks Vast.ai + HF Hub + Jetson Nano):**
- HF Storage Buckets (S3-compatible, deduplicacion Xet) — citado verbatim en video oficial HF mayo 2025: `youtu.be/N7y0OFz98Po?t=37`. Pattern: bucket para checkpoints/logs (mutable), repo Git para artefactos finales (versionado).
- Vast.ai workflow VS Code + SSH: `youtu.be/kcU9U7BYCOs?t=10` (youniss mayo 2025).
- YOLO11 TensorRT Jetson Nano chapter (26 min): `youtu.be/N3adGK66myE?t=3069` (Nicolai Nielsen mar 2025). Unico contenido reciente que cubre TRT engine compile en Nano con familia YOLO actual.
- Validacion pre/post TFLite cross-platform: `youtu.be/OJnaBhCixng?t=2183` (freeCodeCamp). Patron: `tf.lite.Interpreter` + `resize_tensor_input` + `allocate_tensors` sobre test set 10K muestras.
- **Gap confirmado por agente video**: CERO videos cubren validacion sm_89 (Ada training) -> sm_53 (Maxwell deploy). Riesgo tecnico mas alto.

### Track B (busqueda ampliada Ronda 4)

- **discover-13** Vast.ai dual-stack (31 resultados): `vastai/tensorflow:2.19.0-cuda-12.4.1`, `2.16.1-cuda-12.4.1` images existen oficialmente. NO hay `tensorflow:2.15.0` oficial — build custom o usar imagen base + pip install. `vastai/base-image:cuda-12.1.1-cudnn8-devel-ubuntu22.04-py310-2026-03-26` confirmado active.
- **discover-14** HF Hub training (35 resultados): `HFSummaryWriter` (wrapper TensorBoard que push logs automatico), `hf upload` CLI con `--include "*.tfevents.*"`, Storage Buckets recientes (S3-like mutable).
- **discover-15** notebooks headless (19 resultados): patron papermill + jupytext bien establecido. `mwouts/papermill_jupytext` (PyPI v0.0.1) permite ejecutar `.py` con cell markers.
- **discover-16** TFLite forward compat (28 resultados): issues confirmadas `tensorflow/tensorflow#46663` (OperatorCode.BuiltinCode breaking en 2.4), `#80736` (FULLY_CONNECTED v12 falla en TFLite viejos). Existe `upgrade_schema.py` para downgrade schema.
- **discover-17** ONNX -> TRT validation (29 resultados): TRT 8.2.2 Support Matrix oficial. `polygraphy` NO funciona on-device en Jetson Nano (forum NVIDIA #349598). Stack: polygraphy en cloud + trtexec en Nano. INT64 weights downcast automatico con warning.

### Plan operativo Ronda 4 (resumen ejecutable)

1. **Setup Vast.ai (10 min)**: crear instancia RTX 4090 on-demand con imagen `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`. SSH con tmux.
2. **Track A en Vast.ai (1-2 h training)**:
   - `uv venv /opt/venv/tracka --python 3.10`, instalar TF 2.15 stack pineado.
   - Clonar `tensorflow/models` con checkout a SHA `9cafa3d150` (pre-Pillow12 patch).
   - Convertir `train_track_a_ssd.ipynb` -> `train_track_a.py` (jupytext) eliminando bootstrap defensivo.
   - Configurar `CommitScheduler` para tensorboard logs a HF Hub cada 5 min.
   - Ejecutar `python train_track_a.py`. Heartbeat custom + checkpoint upload con `run_as_future=True`.
   - Generar `.tflite` INT8 PTQ con `tf.lite.OpsSet.TFLITE_BUILTINS` y `experimental_new_converter=False` (defensa forward compat).
   - Validacion local: `flatbuffer_utils.read_model()` con check de op_versions vs TF 2.5 max.
   - Upload final a `mitgar14/embebidos-3-models/track_a/final/` + manifest + pipeline.config.
3. **Track B en Vast.ai (30-60 min training + export)**:
   - `uv venv /opt/venv/trackb --python 3.10`, instalar `numpy<2.0` ANTES de Ultralytics, luego `ultralytics==8.4.46 torch==2.1.0+cu121` con index PyTorch.
   - Convertir `train_track_b_yolov8.ipynb` -> `train_track_b.py`. Mantener cascada Roboflow.
   - Ejecutar con `yolo train wandb=True` (W&B nativo) o callback HF Hub.
   - Export ONNX con `opset=11`, `nms=False`, `dynamic=False`, `simplify=True`, `device="cpu"`.
   - Validacion local: `polygraphy run model.onnx --trt --onnxrt --tol 1e-3` con TRT 8.4 (proxy aceptable de 8.2).
   - Upload final a `mitgar14/embebidos-3-models/track_b/final/yolov8n.onnx` + manifest.
4. **Auto-destroy Vast.ai**: `vastai destroy instance $CONTAINER_ID --api-key $CONTAINER_API_KEY` al final del script. Sin instancia detenida -> cero billing.
5. **Deploy en Nano (1-2 h, paralelo a otras tareas)**:
   - `hf download mitgar14/embebidos-3-models track_a/final/detect_int8.tflite` y `track_b/final/yolov8n.onnx`.
   - Track A: cargar `.tflite` con TF 2.5+nv21.8 + custom op resolver `TFLite_Detection_PostProcess` + golden test set 200 imagenes.
   - Track B: compilar engine en Nano con `trtexec --onnx=yolov8n.onnx --fp16 --workspace=2048` (15-45 min). NMS CPU NumPy.
6. **Pre-flight check antes entrega 2026-05-26**: comparar mAP50 cloud baseline vs Nano deploy. Aceptable: caida < 5 pp para Track A, < 0,5 pp para Track B FP16 (segun Karimov 2025).

### Checklist validacion pre-deploy (vinculante)

**`.tflite` INT8 SSD MV2 (TF 2.15 entrenado, TF 2.5 ejecutado en Nano):**
- [ ] Inspeccion flatbuffer: ningun op tiene version > limites TF 2.5 (Cast v1 max, BatchMatMul v4 max, CONV_2D v3 max, DEPTHWISE_CONV_2D v3 max).
- [ ] Custom op `TFLite_Detection_PostProcess` registrado en resolver del runtime Nano.
- [ ] Test load en Nano: `interpreter.allocate_tensors()` sin `kTfLiteError`.
- [ ] Accuracy test: golden set 200 imagenes residuos, mAP50 vs baseline FP32 (drop max 5 pp).
- [ ] Latency: SSD MV2 320 INT8 esperado 30-80 ms/frame en Nano (~12-30 FPS).

**`.onnx` opset 11 YOLOv8n (training cloud, TRT 8.2.1 en Nano):**
- [ ] `assert onnx.checker.check_model(model) is None` (formato valido).
- [ ] `assert model.opset_import[0].version == 11`.
- [ ] `assert model.ir_version <= 10` (TRT 8.2 max IR 10).
- [ ] Ops del grafo ∩ TRT82_UNSUPPORTED = ∅ (script Python del agente research-code).
- [ ] `polygraphy run model.onnx --trt --onnxrt --tol 1e-3` PASA en cloud TRT 8.4 (proxy).
- [ ] NMS aplicado en CPU (NumPy / torchvision.ops.nms), NO usar `EfficientNMS_TRT` plugin.
- [ ] Engine compilado en Nano (NO transferido desde cloud).
- [ ] Validacion mAP en Nano vs baseline PyTorch FP32: drop max 0,5 pp esperado (FP16).

### Fuentes consultadas (Ronda 4, agregadas a tabla principal)

| # | Titulo | URL | Tipo | Track |
|---|--------|-----|------|-------|
| 81 | tensorflow.org/install/source — TF 2.15 CUDA/cuDNN spec | https://www.tensorflow.org/install/source | Doc oficial | research-code |
| 82 | vast-ai/base-image README | https://github.com/vast-ai/base-image/blob/main/README.md | Repo oficial | research-code |
| 83 | vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 Docker Hub | https://hub.docker.com/r/vastai/base-image/tags | Docker Hub API | research-code |
| 84 | huggingface_hub `_tensorboard_logger.HFSummaryWriter` | https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_tensorboard_logger.py | Codigo fuente | discover-14 |
| 85 | huggingface_hub `_upload_large_folder` | https://github.com/huggingface/huggingface_hub/blob/0b55fb46/src/huggingface_hub/_upload_large_folder.py | Codigo fuente | research-code |
| 86 | HF Hub storage limits oficial | https://huggingface.co/docs/hub/storage-limits | Doc oficial | research-web |
| 87 | HF blog Introducing Storage Buckets | https://huggingface.co/blog/storage-buckets | Blog oficial | discover-14 |
| 88 | docs.vast.ai Storage Types | https://docs.vast.ai/documentation/instances/storage/types | Doc oficial | research-web |
| 89 | Vast.ai April 2026 Product Update | https://vast.ai/article/april-2026-product-update | Blog oficial | discover-13 |
| 90 | Vast.ai 2025 Year in Review | https://vast.ai/article/vast-ai-2025-year-in-review | Blog oficial | research-web |
| 91 | tensorflow/tensorflow `flatbuffer_utils.py` v2.15.0 | https://github.com/tensorflow/tensorflow/blob/v2.15.0/tensorflow/lite/tools/flatbuffer_utils.py | Codigo fuente | research-code |
| 92 | onnx/onnx-tensorrt branch 8.2-GA operators.md | https://github.com/onnx/onnx-tensorrt/blob/8.2-GA/docs/operators.md | Doc oficial | research-code |
| 93 | TensorRT 8.2.2 Support Matrix | https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-822/support-matrix/index.html | Doc oficial | discover-17 |
| 94 | issue tensorflow/tensorflow #58651 (TFLite Segfault arm 2.5) | https://github.com/tensorflow/tensorflow/issues/58651 | GitHub Issue | research-web |
| 95 | Karimov et al. 2025 (PTQ INT8 drop YOLO) | https://arxiv.org/abs/2508.19600 | Paper arXiv | research-academic |
| 96 | Rey et al. 2025 (YOLO Jetson Nano analysis) | https://arxiv.org/abs/2502.15737 | Paper MDPI | research-academic |
| 97 | Tariq & Javed 2025 (Small Object Detection YOLO hardware) | https://arxiv.org/abs/2504.09900 | Paper arXiv | research-academic |
| 98 | NVIDIA/TensorRT issue #1538 (EfficientNMS Maxwell broken) | https://github.com/NVIDIA/TensorRT/issues/1538 | GitHub Issue | mnemon-recall |
| 99 | NVIDIA/TensorRT issue #592 (TF->ONNX->TRT Nano JP4.4) | https://github.com/NVIDIA/TensorRT/issues/592 | GitHub Issue | research-web |
| 100 | nteract/papermill 2.7.0 docs | https://papermill.readthedocs.io | Doc oficial | discover-15 |
| 101 | mwouts/papermill_jupytext | https://github.com/mwouts/papermill_jupytext | Repo | discover-15 |
| 102 | jupytext.readthedocs.io CLI | https://jupytext.readthedocs.io/en/stable/using-cli.html | Doc oficial | discover-15 |
| 103 | NVIDIA Developer SSD MobileNet Jetson Nano S3E5 | https://youtu.be/2XMkPW_sIGg | YouTube oficial | research-video |
| 104 | HF Storage Buckets video oficial | https://youtu.be/N7y0OFz98Po | YouTube oficial | research-video |
| 105 | Nicolai Nielsen YOLO11 TensorRT Jetson Nano chapter | https://youtu.be/N3adGK66myE?t=3069 | YouTube | research-video |
| 106 | freeCodeCamp TFLite Edge Devices | https://youtu.be/OJnaBhCixng?t=2183 | YouTube | research-video |
| 107 | youniss Vast.ai GPU rental guide | https://youtu.be/kcU9U7BYCOs | YouTube | research-video |
| 108 | adujardin/tensorrt-compatibility (validacion ONNX vs TRT versiones) | https://github.com/adujardin/tensorrt-compatibility | Repo | research-code |
| 109 | discover-13 — Vast.ai dual-stack | investigaciones/2026-05-12/discover-13-vastai-dual-stack.md | Discovery local | discover.py |
| 110 | discover-14 — HF Hub training pipeline | investigaciones/2026-05-12/discover-14-hf-hub-training.md | Discovery local | discover.py |
| 111 | discover-15 — notebook headless workflows | investigaciones/2026-05-12/discover-15-headless-notebooks.md | Discovery local | discover.py |
| 112 | discover-16 — TFLite forward compat | investigaciones/2026-05-12/discover-16-tflite-fwdcompat.md | Discovery local | discover.py |
| 113 | discover-17 — ONNX/TRT validation | investigaciones/2026-05-12/discover-17-onnx-trt-validation.md | Discovery local | discover.py |

### Gotchas nuevos Ronda 4

12. **TF 2.15 oficial requiere CUDA 12.2 / cuDNN 8.9**, NO 12.1. Usar tag con CUDA 12.4 (minor compat OK) o construir con `pip install tensorflow==2.15.0` sobre base 12.4.
13. **Vast.ai `stop` NO ahorra costo de almacenamiento** (disco sigue facturando). Solo `destroy` corta billing. Subir todo a HF Hub PRIMERO.
14. **Container storage NO portable entre hosts** en Vast.ai. Volumes son host-local. Persistencia robusta = HF Hub.
15. **TRT engines NO portables Maxwell**: confirmado en docs oficiales NVIDIA/TensorRT. `--versionCompatible` solo 8.6+, no aplica a sm_53. Engine SIEMPRE compilado en Nano.
16. **Maxwell sm_53 SIN Tensor Cores INT8**: speedup FP16/FP32 modesto (1,2-1,5x). Comportamiento de redondeo INT8 puede diferir del baseline GPU moderna — gap empirico no resuelto en literatura.
17. **`Cast v2` (TF 2.7+) y `BatchMatMul v5+` (TF 2.6+)** emitidos por TFLite Converter 2.15 fallan en runtime TF 2.5 con `kTfLiteError`. Mitigacion: `converter.experimental_new_converter = False` + verificar flatbuffer post-conversion.
18. **`TFLite_Detection_PostProcess` es custom op**: requiere `resolver.AddCustom("TFLite_Detection_PostProcess", ...)` registro manual en el runtime Nano. Sin registro -> falla en `AllocateTensors()`.
19. **`polygraphy` NO funciona on-device en Jetson Nano** (forum NVIDIA #349598). Validacion siempre en cloud, deploy con `trtexec` en Nano.
20. **HF Hub repos privados consumen el cupo de 100 GB** del free tier 2026. Artefactos del proyecto (~250-450 MB) holgados, pero tener en cuenta si se acumulan experimentos.
21. **TFLite 2.10+ causa Segfault en runtime 2.5 ARM** (issue #58651). Sin garantia oficial de backcompat. Validacion empirica obligatoria en Nano antes entrega.

### Gaps de evidencia Ronda 4

1. **Drop INT8 especifico en Maxwell sm_53 SIN Tensor Cores**: cero papers publicados miden esto. Toda la literatura PTQ usa GPU moderna (sm_75+). Riesgo: el drop empirico en Nano puede exceder los 7,2 pp de Karimov por diferencias de redondeo en CUDA cores.
2. **Soporte explicito de `EfficientNMS_TRT` plugin en sm_53**: NVIDIA/TensorRT docs no especifican. Workaround conservador: `nms=False` en export + NMS CPU.
3. **`Cast v2` emitido por modelo SSD MV2 especifico**: depende del converter path. Verificar post-export con script de inspeccion flatbuffer. No asumir a priori.
4. **W&B free tier 2026 storage limit**: no confirmado oficialmente esta sesion. Historicamente 100 GB. Verificar antes de subir checkpoints grandes.
5. **`vastai destroy --api-key` flag exacto**: verificar con `vastai destroy --help` in-container — la CLI versiona con frecuencia.
6. **Reddit r/MachineLearning bloqueado por Exa**: datos de UX vienen de blogs analiticos, no threads directos. Confianza baja en reportes de comunidad.
7. **TRT container `nvcr.io/nvidia/tensorrt:22.06-py3` versiones TRT exactas**: documentado como 8.4.0 pero requiere `dpkg -l | grep tensorrt` in-container para verificar.
8. **Validacion sm_89 -> sm_53 cross-architecture**: cero videos YouTube cubren este flow (confirmado por research-video). Solo documentacion oficial NVIDIA + foros developer.nvidia.com.

### Decision tecnica Ronda 4

- Plataforma: **Vast.ai con RTX 4090 on-demand** (mediana 0,40 USD/h, 1-3 h training).
- Container: **`vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`** con dos virtualenvs en mismo container.
- Repo HF: **`mitgar14/embebidos-3-models`** privado, subcarpetas `track_a/` `track_b/`, public antes 2026-05-26.
- Persistencia: **HF Hub con `CommitScheduler` + `upload_folder(run_as_future=True)` + `upload_large_folder`** (resumible).
- Logging: **W&B nativo en Track B**, TensorBoard en HF Hub via Storage Buckets en Track A.
- Workflow: **jupytext + papermill** convertir `.ipynb` -> `.py` headless. Eliminar bootstrap defensivo (no aplica fuera Colab).
- Engine TRT: **SIEMPRE compilado en Nano**, nunca transferido desde Vast.ai.
- Validacion pre-deploy: **checklist Ronda 4** vinculante para ambos tracks antes de cualquier deploy al Nano.
- Auto-destroy Vast.ai: `vastai destroy $CONTAINER_ID --api-key $CONTAINER_API_KEY` al final del script.

Vinculante para implementacion. Pipeline end-to-end (Vast.ai training -> HF Hub artefactos -> Nano deploy) cubierto por defensa en cada capa: container CUDA 12.4 + cuDNN 8.9, virtualenvs separados, pin de Pillow/protobuf/numpy, HF Hub upload non-blocking, validacion flatbuffer + polygraphy, NMS CPU, engine TRT in-situ.
