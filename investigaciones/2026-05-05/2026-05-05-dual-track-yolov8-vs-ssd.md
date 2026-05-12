# Investigación: Dual-track YOLOv8n TensorRT FP16 vs SSD MobileNet v2 TFLite INT8

> **Decisión arquitectónica:** ejecutar AMBOS pipelines en paralelo, comparar empíricamente, elegir según frontera de Pareto accuracy/latencia/energía.
> **Plataforma:** Jetson Nano B01 4 GB con JetPack 4.6.x (TensorRT 8.0.1 / 8.2.1, CUDA 10.2, Python 3.6.9).
> **Dataset común:** versión Roboflow versionada con splits idénticos exportados a YOLOv8 (`yolov8`) y a TFRecord (`tfrecord`).
> **Clases MVP:** paper, glass, plastic (3 clases). Cardboard NO se fusiona con paper.

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-05 | alto | Datasets de detección, domain adaptation, anotación (doc previo) |
| 2 | 2026-05-05 | alto | Dual-track: pipeline TRT en Nano, Roboflow multi-format, harness comparativo |

---

## Síntesis ejecutiva

**Hallazgo central:** ningún paper publicado compara directamente SSD MobileNet v2 FPNLite TFLite INT8 vs YOLOv8n TensorRT FP16 sobre Jetson Nano B01 con dataset custom de 3 clases. La comparativa que entregaremos es contribución empírica original — buen punto para el informe IEEE.

**Inversión de la regla común:** TFLite INT8 NO siempre gana a TensorRT FP16 en Jetson Nano. La GPU Maxwell B01 carece de tensor cores INT8 nativos (introducidos en Volta). El camino óptimo en Maxwell es FP16 vía TensorRT, que usa unidades CUDA half-precision. INT8 va por XNNPACK + NEON SIMD en CPU Cortex-A57. Resultado documentado: YOLOv8n FP16 GPU (~16-19 ms) supera a SSD INT8 CPU (~60-90 ms) — un modelo arquitectónicamente más complejo gana porque el backend explota mejor el hardware.

**Roboflow es la pieza que une ambos tracks:** una versión inmutable del dataset se exporta a YOLOv8 y TFRecord con splits idénticos. El número de versión es el identificador citable para reproducibilidad (no hay MD5 público pero el zip se cachea por (versión, formato), garantizando reproducibilidad binaria).

**EfficientNMS_TRT está roto en Maxwell.** Bug confirmado en TRT 8.0.1 issue NVIDIA/TensorRT#1538. Solución validada: NMS en CPU con NumPy puro (overhead 1-3 ms). No usar plugin TRT para NMS.

---

## (1) Pipeline Track A — SSD MobileNet v2 FPNLite TFLite INT8

### 1.1 Workflow

```
┌─ HOST x86 (Python 3.10+) ───────────────────────────────────────┐
│  Roboflow project waste-3class                                   │
│    └─ Version N: 70/15/15 stratified split + augmentations      │
│       └─ Export "tfrecord" → train.tfrecord, valid.tfrecord,    │
│          test.tfrecord, _label_map.pbtxt                        │
│                                                                  │
│  TF Object Detection API (tensorflow/models)                    │
│    pipeline.config: ssd_mobilenet_v2_fpnlite_320x320_coco17    │
│      ├─ num_classes: 3                                          │
│      ├─ image_resizer.fixed_shape_resizer: 320                  │
│      ├─ quantization: { delay: 2000 }   ← QAT activo           │
│      └─ paths: TFRecord + label_map del export Roboflow         │
│                                                                  │
│  python model_main_tf2.py --pipeline_config_path=...            │
│    → checkpoint cuantizado QAT-ready                            │
│                                                                  │
│  python export_tflite_ssd_graph.py + tflite_convert             │
│    --inference_type=QUANTIZED_UINT8                             │
│    --representative_dataset (200-300 muestras val)              │
│    → detect_int8.tflite (~3 MB)                                 │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼ (scp)
┌─ JETSON NANO B01 (Python 3.6.9, JetPack 4.6.x) ──────────────────┐
│  tf.lite.Interpreter(model_path="detect_int8.tflite",            │
│                      num_threads=4)                              │
│    → invoke() libera GIL                                         │
│    → output incluye TFLite_Detection_PostProcess (NMS embebido)  │
└──────────────────────────────────────────────────────────────────┘
```

### 1.2 Latencia esperada

| Configuración | Latencia/frame | FPS |
|---|---|---|
| SSD MobileNet v2 FPNLite 320×320 INT8, 1 hilo | ~168 ms | 6 |
| Idem, 4 hilos XNNPACK | **60-90 ms** | **11-16** |
| EfficientDet-Lite0 320×320 INT8, 4 hilos | 55-70 ms | 14-18 |

Fuentes: NobuoTsukamoto/benchmarks, Tobiasz et al. 2023 (arXiv 2306.12093), Zagitov et al. 2024.

### 1.3 Conflictos

- **`num_threads > 1` con XNNPACK** puede degradar en TFLite Python 3.6 antiguo (issues TF #52076, #53146). Validar empíricamente; fallback a backend default si XNNPACK no escala.
- **Custom op `TFLite_Detection_PostProcess`** requiere build TFLite con custom ops habilitados. El wheel oficial NVIDIA `2.5.0+nv21.8` lo incluye. Wheel comunitario Qengineering 2.7.0 también.

---

## (2) Pipeline Track B — YOLOv8n TensorRT FP16

### 2.1 Workflow (3 etapas obligatorias)

**Etapa A — Export ONNX en host x86 (NO en el Nano)**

Ultralytics ≥8.0.0 exporta opset 17 por defecto; **TensorRT 8.0/8.2 solo lee hasta opset 13**. Forzar opset 11:

```bash
# Opción 1 (recomendada): triple-Mu/YOLOv8-TensorRT con head custom EfficientNMS
git clone https://github.com/triple-Mu/YOLOv8-TensorRT.git
cd YOLOv8-TensorRT
pip install ultralytics onnx onnxsim

python3 export-det.py \
  --weights yolov8n_waste.pt \
  --iou-thres 0.65 \
  --conf-thres 0.25 \
  --topk 100 \
  --opset 11 \
  --sim \
  --input-shape 1 3 416 416 \
  --device cpu

# Opción 2 (más portable, NMS en CPU): export estándar
yolo export model=yolov8n_waste.pt format=onnx opset=11 imgsz=416 simplify=True
```

**Etapa B — Compilar engine en el Nano (OBLIGATORIO en device target)**

TensorRT engines son específicos de arquitectura GPU + versión TRT. Engine compilado en x86 NO funciona en Nano.

```bash
# scp yolov8n_waste.onnx jetson@nano:/home/jetson/models/
# En el Nano:
sudo systemctl stop lightdm                         # liberar memoria de X11
sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
export PATH=$PATH:/usr/src/tensorrt/bin

trtexec \
  --onnx=/home/jetson/models/yolov8n_waste.onnx \
  --saveEngine=/home/jetson/models/yolov8n_waste_fp16.engine \
  --fp16 \
  --workspace=1024 \
  --verbose
```

Tiempo: **15-45 min** para YOLOv8n a 416×416. Memoria pico: hasta 3,5 GB unificada durante compilación → cerrar X11 antes.

**Importante:** `--workspace=1024` (1 GiB MB en `trtexec`) es el valor seguro. Issue ultralytics/ultralytics#14751 documenta `Killed` por OOM con workspace mayor.

**Etapa C — Inferencia con wrapper Python (pycuda + tensorrt)**

```python
import ctypes, numpy as np
import pycuda.autoinit, pycuda.driver as cuda
import tensorrt as trt
import cv2

class YoloV8TRT:
    def __init__(self, engine_path, input_shape=(1, 3, 416, 416)):
        self.input_shape = input_shape
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = [], [], [], cuda.Stream()
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            (self.inputs if self.engine.binding_is_input(binding) else self.outputs).append(
                {"host": host_mem, "device": device_mem})

    def detect(self, bgr, conf_thr=0.25, iou_thr=0.45):
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2,0,1)[None])
        np.copyto(self.inputs[0]["host"], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for o in self.outputs:
            cuda.memcpy_dtoh_async(o["host"], o["device"], self.stream)
        self.stream.synchronize()
        # Decode + NMS en NumPy (overhead ~1-3 ms)
        return _decode_yolov8_nms_numpy(self.outputs[0]["host"], bgr.shape, conf_thr, iou_thr)
```

### 2.2 Latencia esperada

| Configuración | Latencia/frame | FPS | Fuente |
|---|---|---|---|
| YOLOv8n 416×416 FP16 (Maxwell GPU, solo forward) | 19 ms | ~52 | Qengineering, Nano B01 |
| YOLOv8n 416×416 FP16 + NMS NumPy + preprocess | 25-35 ms | 28-40 | Estimado realista |
| YOLOv8n 640×640 FP16 | 33 ms | 30 | Scientific Reports 2024 |
| MobileNetV2-SSD 320×320 FP16 (referencia Tsukamoto) | 37,99 ms | ~26 | NobuoTsukamoto Nano B01 JP4.6.1 |

### 2.3 Conflictos críticos

| Issue | Impacto | Workaround |
|---|---|---|
| Opset 17+ rechazado por parser TRT 8.0/8.2 | Engine no compila | Forzar `opset=11` en export |
| EfficientNMS_TRT crash en Maxwell (TRT 8.0.1, issue NVIDIA/TensorRT#1538) | `kSTATUS_SUCCESS assertion failed` | NMS en CPU NumPy (overhead 1-3 ms) |
| `Killed` durante `trtexec` con workspace alto | Compilación termina antes de finalizar | `--workspace=1024`, cerrar X11 antes |
| Engine compilado en x86 no corre en Maxwell | "engine mismatch device" | Compilar siempre en device target |
| Detecciones duplicadas FP16 (issue Linaom1214 #112) | Falsos positivos múltiples bbox | Subir `conf_thr` a 0,3+ |
| Python 3.6.9 incompatible con bindings modernos (f-strings, walrus) | ImportError | Docker `mwlvdev/jetson-nano:bionic-cp38-cuda10.2-TRT` o usar Python nativo del JetPack |
| Engine no portable entre versiones TRT/JetPack | Re-compilar por device | Documentar para informe IEEE |

---

## (3) Roboflow para pipeline dual

### 3.1 Workflow consolidado

```
1. Subir arshnoor7389 al workspace (Python SDK, batch 4,3 GB):
   workspace.upload_dataset(dataset_path="./arshnoor_raw",
                            project_id="waste-3class",
                            project_type="object-detection",
                            project_license="MIT",
                            num_workers=10)

2. Crear Version N en UI:
   ├─ Preprocessing > Modify Classes:
   │    ├─ Delete: cardboard, miscellaneous, organic, metal
   │    └─ Keep: paper, glass, plastic
   │  + Filter Null (descarta imágenes que quedan vacías)
   │  + Auto-Orient + Resize (640×640 universal o 320×320)
   │
   ├─ Augmentations (3x en Public Plan):
   │    ├─ Flip: solo Horizontal (NO Vertical, residuos sobre banda)
   │    ├─ Rotation: ±15°
   │    ├─ Brightness/Exposure/Hue/Saturation: ±25%
   │    ├─ Noise (Salt & Pepper): 5%
   │    ├─ Blur: 1-2 px
   │    └─ Motion Blur: leve
   │
   └─ Generate Version → Version 1 inmutable

3. Export dual desde MISMA versión:
   project.version(1).download("yolov8",   location="./ds_yolo")
   project.version(1).download("tfrecord", location="./ds_tfr")
   # Splits idénticos garantizados (atributos de la versión, no del formato)
```

### 3.2 Augmentations Roboflow vs literatura VisDA 2022

| Augmentation | Roboflow Basic (free) | Validada |
|---|---|---|
| HorizontalFlip | sí (Flip) | sí |
| Rotation libre | sí | sí |
| ColorJitter (B/S/H/V) | sí (Brightness, Hue, Saturation, Exposure) | sí |
| GaussianNoise | parcial (Salt & Pepper) | sí |
| Blur | sí | sí |
| MotionBlur | sí | sí |
| Cutout/CoarseDropout | NO en free (Enhanced premium) | sí |
| Mosaic | NO (solo en YOLOv8 training nativo) | sí |
| PerspectiveWarp | NO nativo (Shear como proxy) | sí |

**Vacío:** Mosaic y PerspectiveWarp reales no están en Roboflow Basic. Compensar:
- Mosaic: Ultralytics YOLOv8 lo aplica online durante training (`mosaic=1.0` en `model.train(...)`).
- PerspectiveWarp real: aplicar en preprocesamiento offline post-export, antes de entrenar SSD.

### 3.3 Límites Public Plan relevantes

| Recurso | Límite |
|---|---|
| Workspace | 1 por usuario, 250.000 imágenes |
| Augmentation máxima | 3x |
| Auto Label (Grounding DINO) | 1.000 imágenes/job (beta), múltiples jobs sí |
| Datos privados | NO (todo público en Universe) |
| Múltiples exports misma versión | Sin límite (zip cacheado) |
| Créditos mensuales | USD 60/mes (refresh mensual) |

### 3.4 Reproducibilidad para informe IEEE

```
Cita: Dataset waste-3class v1, Roboflow Universe.
URL:  https://universe.roboflow.com/{workspace}/waste-3class/dataset/1
Generated: 2026-05-XX
Splits: 70/15/15 estratificados
Augmentations: 3x con [Flip H, Rot ±15°, B/S/H/V, Noise 5%, Blur, MotionBlur leve]
Exports: yolov8 (formato YOLO), tfrecord (TF OD API)
```

La inmutabilidad de la versión sustituye al MD5. El zip de cada (versión, formato) se cachea y re-sirve idéntico.

### 3.5 Gotchas reportados

- **`data.yaml` con clases fantasma** (issue roboflow-python#88): validar `nc` y `names` post-descarga. Si discrepa, corregir manualmente.
- **`overwrite=True` por defecto en `.download()`** (issue #108): pasar siempre `overwrite=False`.
- **Datasets >5.000 imágenes pueden tomar hasta 1 h en generar export.** No interrumpir.
- **Coordenadas YOLO fuera de rango [0,1]** (foro discuss.roboflow.com): script de validación pre-training:

```python
import glob
errors = []
for lf in glob.glob(f"{ds.location}/train/labels/*.txt"):
    with open(lf) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: errors.append(lf); break
            if any(float(x) < 0 or float(x) > 1 for x in parts[1:]): errors.append(lf); break
print(f"{len(errors)} archivos con coords problemáticas")
```

---

## (4) Harness comparativo dual-track

### 4.1 Métricas obligatorias (estilo IEEE)

| Métrica | Definición | Herramienta de medición |
|---|---|---|
| Latencia p50/p95/p99 (ms) | Percentiles de inferencia + NMS, sin captura | `time.perf_counter()` × 200 iter |
| FPS sostenido | Promedio sobre 200 frames consecutivos | Bucle real (no batch) |
| mAP@50 global | IoU≥0,5 sobre test set | `pycocotools` o `ultralytics val` |
| AP@50 por clase | Por paper, glass, plastic | Ídem |
| RAM peak (MB) | Memoria pico del proceso | `tegrastats` columna RAM |
| GPU mem peak (MB) | Solo Track B | `tegrastats` columna IRAM |
| Potencia media (W) | Durante inferencia sostenida | `tegrastats` columna POM_5V_SYS |
| Energía/inferencia (mJ) | Potencia × latencia p50 | Calculado |
| Temperatura pico (°C) | Stress test 5 min | `tegrastats` thermal |
| Portabilidad engine | Sí/No entre devices | Documentado |

### 4.2 Protocolo de medición

```
1. Fijar power mode: sudo nvpmodel -m 0   # MAXN 10 W
   sudo jetson_clocks
2. Esperar steady-state térmico (~10 min de operación)
3. Warmup: 50 inferencias antes de medir
4. Medición: 200 inferencias para estabilidad estadística
5. tegrastats --interval 500 --logfile bench.log &
6. Reportar p50/p95/p99 + std + condiciones (T amb, modo power)
```

### 4.3 Diseño 2×2 a igualdad de input

Para evitar el sesgo "modelo a su preset óptimo":

| Modelo | Input 320×320 | Input 416×416 |
|---|---|---|
| **SSD MobileNet v2 FPNLite TFLite INT8** | Track A primario | Track A extendido |
| **YOLOv8n TensorRT FP16** | Track B reducido | Track B primario |

Las 4 celdas se entrenan y evalúan; la tabla resultante muestra trade-offs claros. Para el informe IEEE basta la diagonal (presets óptimos), pero medir las 4 elimina dudas.

### 4.4 Plantilla de tabla benchmark

```
Tabla I — RESULTADOS BENCHMARK DUAL-TRACK
Jetson Nano B01 4GB / JetPack 4.6.x / nvpmodel=MAXN

+---------------------------+----------------+-----------------+
| Métrica                   | Track A        | Track B         |
|                           | SSD-MV2-FPNLite| YOLOv8n         |
|                           | 320×320 INT8   | 416×416 FP16    |
|                           | TFLite/CPU     | TensorRT/GPU    |
+---------------------------+----------------+-----------------+
| Parámetros (M)            | 3,27           | 3,16            |
| Tamaño modelo (MB)        | ~3,0           | ~6,2            |
| Latencia p50 (ms)         | XX,X ± σ       | XX,X ± σ        |
| Latencia p95 (ms)         | XX,X           | XX,X            |
| FPS sostenido             | XX             | XX              |
| mAP@50 global (%)         | XX,X           | XX,X            |
| AP@50 — Paper (%)         | XX,X           | XX,X            |
| AP@50 — Glass (%)         | XX,X           | XX,X            |
| AP@50 — Plastic (%)       | XX,X           | XX,X            |
| RAM peak (MB)             | XXX            | XXX             |
| GPU mem peak (MB)         | N/A            | XXX             |
| Potencia media (W)        | X,X            | X,X             |
| Energía/inferencia (mJ)   | XXX            | XXX             |
| Temp peak 5 min (°C)      | XX             | XX              |
| Engine portátil           | Sí (ARM)       | No (TRT-locked) |
+---------------------------+----------------+-----------------+
```

### 4.5 Frontera de Pareto

Si AMBOS cumplen ≥10 FPS, el criterio de selección se mueve a multi-objetivo:
1. mAP@50 global más alto
2. Menor energía/inferencia (mJ)
3. Menor latencia p95
4. Reproducibilidad/portabilidad (Track A gana aquí)

La presentación final en el informe es un scatter `latencia vs mAP@50` con la frontera Pareto trazada.

---

## (5) Implementaciones de referencia validadas

| Repo | Track | Stars | Notas |
|---|---|---:|---|
| `triple-Mu/YOLOv8-TensorRT` | B | 1.763 | `export-det.py` con head NMS embebido (opset 11). Doc Jetson |
| `Qengineering/YoloV8-TensorRT-Jetson_Nano` (branch `tensorrt8`) | B | 44 | C++ benchmark 19 FPS confirmado en Nano B01 |
| `Linaom1214/TensorRT-For-YOLO-Series` | B | 1.150 | Wrapper Python `tensorrt` + `cuda-python`, NMS NumPy |
| `marcoslucianops/DeepStream-Yolo` | B+ | 2.006 | Solo si se quiere DeepStream pipeline (overhead) |
| `ecoCrafters/waste-detection` | A | 1 | SSD MV2 FPNLite 320×320 + TF OD API + 7 clases custom |
| `tensorflow/models` | A | 77.667 | TF OD API official, configs SSD MV2 FPNLite QAT |
| `google-ai-edge/mediapipe` | A (alt) | 35.066 | MediaPipe Model Maker `MOBILENET_V2_I320` con QAT nativo |
| `NobuoTsukamoto/benchmarks` | A+B | 23 | Benchmarks confirmados Nano B01 JP4.6.1 |
| `roboflow/roboflow-python` | unifica | — | SDK Python para upload/download/version |
| `autodistill/autodistill-grounded-sam-2` | datos target | 136 | Auto-labeling Florence-2 + SAM 2 |

---

## Conceptos clave

- **Frontera de Pareto:** cuando ambos modelos cumplen restricción mínima (latencia ≥10 FPS), la decisión se desplaza a optimización multi-objetivo (accuracy + energía + portabilidad).
- **Iso-protocolo, no iso-backend:** comparación justa requiere mismas condiciones de medición (warmup, # iter, power mode, T amb), no mismo backend.
- **Maxwell sin tensor cores INT8:** consecuencia operacional crítica — TFLite INT8 va por CPU, FP16 va por GPU. Decisión técnica revierte intuición común "INT8 siempre mejor que FP16".
- **Engine TRT no portable:** el `.engine` se rompe entre versiones TRT/CUDA/JetPack/arquitectura GPU. TFLite es portable. Argumento de reproducibilidad.
- **EfficientNMS_TRT en Maxwell:** roto en TRT 8.0.1 (issue #1538), uso NMS NumPy.
- **Roboflow Version inmutable:** sustituye al MD5 como identificador citable.
- **Splits = atributos de versión:** export multi-formato desde misma versión = splits idénticos por construcción.

---

## Recomendaciones operacionales

1. **Configurar Roboflow workspace ya.** Subir arshnoor7389 (4,3 GB, ~30-60 min upload), crear versión con Modify Classes (delete cardboard/miscellaneous/organic/metal, keep paper/glass/plastic) + Filter Null + augmentations Basic.
2. **Entrenar Track A y Track B en paralelo** en dos cuentas Colab T4 (free tier 12h/cuenta) o Colab + Kaggle GPU.
3. **Compilar engine TRT en el Nano**, no en host. 15-45 min con `--workspace=1024` y X11 cerrado.
4. **NMS en CPU NumPy para Track B**, no usar EfficientNMS_TRT en TRT 8.0.1.
5. **Harness único `bench.py`** con interfaz `Detector` agnóstica de backend, mide ambas métricas en mismas condiciones.
6. **Plan C documentado:** si AMBOS no llegan a ≥10 FPS sostenidos → EfficientDet-Lite0 320×320 INT8 (alt Track A) o YOLOv5n FP16 (alt Track B, opcional).

---

## Fuentes consultadas — Ronda 2 (acumulado con ronda 1)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 45 | Benchmarking Deep Learning Models for Object Detection on Edge Devices (Alqahtani et al.) | https://arxiv.org/abs/2409.16808 | Paper | 2 |
| 46 | Benchmarking Deep Learning Models on Jetson Nano (Swaminathan et al.) | https://arxiv.org/abs/2406.17749 | Paper | 2 |
| 47 | Small Object Detection with YOLO (Tariq & Javed) | https://arxiv.org/abs/2504.09900 | Paper | 2 |
| 48 | DWaste: Greener AI for Waste Sorting | https://arxiv.org/abs/2510.18513 | Paper | 1+2 |
| 49 | NobuoTsukamoto/benchmarks (Jetson Nano JP4.6.1) | https://github.com/NobuoTsukamoto/benchmarks | Repo | 2 |
| 50 | triple-Mu/YOLOv8-TensorRT | https://github.com/triple-Mu/YOLOv8-TensorRT | Repo | 2 |
| 51 | Qengineering/YoloV8-TensorRT-Jetson_Nano | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo | 2 |
| 52 | Linaom1214/TensorRT-For-YOLO-Series | https://github.com/Linaom1214/TensorRT-For-YOLO-Series | Repo | 2 |
| 53 | NVIDIA/TensorRT issue #1538 (EfficientNMS_TRT crash Maxwell) | https://github.com/NVIDIA/TensorRT/issues/1538 | Issue | 2 |
| 54 | ultralytics/ultralytics issue #14751 (workspace OOM) | https://github.com/ultralytics/ultralytics/issues/14751 | Issue | 2 |
| 55 | Roboflow Docs: Export a Dataset Version | https://docs.roboflow.com/datasets/dataset-versions/exporting-data | Doc oficial | 2 |
| 56 | Roboflow Docs: Modify Classes (Preprocessing) | https://docs.roboflow.com/datasets/dataset-versions/image-preprocessing | Doc oficial | 2 |
| 57 | Roboflow Docs: Image Augmentation | https://docs.roboflow.com/datasets/dataset-versions/image-augmentation | Doc oficial | 2 |
| 58 | Roboflow Pricing | https://roboflow.com/pricing | Página oficial | 2 |
| 59 | Roboflow CLI Commands | https://github.com/roboflow/roboflow-python/blob/74885a27/CLI-COMMANDS.md | Doc oficial | 2 |
| 60 | roboflow-python issue #88 (clases fantasma) | https://github.com/roboflow/roboflow-python/issues/88 | Issue | 2 |
| 61 | roboflow-python issue #108 (overwrite default) | https://github.com/roboflow/roboflow-python/issues/108 | Issue | 2 |
