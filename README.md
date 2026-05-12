# embebidos-3 — clasificador de residuos en Jetson Nano

> **MVP académico** para curso de IA en sistemas embebidos (UAO).
> **Entrega:** 2026-05-26.
> **Tarea:** detección de objetos en tiempo real sobre banda transportadora — 3 clases: `paper`, `glass`, `plastic`.
> **Hardware target:** Jetson Nano Developer Kit 4 GB rev. B01, JetPack 4.6.x (Ubuntu 18.04, Python 3.6.9, CUDA 10.2, TensorRT 8.0.1/8.2.1, GPU Maxwell 128 CUDA cores **sin tensor cores INT8**).
> **Actuación:** 3 servomotores SG90 vía PCA9685 (I2C) — rampas deflectoras por clase.
> **Cámara:** USB UVC 720p/30 fps montada en diagonal sobre la banda.

---

## 1. Reframing dual-track (lectura obligatoria antes de tocar código)

La decisión arquitectónica central NO es "INT8 vs FP16". Es **"backend hardware compatible con la GPU Maxwell del Nano B01"**:

| Track | Modelo | Resolución | Cuantización | Backend en Nano | Razón |
|-------|--------|-----------:|--------------|-----------------|-------|
| **A** | SSD MobileNet v2 FPNLite | 320×320 | TFLite **INT8** (QAT) | CPU Cortex-A57 + XNNPACK + NEON SIMD | Maxwell carece de tensor cores INT8 (introducidos en Turing/Ampere); INT8 acelera vía instrucciones SIMD packed en CPU ARM. |
| **B** | YOLOv8n | 416×416 | ONNX → TensorRT **FP16** | GPU Maxwell vía CUDA half-precision | FP16 es el sweet spot de Maxwell. Qengineering verbatim: *"INT8 no aumenta FPS y degrada mAP significativamente"* en Nano. |

Resultado documentado de literatura comparable (Nature Sci Rep oct 2024, DOI 10.1038/s41598-024-74798-3, Tabla 4):

- YOLOv8n + TensorRT FP16 Jetson Nano: **30 FPS @ 416×416** vs 24 FPS @ 640×640.
- SSD MobileNet v2 FPNLite 320×320 INT8 + XNNPACK 4 hilos: ~11-16 FPS extrapolado de NobuoTsukamoto/benchmarks + Tobiasz 2023.

**No es una guerra**: ambos tracks corren en paralelo. La elección final se hace por benchmark empírico en el Nano contra threshold ≥10 FPS sostenidos con servos+I2C concurrentes.

---

## 2. Estado actual

| Componente | Estado | Notas |
|------------|--------|-------|
| Dataset combinado (Kaggle + Roboflow Universe) subido a Roboflow `embebidos3/waste-3class-lwld8` | ✅ Hecho | 11.558 imágenes / 13.873 bbox tras dedup phash + filtro 3 clases |
| 4 investigaciones documentadas en `investigaciones/` | ✅ Hecho | 95+ fuentes, ronda 2 con filtro hardware Jetson Nano-class |
| Spec Generate Version 1 (Roboflow UI) | ✅ Documentado | Ver `scripts/03_generate_roboflow_v1.md` |
| Generate Version 1-A (320×320 tfrecord) y 1-B (416×416 yolov8) en Roboflow UI | ✅ Hecho | Generadas en Roboflow UI 2026-05-11 |
| Notebooks Track A (Colab) y Track B (Kaggle) | ✅ Listos | `notebooks/train_track_a_ssd.py`, `notebooks/train_track_b_yolov8.py` |
| Bench harness en Nano | ✅ Esqueleto | `scripts/bench_jetson.py` |
| Pipeline runtime (captura+infer+I2C 3 hilos) | ⏳ Pendiente | Spec en `investigaciones/2026-05-05-arquitectura-software-jetson-nano.md` |

---

## 3. Estructura del repositorio

```
embebidos-3/
├── README.md                    ← este archivo
├── pyproject.toml               ← uv project (Python ≥3.10, deps host: roboflow, kaggle, imagehash, pillow, pyyaml)
├── main.py                      ← stub
│
├── investigaciones/             ← 4 .md con 95+ fuentes citadas (input para informe IEEE)
│   ├── 2026-05-05-arquitectura-software-jetson-nano.md
│   ├── 2026-05-05-datasets-deteccion-residuos.md
│   ├── 2026-05-05-dual-track-yolov8-vs-ssd.md
│   └── 2026-05-05-preprocessing-roboflow.md
│
├── prototipos/                  ← (vacío, reservado para experimentación local)
│
├── scripts/                     ← utilidades host + playbooks
│   ├── 03_generate_roboflow_v1.md   ← playbook copy-paste para Roboflow UI
│   └── bench_jetson.py              ← harness FPS/latencia/RAM en Nano
│
├── notebooks/                   ← entrenamiento en cloud (formato celular # %%)
│   ├── train_track_a_ssd.py         ← Colab T4: TF OD API + QAT → TFLite INT8
│   └── train_track_b_yolov8.py      ← Kaggle T4: Ultralytics → ONNX opset 11 → (TRT FP16 en Nano)
│
└── uv.lock
```

---

## 4. Investigaciones (input directo para el informe IEEE)

Las 4 investigaciones combinadas constituyen el sustento bibliográfico del proyecto. Cada una tiene historial de rondas y tabla acumulativa de fuentes.

| Documento | Foco | Decisiones que ancla |
|-----------|------|----------------------|
| [`arquitectura-software-jetson-nano.md`](2026-05-05-arquitectura-software-jetson-nano.md) | Stack JetPack 4.6.x, concurrencia 3 hilos, GIL libre en `Interpreter.invoke()`, regla del medio núcleo (Chakraborty 2025) | Patrón producer-consumer + queue drop-oldest; servos en hilo daemon aislado |
| [`datasets-deteccion-residuos.md`](2026-05-05-datasets-deteccion-residuos.md) | 12 datasets evaluados, auto-labeling Grounded-SAM2, domain adaptation industrial | Combinación Kaggle arshnoor7389 + Roboflow Universe + capturas propias |
| [`dual-track-yolov8-vs-ssd.md`](2026-05-05-dual-track-yolov8-vs-ssd.md) | Comparativa SSD-INT8 vs YOLOv8n-FP16, EfficientNMS_TRT roto en Maxwell (NVIDIA/TensorRT#1538), Roboflow multi-format | Reframing dual-track; NMS en CPU NumPy puro |
| [`preprocessing-roboflow.md`](2026-05-05-preprocessing-roboflow.md) | Resize Fit-black vs Stretch (zxq309 -4-5 pp), Crasto 2024 mosaic+mixup +11,3 pp, Yun&Wong QAT, repr dataset 400 muestras | Spec Generate Version 1-A / 1-B; NO Cutout, NO class weights |

---

## 5. Spec Roboflow Generate Version 1

Estrategia: **dos versiones separadas con preprocessing común**, una por target de inferencia.

### 5.1 Preprocessing común (ambas versiones)

| Paso | Configuración | Razón |
|------|---------------|-------|
| Auto-Orient | **ON** | Roboflow Docs verbatim: *"recommends defaulting to leaving this on"*. Safe en producción (frames OpenCV sin EXIF). |
| Modify Classes | Eliminar `cardboard`, `metal`, `miscellaneous`, `organic` (mantener `paper`, `glass`, `plastic`) | Reducir a 3 clases del MVP. |
| Filter Null | **ON (explícito)** | Sin esto, ~1.119 imgs vacías post-Modify Classes (9,7%) actuarían como background negatives ruidosos. |
| Resize | **Fit (black edges) in** | Equivalente a `LetterBox` Ultralytics. Evita train/inference mismatch (zxq309 issue YOLOv5#7454: -4-5 pp con stretch + letterbox interno). |

### 5.2 Augmentations (ambas versiones, multiplicador 3x)

Bbox-aware solamente. **NO Cutout** (image-level, puede borrar el único objeto-paper en una imagen — clase minoritaria con 1.384 bbox). **NO Mosaic** (Roboflow Docs verbatim: *"do not recommend using this with Roboflow 3.0 or YOLOv8"* — doble-mosaic destructivo; Ultralytics ya lo aplica online).

| Augmentation | Rango | Justificación |
|--------------|-------|---------------|
| Flip Horizontal | ON | Banda simétrica izq↔der. |
| Flip Vertical | OFF | Gravedad: residuos sobre banda nunca aparecen invertidos. |
| Rotation | ±15° | Tolerancia montaje cámara diagonal. |
| Shear | ±2° H/V | Microvibración banda. |
| Brightness | ±25% | Variación luz ambiente día/tarde. |
| Exposure | ±15% | Cambios apertura/iluminación. |
| Saturation | ±20% | Tolerancia al sensor UVC barato. |
| Blur | hasta 1,5 px | Movimiento sutil banda. |
| Noise | hasta 5% pixeles | Sensor low-light. |

### 5.3 Split y dos versiones generadas

- **Split común:** 70% train / 20% valid / 10% test, estratificado por clase (ya configurado en Roboflow).
- **Version 1-A** (Track A SSD): Resize Fit-black **320×320**, export `tfrecord`.
- **Version 1-B** (Track B YOLOv8): Resize Fit-black **416×416**, export `yolov8`.

Playbook completo paso-a-paso con citas verbatim de Roboflow Docs y código de validación gotcha clases fantasma: [`scripts/03_generate_roboflow_v1.md`](scripts/03_generate_roboflow_v1.md).

---

## 6. Quick start

### 6.1 Setup local (host x86, Windows/Linux)

```powershell
# Python 3.10+ con uv (gestor por defecto del proyecto)
uv sync
$env:ROBOFLOW_API_KEY = "<tu_api_key>"
$env:KAGGLE_USERNAME = "<tu_user>"   # opcional, si re-descargas el dataset original
$env:KAGGLE_KEY = "<tu_key>"
```

### 6.2 Generate Version 1 en Roboflow (acción manual)

Seguir paso a paso `scripts/03_generate_roboflow_v1.md`. Tiempo estimado: 20-30 min UI + 10-15 min de procesamiento Roboflow para cada versión.

### 6.3 Track A — entrenar SSD MobileNet v2 → TFLite INT8

Plataforma: **Google Colab T4** (Python 3.10, GPU). Tiempo: ~2-3 h para 12.000 steps.

```bash
# En Colab/local con GPU:
python notebooks/train_track_a_ssd.py
# → genera ./detect_int8.tflite (~3 MB) listo para Jetson Nano
```

Decisiones críticas hard-coded:
- QAT `delay=2000` **OBLIGATORIO** (Yun&Wong CVPR 2021: MobileNet-V1 sin QAT cae 71% → 3% accuracy en QUINT8).
- Repr dataset 400 muestras del val split, shuffle seed=42.
- `inference_input_type=tf.uint8` (no float).

### 6.4 Track B — entrenar YOLOv8n → ONNX

Plataforma: **Kaggle T4** o Colab T4. Tiempo: ~1-2 h para 100 epochs.

```bash
python notebooks/train_track_b_yolov8.py
# → genera yolov8n_waste.onnx (opset 11)
```

Decisiones críticas hard-coded:
- `imgsz=416` (Nature 2024 Tabla 4: +25% FPS vs 640 en Nano).
- `mosaic=1.0, close_mosaic=10, mixup=0.15, fliplr=0.5` (Crasto 2024 +11,3 pp mAP50).
- `opset=11` (TRT 8.0/8.2 del JetPack no lee opset > 13).
- Versión Ultralytics `>=8.4.31` (PR #24028 fix INT8 calibration imgsz no-cuadrado).

### 6.5 Compilar TensorRT engine en el Nano (Track B)

```bash
# En el Jetson Nano:
sudo systemctl stop lightdm                              # liberar RAM de X11
sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
export PATH=$PATH:/usr/src/tensorrt/bin
trtexec --onnx=/home/jetson/models/yolov8n_waste.onnx \
        --saveEngine=/home/jetson/models/yolov8n_waste_fp16.engine \
        --fp16 --workspace=1024 --verbose
```

`--workspace=1024` (1 GiB) seguro: ultralytics issue #14751 reporta `Killed` por OOM con workspace mayor en Nano 4 GB. Tiempo: 15-45 min.

### 6.6 Bench en el Nano

```bash
python scripts/bench_jetson.py --backend tflite_int8 --model detect_int8.tflite --imgsz 320
python scripts/bench_jetson.py --backend trt_fp16    --model yolov8n_waste_fp16.engine --imgsz 416
```

Threshold de viabilidad MVP: **≥10 FPS sostenidos** con servos+I2C corriendo concurrentemente.

---

## 7. Aportes para el informe IEEE

Los 3 ejes de novedad detectados en la búsqueda bibliográfica de la ronda 2 (sin repo público que combine los tres):

1. **QAT 320×320 + Jetson Nano deployment** — gap confirmado (Track A2 verbatim).
2. **Comparativa SSD-INT8 CPU vs YOLOv8n-FP16 GPU sobre Nano B01 con dataset waste 3-clase** — ningún paper publicado lo hace head-to-head.
3. **Ablación letterbox-vs-stretch en waste detection con aspect ratios mixtos** — brecha documentada en `preprocessing-roboflow.md` sec 8. Opcional: generar Version 1-B-alt-stretch (ver `scripts/03_generate_roboflow_v1.md` paso 8) y comparar mAP@50 contra 1-B canónica.

Otras 2 ablaciones IEEE-grade identificadas:

4. **imgsz 416 vs 640 en Track B** — Nature 2024 lo midió pero no en dataset custom waste.
5. **Class weights ON vs OFF (Track A)** — Crasto 2024 lo midió en YOLOv5 single-stage pero no en SSD MobileNet v2 FPNLite con desbalance moderado 5,15× foreground-foreground.

---

## 8. Gotchas conocidos (de la investigación, no perderlos)

- **EfficientNMS_TRT plugin roto en Maxwell** (TRT 8.0.1, NVIDIA/TensorRT#1538). Solución: NMS en CPU NumPy puro, overhead 1-3 ms.
- **`tflite.Interpreter` NO es thread-safe** para llamadas concurrentes con un único objeto. MVP: 1 instancia, 1 hilo de inferencia.
- **`num_threads > 1` con XNNPACK** puede degradar en TFLite Python 3.6 antiguo (TF #52076, #53146). Validar empíricamente.
- **Padding=114 (Ultralytics) vs Fit-black=0 (Roboflow)** — mismatch teórico no medido en literatura. Mitigación documentada en `preprocessing-roboflow.md` sec 1.
- **Roboflow Versions son inmutables y consumen créditos** — no regenerar Version 1-A/1-B sin necesidad.
- **Roboflow `roboflow-python` issue #88: clases fantasma** post-Modify Classes — código de validación incluido en el playbook (`scripts/03_generate_roboflow_v1.md` paso 6).

---

## 9. Referencias citables principales

- Crasto, K. (2024). *Class Imbalance in Object Detection: An Experimental Diagnosis and Study of Mitigation Strategies*. arXiv:2403.07113.
- Yun, S. & Wong, A. (2021). *Do all MobileNets quantize poorly? Gaining insights into the effect of quantization on depthwise separable convolutional networks through the eyes of multi-scale distributional dynamics*. CVPR 2021. arXiv:2104.11849.
- Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR 2018.
- Bashkirova, D. et al. (2022). *ZeroWaste Dataset: Towards Deformable Object Segmentation in Cluttered Scenes*. CVPR 2022.
- Bochkovskiy, A. et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*. arXiv:2004.10934.
- Zhong, Z. et al. (2020). *Random Erasing Data Augmentation*. AAAI 2020.
- Chakraborty et al. (2025). *Half-core utilization rule on Jetson edge devices*. arXiv:2508.08430.
- Nature Scientific Reports (oct 2024). *Real-time waste detection on Jetson Nano* (DOI: 10.1038/s41598-024-74798-3).

Lista completa con 95+ entradas en las tablas acumulativas de las 4 investigaciones.
