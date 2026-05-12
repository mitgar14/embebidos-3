# Investigación: Datasets de detección y pipeline de entrenamiento para clasificador de residuos MVP

> **Proyecto:** `embebidos-3` — clasificador de residuos en Jetson Nano B01 4 GB con JetPack 4.6.x.
> **Tarea:** detección de objetos (no clasificación pura) sobre 3 clases — paper, glass, plastic.
> **Modelo target:** SSD MobileNet v2 FPNLite 320×320 → TFLite INT8 + XNNPACK CPU.
> **Plan B:** EfficientDet-Lite0 320×320; o TensorRT FP16 si TFLite INT8 no alcanza ≥10 FPS.
> **Entrega:** 2026-05-26.

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-05 | alto | Datasets con bbox, domain adaptation, pipeline anotación |

---

## Síntesis ejecutiva

Tres hallazgos rectores definen el plan de entrenamiento:

1. **Existe un dataset que cubre las 3 clases con bbox y licencia abierta:** `arshnoor7389/garbage-classification-dataset` en Kaggle (13.714 imágenes, formato YOLO, MIT, junio 2025). Es el punto de partida principal. Roboflow Universe `material-identification/garbage-classification-3` (~10.000 imágenes, CC BY 4.0, 7 clases incluyendo paper/glass/plastic) es el complemento.
2. **Ningún dataset público combina simultáneamente paper/glass/plastic con condiciones de banda transportadora**. Los datasets industriales (ZeroWaste-f, SortWaste, SpectralWaste) tienen taxonomías de plástico y cartón, no glass ni paper genérico. El gap se cierra con recolección propia + fine-tuning.
3. **Auto-labeling con Grounded-SAM 2 / autodistill reduce 60-80% el tiempo de anotación** vs. manual puro. Para 300-600 imágenes propias del setup real, esperamos 1-2 h totales (incluyendo revisión humana) en lugar de 5-10 h.

Una decisión arquitectónica adicional emerge de la evidencia: **TensorRT FP16 con YOLOv8n a 416×416 supera en latencia (30 FPS) al SSD MobileNet v2 320×320 en TFLite INT8 sobre CPU (~12-16 FPS)** porque la GPU Maxwell del Nano B01 no tiene tensor cores INT8 nativos. La decisión actual del MVP (TFLite INT8 primario) es válida por simplicidad y estabilidad, pero el plan B documentado debe ejecutarse si los benchmarks empíricos no llegan a ≥10 FPS sostenidos.

---

## (1) Datasets de detección con bounding boxes

### Tabla acumulada — ranking por fitness al MVP

| # | Dataset | Imágenes | Clases relevantes | Tipo anotación | Formato | Licencia | Última actualización | Fitness MVP | URL |
|---|---------|---------:|-------------------|----------------|---------|----------|--------------------:|-------------|-----|
| 1 | **arshnoor7389/garbage-classification-dataset** (Kaggle) | 13.714 | paper, glass, plastic, +4 más (7 total) | bbox YOLO | YOLO + data.yaml | **MIT** | 2025-06-12 | **MUY ALTO** — clases exactas, bbox listas, licencia limpia, volumen suficiente | https://www.kaggle.com/datasets/arshnoor7389/garbage-classification-dataset |
| 2 | **material-identification/garbage-classification-3** (Roboflow Universe) | ~10.000 | PAPER, PLASTIC, GLASS, METAL, CARDBOARD, CLOTH, BIODEGRADABLE | bbox | YOLO/COCO/Pascal VOC/TFRecord | CC BY 4.0 | 2026-01 | **MUY ALTO** — multiformato export, las 3 clases exactas | https://universe.roboflow.com/material-identification/garbage-classification-3 |
| 3 | **trash-detection-1fjjc** (Roboflow, OBB) | 2.800 | paper, plastic, glass + 61 sub-clases | OBB (convertibles a axis-aligned) | YOLO | CC BY 4.0 | 2026-01 | ALTO — requiere merging de sub-clases | https://universe.roboflow.com/trash-dataset-for-oriented-bounded-box/trash-detection-1fjjc |
| 4 | **TACO** (Proença & Simões) | 1.500 (creciendo) | 60 sub-categorías → mapeable a paper, glass, plastic | segmentación poligonal (bbox derivable) | COCO JSON | CC BY 4.0 | activo | ALTO — "in the wild", taxonomía rica, requiere remapping | https://github.com/pedropro/TACO |
| 5 | **AquaTrash** (Harsh9524) | 369 | glass, paper, metal, plastic | bbox | CSV → COCO/YOLO | CC BY 4.0 | 2022 | MEDIO — exactamente 4 clases pero muy pequeño | https://github.com/Harsh9524/AquaTrash |
| 6 | **WASTE CLASSIFICATION** (Roboflow) | ~305 | paper, glass, plastic, cardboard, metal, trash | bbox | YOLO/COCO | CC BY 4.0 | 2023-10 | MEDIO — clases correctas pero pequeño | https://universe.roboflow.com/garbage-classification-qagjx/waste-classification-3nh5y |
| 7 | **ZeroWaste-f** (Bashkirova et al., CVPR 2022) | 4.503 + 6.212 sin etiquetar | soft plastic, rigid plastic, cardboard, metal | bbox + segmentación | COCO | CC BY-NC 4.0 | 2022 | MEDIO — entorno banda real, sin glass ni paper genérico | https://github.com/dbash/zerowaste |
| 8 | **SortWaste** (Inácio et al., WACV 2026) | 5.261 | HDPE, PET, PET Oil, Mixed Soft/Rigid Plastic, ECAL, Cardboard, Metal | bbox densas | COCO | académica | 2026-01 | MEDIO — banda industrial real, sin glass ni paper genérico | https://github.com/sarainacio/SortWaste |
| 9 | **SpectralWaste** (Casao et al.) | 852 + 6.803 sin etiquetar | film, basket, cardboard, video tape, filament, bag | segmentación | COCO | académica | 2024 | BAJO-MEDIO — taxonomía industrial específica | https://arxiv.org/abs/2403.18033 |
| 10 | **Drinking Waste Classification** (Kaggle) | 9.640 | aluminium cans, glass bottles, PET, HDPE | bbox | no estándar | CC0 | 2020 | BAJO-MEDIO — solo botellas y latas | https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification |
| 11 | **TrashCan 1.0** | 7.212 | plástico, metal, otro (submarino) | segmentación | COCO | académica + permiso comercial | 2020 | BAJO — entorno submarino, dominio muy distinto | https://arxiv.org/abs/2007.08097 |
| 12 | **infocomm_final-8i8is** (Roboflow) | 0 | declaradas paper/plastic/glass/metal/cardboard/trash | n/a | n/a | n/a | 2025-09 | **NO USABLE** — workspace vacío, 0 vistas, 0 descargas | https://universe.roboflow.com/finalinfocomm/infocomm_final-8i8is |

**Notas clave:**

- **arshnoor7389:** confirmado vía API Kaggle (2025-06-12, MIT, 13.714 imágenes en YOLO, total 4,3 GB). Estructura: `Dataset/images/` (13.714 archivos), `Dataset/labels/` (13.714 archivos), `Dataset/data.yaml` (269 bytes con nombres de clases). 7 clases anunciadas pero los nombres exactos están en una imagen embebida en la página de Kaggle (no extraíble por scraping); hay que descargar el dataset y leer `data.yaml` para confirmar.
- **infocomm_final-8i8is:** verificado, no es utilizable. El workspace existe pero sin imágenes accesibles (0 vistas, 0 descargas). Lo descartamos.
- **DWaste paper (arXiv 2510.18513):** menciona un dataset interno `dwaste-data-v4-annotated` con 11.163 imágenes y 19.700 bbox sobre las 7 clases exactas (biological, cardboard, glass, metal, paper, plastic, trash). Sin embargo, ese dataset anotado **no está disponible públicamente**; solo la versión sin bbox (`sumn2u/garbage-classification-v2`) es pública. Mantener seguimiento por si se publica.

### Estrategia de datos recomendada

```
Volumen objetivo entrenamiento: 4.000-6.000 imágenes (después de filtro y limpieza)
Volumen objetivo validación target: 100-200 imágenes propias del setup real

Fuente A (base):     arshnoor7389/garbage-classification-dataset
                     → filtrar a paper/glass/plastic, ~5.000-7.000 imágenes esperadas
Fuente B (refuerzo): material-identification/garbage-classification-3
                     → exportar formato COCO, filtrar a 3 clases, ~3.000-5.000 imágenes
Fuente C (in-the-wild): TACO (super-categorías paper, glass, plastic) → ~600-1.000 imágenes
Fuente D (target):   capturas propias del setup real
                     → 100-200 imágenes anotadas con auto-labeling + revisión humana
```

Combinación: A + B con deduplicación por perceptual hash (`imagehash.phash`) + C (opcional para diversidad in-the-wild) + D (target domain).

### Hallazgos contraintuitivos

**1a — Modelos zero-shot subrendimiento severo:** Grounding DINO y OWLv2 con prompts simples de clase obtienen solo 5-7,3 mAP en ZeroWaste vs 51,6 mAP de un modelo supervised fine-tuned (Abid et al. 2025, arXiv 2508.18799). Implicación: auto-labeling en condiciones de clutter alto requiere prompts cuidadosamente optimizados o fall-back a anotación manual. Para nuestras 3 clases en escenario MVP (objetos aislados sobre banda corta), el gap será menor — los prompts "paper", "glass bottle", "plastic" mapean a clases COCO estándar y tienen buena calidad zero-shot.

**1b — Datos sintéticos pueden empeorar:** En VisDA 2022, agregar 20.990 imágenes sintéticas (SynthWaste, Unity engine) bajó 2,9 puntos de mIoU vs solo entrenar con datos reales (45,5 → 42,6). El gap sintético-real es a veces peor que no tener datos extra. Para el MVP esto desaconseja generar imágenes sintéticas con Blender/Unity sin validación rigurosa en dominio target.

---

## (2) Domain adaptation para condiciones industriales (2024-2026)

### Cuantificación del gap empírico

VisDA 2022 (Bashkirova et al., arXiv 2303.14828) midió el gap entre dos plantas recicladoras reales del mismo tipo: **17,29 puntos de mIoU** con DeepLabv2; **10,5 puntos** con SegFormer (transformer). Es una referencia para estimar la degradación esperada al desplegar un modelo entrenado en datos web sobre nuestro setup.

### Curva de aprendizaje few-shot

| Volumen target | Recuperación del gap | Fuente |
|----------------|---------------------|--------|
| 0 (zero-shot) | 5-13,5 mAP en ZeroWaste (vs 51 fully supervised) | Abid et al. 2025 |
| k=5 por clase | **65%** del gap recuperado | Tabata & Nyirenda 2026 |
| k=15 por clase | **85%** del gap recuperado | Tabata & Nyirenda 2026 |
| 250+ imágenes | 52-59 mIoU (cerca de fully supervised) | VisDA 2022 ganadores |

**Implicación práctica:** con 100-200 imágenes target propias bien anotadas, recuperamos >85% del gap. Más de 500 da retornos decrecientes rápidamente.

### Augmentations validadas en literatura

Las técnicas dominantes en VisDA 2022 (top 3 ganadores) y manufacturing domain randomization (arXiv 2506.07539, 2025):

| Augmentation | Validada en | Efecto reportado |
|--------------|-------------|-------------------|
| **PhotoMetricDistortion** (brillo/contraste/saturación/tono) | SI-Analytics (1er VisDA 2022) | +2-4 mIoU consistente |
| **GaussNoise + RandomGridShuffle** | SI-Analytics | importante para robustez en líneas reales |
| **ColorJitter + GaussianBlur** | PICO++ (3er) | aplicado solo al branch student en student-teacher |
| **Perspective warp + Lighting jitter** | Leibniz Univ. Hannover 2025 | combinación más efectiva en líneas de producción |
| **Pseudo-labeling multi-augmentación** | SI-Analytics, PICO++ | crítico para self-training |
| **HRDA (multi-resolución, low+high crops)** | Pros (2do) | 55,46 mIoU |

**No incluir flip vertical:** los residuos sobre banda tienen orientación implícita (gravedad). Flip horizontal sí, vertical no.

### Hallazgo contraintuitivo (2)

**Cuantización INT8 puede actuar como regularizador.** Zaritskyi (2026) reportó "mejora inesperada de rendimiento tras cuantización" para MobileNetV2-SSD en detección de UAVs, atribuida a regularización implícita del pipeline TFLite. No es universal, pero contradice la asunción de "INT8 siempre degrada".

---

## (3) Pipeline de anotación rápida

### Tabla comparativa Roboflow vs CVAT vs Label Studio

| Criterio | **Roboflow** | **CVAT** | **Label Studio Community** |
|----------|--------------|----------|----------------------------|
| Precio free tier | Public Plan free forever (datos públicos en Universe). USD 99/mes para privados | Self-hosted gratis (Docker). Online tier gratuito limitado | Gratis ilimitado (MIT) |
| Auto-label IA | Auto Label con Grounding DINO (text prompt). 1.000 imágenes/job. SAM 2 smart polygon | YOLOv11 + SAM 2 vía Nuclio (self-hosted). Tracking video solo Enterprise | SAM/SAM2 + Grounding DINO como ML backend (requiere GPU local) |
| Export formats | YOLO, COCO, Pascal VOC, TFRecord, CreateML, CSV | YOLO, COCO, Pascal VOC, TFRecord, CVAT XML, Datumaro | YOLO, COCO, Pascal VOC, JSON nativo |
| Fricción setup | **Mínima** (browser) | Media-alta (Docker Compose) | Media (pip + ML backend) |
| Tiempo 300-600 imágenes | 50-100 min con auto-label | 2-4 h (incluyendo deploy backend) | 1-3 h (con Grounding DINO backend) |

**Recomendación para el MVP:** Roboflow Public Plan. Los datos del proyecto académico no son sensibles, la fricción es mínima, y Auto Label con Grounding DINO entrega la calidad necesaria para nuestras 3 clases (todas mapean a clases COCO estándar).

### Pipeline auto-labeling con autodistill + Grounded-SAM 2

`autodistill/autodistill-grounded-sam-2` (136 stars, Apache 2.0, Florence-2 + SAM 2). Pipeline trivial:

```python
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

base_model = GroundedSAM2(
    ontology=CaptionOntology({
        "paper": "paper",
        "glass bottle": "glass",
        "plastic bottle": "plastic",
        "plastic container": "plastic",
        "cardboard": "paper",   # opcional: fusionar
    })
)
base_model.label("./capturas_setup_real", extension=".jpg")
# Output: dataset_anotado/{train,valid,test}/images + labels (formato YOLO)
```

**Requisitos hardware:** GPU con 8-12 GB VRAM (Colab T4 16 GB sirve). En workstation local sin GPU dedicada, usar variantes ligeras: SAM ViT-B (en lugar de ViT-H) + Grounding DINO SwinT (en lugar de SwinB) → 6-7 GB VRAM total.

**Tiempo estimado realista:**
- Auto-labeling 300 imágenes en Colab T4: 20-40 min
- Revisión humana en CVAT/Roboflow: 30-60 min (aceptar/rechazar/corregir)
- Export COCO/YOLO: 5 min
- **Total: 55-105 min** vs 5-10 h manual puro

### Implementación de referencia identificada

`ecoCrafters/waste-detection` (GitHub) **es exactamente nuestra arquitectura objetivo:**
- Modelo: SSD MobileNet V2 FPNLite 320×320 (TF2 Object Detection API)
- Datasets: TACO + Drinking Waste Kaggle
- Notebooks listos:
  - `Collect_Data.ipynb` — preprocesamiento TACO/Kaggle a TFRecord
  - `Fine_tuning_ssd_mobilenet_fpnlite_320.ipynb` — fine-tuning Colab
  - `Eval_ssd_mobilenet_fpnlite_320.ipynb` — evaluación mAP
- 7 clases custom (drink can, glass bottle, plastic bottle types)

Es el repo más alineado con nuestro plan. No tiene export a TFLite INT8, pero ese paso lo cubre `tensorflow/models/research/object_detection/export_tflite_ssd_graph.py` + `tflite_convert --inference_type=QUANTIZED_UINT8`.

---

## (4) Stack de entrenamiento recomendado

### Decisión: TF Object Detection API vs MediaPipe Model Maker

**Primario: TF Object Detection API (`tensorflow/models`).**
- Provee checkpoints COCO cuantizados (QAT-ready) para SSD MobileNet v2 FPNLite 320×320.
- Pipeline export TFLite INT8 documentado y probado.
- Compatible con TF 2.5.0 + nv21.8 (wheel oficial NVIDIA para JetPack 4.6.1).
- Config en `research/object_detection/samples/configs/ssd_mobilenet_v2_quantized_300x300_coco.config`. Cambios mínimos: `num_classes: 3`, `image_resizer.fixed_shape_resizer: 320`, `quantization_aware_training: true`, paths de TFRecord y label map.

**Alternativa: MediaPipe Model Maker.**
- API de más alto nivel: `Dataset.from_coco_folder()` + `ObjectDetector.create()` + `quantization_aware_training()` + `export_model()`.
- Soporta `MOBILENET_V2_I320` con QAT nativo y exporta TFLite INT8 con metadatos.
- Requiere Python 3.8+ (no es problema porque entrenamos en host x86, no en Nano).

**Donde se entrena vs donde se infiere:**
- Entrenamiento: Colab T4 / Kaggle GPU / workstation x86 con Python 3.10+.
- Inferencia: Jetson Nano JetPack 4.6.1 con `tf.lite.Interpreter` desde TF 2.5.0+nv21.8 (wheel oficial NVIDIA) o `tflite_runtime` de Qengineering.

### Conflictos críticos identificados

| # | Conflicto | Mitigación |
|---|-----------|-----------|
| 1 | `tflite-model-maker` deprecado desde 2022, depende de TF 2.5-2.8 | Usar MediaPipe Model Maker (vivo, mismo backbone, QAT, Apache 2.0) |
| 2 | tflite_runtime para Python 3.6 aarch64 sin wheel oficial PyPI | Wheel comunitario Qengineering TFLite 2.7.0; alternativa: `tf.lite.Interpreter` desde TF 2.5+nv21.8 (wheel oficial NVIDIA, sí incluye `tf.lite`) |
| 3 | Custom op `TFLite_Detection_PostProcess` requiere build TFLite con custom ops | Wheel Qengineering ya lo incluye; con TF 2.5+nv21.8 también funciona |
| 4 | XNNPACK con `num_threads > 1` puede degradar en TFLite Python antiguo (issues TF #52076 #53146) | Validar empíricamente; fallback a backend default con `num_threads=4` |
| 5 | TensorRT INT8 en Maxwell GPU no tiene tensor cores → fallback FP32 | Si va plan B TensorRT, usar **FP16** no INT8 (alineado con `2026-05-05-arquitectura-software-jetson-nano.md:117`) |
| 6 | Ultralytics AGPL-3.0 si se distribuye como producto | Sin impacto para MVP académico, pero documentar |

### Latencias proyectadas en Jetson Nano B01

| Modelo | Resolución | Backend | Latencia/frame | FPS estimados | Fuente |
|--------|-----------|---------|----------------|---------------|--------|
| SSD MobileNet v2 300×300 INT8 | 300 | TFLite CPU 4 hilos | 60-80 ms | 12-16 | Tobiasz et al. 2023; NobuoTsukamoto |
| SSD MobileNet v2 320×320 INT8 | 320 | TFLite CPU 4 hilos | ~70-90 ms | 11-14 | Extrapolado |
| EfficientDet-Lite0 320×320 INT8 | 320 | TFLite CPU 4 hilos | ~55-70 ms | 14-18 | Zagitov et al. 2024 |
| YOLOv8n 416×416 FP16 | 416 | TensorRT | ~33 ms | ~30 | Scientific Reports 2024 |
| MobileNetV2 backbone 224×224 FP16 | 224 | TensorRT | ~12 ms | ~83 | Tobiasz et al. 2023 |

**Threshold de viabilidad:** ≥10 FPS sostenidos. SSD MobileNet v2 320×320 TFLite INT8 está en el borde inferior. EfficientDet-Lite0 da ~3-4 FPS extra a costa de 2-3 mAP@50. Si el bench empírico cae bajo 10 FPS, plan B inmediato a TensorRT FP16 (alineado con `2026-05-05-arquitectura-software-jetson-nano.md:121-127`).

---

## Conceptos clave

- **Bounding box (bbox):** rectángulo (x_min, y_min, x_max, y_max) o (cx, cy, w, h normalizados YOLO) que delimita un objeto en imagen.
- **mAP (mean Average Precision):** métrica estándar de detección. mAP@50 = IoU ≥ 0,5 entre predicción y ground truth; mAP@50:95 = promedio sobre IoU 0,5-0,95 paso 0,05.
- **QAT (Quantization-Aware Training):** entrenamiento simulando cuantización para reducir caída de accuracy post-quantización. Soportado por TF Object Detection API y MediaPipe Model Maker.
- **TFLite custom op `TFLite_Detection_PostProcess`:** operación específica que combina decodificación de boxes + NMS dentro del modelo. Requiere build TFLite con custom ops habilitados.
- **Auto-labeling / autodistill:** pipeline donde un modelo grande (Florence-2, Grounding DINO, SAM) genera anotaciones zero-shot que un anotador humano revisa y corrige, en lugar de anotar desde cero.
- **Domain adaptation:** conjunto de técnicas para reducir el gap entre el dominio de entrenamiento (datasets web) y el dominio de despliegue (banda transportadora real). Few-shot fine-tuning, augmentations específicas, pseudo-labeling, y self-training son las dominantes.
- **FPNLite:** Feature Pyramid Network ligera. SSD MobileNet v2 **FPNLite** mejora detección de objetos pequeños vs SSD plain a costa de ~10 ms de latencia.

---

## Recomendaciones operacionales

1. **Empezar el día 1 con `arshnoor7389`.** Descargar, leer `data.yaml`, filtrar las 3 clases objetivo. Es el camino más corto a un dataset entrenable.
2. **Combinar con `material-identification/garbage-classification-3`** (export COCO/YOLO desde Roboflow) tras deduplicación por phash.
3. **Recolectar imágenes propias del setup real desde el día 1**, no en fase 4. 100-200 imágenes anotadas con auto-labeling cierran 85% del gap.
4. **Usar `ecoCrafters/waste-detection` como repo plantilla** para el pipeline de fine-tuning SSD MobileNet v2 FPNLite 320×320.
5. **Validar latencia empíricamente sobre el modelo cuantizado**, no extrapolar; si <10 FPS, ejecutar plan B TensorRT FP16.

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | TACO: Trash Annotations in Context for Litter Detection (Proença & Simões) | https://arxiv.org/abs/2003.06975 | Paper | 1 |
| 2 | ZeroWaste Dataset (Bashkirova et al., CVPR 2022) | https://arxiv.org/abs/2106.02740 | Paper | 1 |
| 3 | VisDA 2022 Challenge: Domain Adaptation for Industrial Waste Sorting | https://arxiv.org/abs/2303.14828 | Paper | 1 |
| 4 | SpectralWaste Dataset (Casao et al.) | https://arxiv.org/abs/2403.18033 | Paper | 1 |
| 5 | Robust and Label-Efficient Deep Waste Detection (Abid et al.) | https://arxiv.org/abs/2508.18799 | Paper | 1 |
| 6 | The Garbage Dataset (GD) (Kunwar) | https://arxiv.org/abs/2602.10500 | Paper | 1 |
| 7 | DWaste: Greener AI for Waste Sorting | https://arxiv.org/abs/2510.18513 | Paper | 1 |
| 8 | TrashDet: Iterative NAS for Efficient Waste Detection | https://arxiv.org/abs/2512.20746 | Paper | 1 |
| 9 | SortWaste: Densely Annotated Dataset for Industrial Waste Sorting | https://arxiv.org/abs/2601.02299 | Paper | 1 |
| 10 | Edge Devices Inference Performance Comparison (Tobiasz et al.) | https://arxiv.org/abs/2306.12093 | Paper | 1 |
| 11 | Benchmarking DL Models for Object Detection on Edge Devices | https://arxiv.org/abs/2409.16808 | Paper | 1 |
| 12 | Comparative analysis of NN models on low-power devices (Zagitov et al.) | https://doi.org/10.18287/2412-6179-CO-1315 | Paper | 1 |
| 13 | TACO repo | https://github.com/pedropro/TACO | Repo | 1 |
| 14 | ZeroWaste repo | https://github.com/dbash/zerowaste | Repo | 1 |
| 15 | SortWaste repo | https://github.com/sarainacio/SortWaste | Repo | 1 |
| 16 | Garbage Dataset Experiments (sumn2u) | https://github.com/sumn2u/garbage-dataset-experiments | Repo | 1 |
| 17 | autodistill-grounded-sam-2 | https://github.com/autodistill/autodistill-grounded-sam-2 | Repo | 1 |
| 18 | Grounded-Segment-Anything | https://github.com/IDEA-Research/Grounded-Segment-Anything | Repo | 1 |
| 19 | mhyeonsoo/SAM_gDINO_AutoLabeling | https://github.com/mhyeonsoo/SAM_gDINO_AutoLabeling | Repo | 1 |
| 20 | ecoCrafters/waste-detection | https://github.com/ecoCrafters/waste-detection | Repo | 1 |
| 21 | SebastianCharmot/recycle_net | https://github.com/SebastianCharmot/recycle_net | Repo | 1 |
| 22 | Recyclero/PlaNet | https://github.com/Recyclero/PlaNet | Repo | 1 |
| 23 | dusty-nv/jetson-inference | https://github.com/dusty-nv/jetson-inference | Repo | 1 |
| 24 | EdjeElectronics/TFLite Object Detection on RPi | https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi | Repo | 1 |
| 25 | Qengineering TFLite SSD Jetson Nano | https://github.com/Qengineering/TensorFlow_Lite_SSD_Jetson-Nano | Repo | 1 |
| 26 | NobuoTsukamoto benchmarks | https://github.com/NobuoTsukamoto/benchmarks | Repo | 1 |
| 27 | tensorflow/models (TF Object Detection API) | https://github.com/tensorflow/models | Repo | 1 |
| 28 | google-ai-edge/mediapipe (Model Maker) | https://github.com/google-ai-edge/mediapipe | Repo | 1 |
| 29 | google-ai-edge/LiteRT | https://github.com/google-ai-edge/LiteRT | Repo | 1 |
| 30 | roboflow/supervision | https://github.com/roboflow/supervision | Repo | 1 |
| 31 | Ultralytics YOLO export to TFLite | https://docs.ultralytics.com/modes/export/ | Doc oficial | 1 |
| 32 | Ultralytics: Quick Start NVIDIA Jetson | https://docs.ultralytics.com/guides/nvidia-jetson/ | Doc oficial | 1 |
| 33 | Roboflow Universe: GARBAGE CLASSIFICATION 3 | https://universe.roboflow.com/material-identification/garbage-classification-3 | Dataset | 1 |
| 34 | Roboflow Universe: Trash Detection OBB | https://universe.roboflow.com/trash-dataset-for-oriented-bounded-box/trash-detection-1fjjc | Dataset | 1 |
| 35 | Roboflow Universe: WASTE CLASSIFICATION | https://universe.roboflow.com/garbage-classification-qagjx/waste-classification-3nh5y | Dataset | 1 |
| 36 | Kaggle: arshnoor7389 garbage-classification-dataset | https://www.kaggle.com/datasets/arshnoor7389/garbage-classification-dataset | Dataset | 1 |
| 37 | Kaggle: sumn2u garbage-classification-v2 | https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2 | Dataset | 1 |
| 38 | Kaggle: arkadiyhacks drinking-waste-classification | https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification | Dataset | 1 |
| 39 | AgaMiko/waste-datasets-review | https://github.com/AgaMiko/waste-datasets-review | Lista curada | 1 |
| 40 | Galliot: Deploying SSD MobileNet on Jetson Nano | https://galliot.us/blog/deploying-ssd-mobilenet-on-jetson-nano/ | Blog editorial | 1 |
| 41 | Edge Impulse: High-speed Counting Jetson Nano | https://docs.edgeimpulse.com/experts/readme/featured-machine-learning-projects/high-speed-counting-jetson-nano | Tutorial editorial | 1 |
| 42 | Roboflow: Auto Label Launch | https://blog.roboflow.com/launch-auto-label/ | Blog oficial | 1 |
| 43 | CVAT: automated annotation docs | https://docs.cvat.ai/docs/annotation/auto-annotation/automatic-annotation/ | Doc oficial | 1 |
| 44 | Label Studio: Segment Anything 2 docs | https://labelstud.io/guide/ml_tutorials/segment_anything_2_image | Doc oficial | 1 |
