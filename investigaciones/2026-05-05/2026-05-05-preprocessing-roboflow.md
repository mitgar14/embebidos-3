# Investigación: Preprocessing y Augmentations en Roboflow para pipeline dual-track waste-3class

> **Proyecto:** `embebidos-3` — clasificador de residuos en Jetson Nano B01.
> **Paso bloqueado:** Generate Version 1 en Roboflow `embebidos3/waste-3class-lwld8` (11.558 imgs, 13.873 bbox).
> **Pipeline target:** dual-track {SSD MobileNet v2 FPNLite 320×320 → TFLite INT8} + {YOLOv8n 416×416 → ONNX → TensorRT FP16}.
> **Decisión clave a tomar:** Resize (Stretch / Fit), Auto-Orient, Modify Classes + Filter Null, Augmentations 5x.
> **Restricción operativa:** Roboflow Versions son inmutables; cada versión cuesta créditos del Premium Trial.

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-05 | alto | Preprocessing Roboflow + Augmentations + Resize stretch vs letterbox + Class imbalance handling |
| 2 | 2026-05-05 | alto | Contextualización hardware-constrained: validar/refinar 6 decisiones filtrando evidencia por Jetson Nano-class (Maxwell sin tensor cores INT8, Pi 4, Coral, JetPack 4.6.x) |

Investigaciones cruzadas (no duplicar):
- `2026-05-05-arquitectura-software-jetson-nano.md` — domain camera diagonal, perspective warp implications.
- `2026-05-05-datasets-deteccion-residuos.md` — VisDA 2022 augmentations validadas, table en sec 96-107.
- `2026-05-05-dual-track-yolov8-vs-ssd.md` — table augmentations Roboflow Basic vs validadas (sec 233-249), pipeline.config hint (`image_resizer.fixed_shape_resizer: 320`).

---

## Síntesis ejecutiva

> **⚠ Update ronda 2 (2026-05-05)** — Tras filtrar la evidencia por hardware comparable a Jetson Nano B01 (Maxwell sin tensor cores INT8, JetPack 4.6.x, Pi 4, Coral):
>
> 1. **Reframing dual-track:** ya NO es "INT8 vs FP16" — es **"TFLite INT8 (CPU XNNPACK + NEON SIMD ARM) vs TensorRT FP16 (GPU Maxwell sin tensor cores)"**. La ganancia/penalización viene del *backend hardware*, no de la cuantización per se. Maxwell no tiene unidades INT8 VNNI (solo desde Turing/Ampere) → TensorRT INT8 no acelera en Maxwell pero **TFLite INT8 sí acelera en CPU vía NEON SIMD**.
> 2. **imgsz=416 Track B VALIDADO cuantitativamente.** Nature Scientific Reports oct 2024 (DOI: 10.1038/s41598-024-74798-3) Tabla 4 reporta YOLOv8n+TensorRT FP16 Jetson Nano: 416=**30 FPS** / 512=29 FPS / 640=24 FPS. Špeh Medium 2023 confirmó YOLOv7 Tiny: 416=17 FPS / 640=9 FPS (1.9× ganancia). 416 es la elección defensiva con margen de FPS para concurrencia con servos+I2C.
> 3. **QAT en Track A NO es opcional, es CRÍTICO.** Yun & Wong CVPR 2021: MobileNet-V1 ImageNet sin QAT cae de **71.04% FP32 a 3.00% QUINT8** por distributional mismatch entre depthwise/pointwise convolutions. SSD MobileNet v2 FPNLite usa exactamente estas convoluciones. Mi `pipeline_custom.config` ya tiene `quantization_aware_training: true` con `delay: 2000` — se mantiene como configuración obligatoria.
> 4. **Repr dataset INT8 calibration: subir de 200 a 300-500 muestras.** Google LiteRT docs dice 100-500 ok; Ultralytics #14121 (Glenn Jocher) recomienda 1000+; Boddu&Mukherjee 2025 RPi 5 con 100 muestras logró robustez. Para Track A SSD usar 300-500 distribuidas proporcionalmente entre las 3 clases del val split. Bug crítico ya fixed: PR #1695 jul 2023 (normalización ImageNet en calibration causaba mAP 0.678→0.318 en SKU-110K). Verificar `ultralytics > 8.4.31` para PR #24028 (mar 2026) que arregla calibration con imgsz no-cuadrado.
> 5. **Padding=114 (Ultralytics default) vs Roboflow Fit-black=0:** mismatch teórico no medido en literatura (gap declarado). Ultralytics PR #21652 ago 2025 finalmente hizo `padding_value` configurable. Mitigación: o aplicar padding=0 explícito en inferencia Jetson para alinear con Roboflow Fit-black, o aplicar `cv2.copyMakeBorder(value=(114,114,114))` y aceptar el mismatch (riesgo bajo según razonamiento teórico Krishnamoorthi 2018).
> 6. **Mosaic+Mixup VALIDADO cuantitativamente:** AliHamzaAzam/vision-dl-waste-detection (TACO, nov 2025) reporta **+17% mAP con mosaic**. techishthoughts: "With only 891 images, mosaic is essential to prevent memorization." Karimov et al. 2025 confirma que mosaic en training NO daña INT8 calibration (mixed calibration con degradaciones tampoco mejora pero no perjudica para nano-scale).
> 7. **Originalidad confirmada:** ningún repo público combina waste sorting + Jetson Nano B01 + Roboflow + dual-track. Caso más cercano: Li & Grammenos UCL arXiv 2210.00448 (2022, MobileNet V3 classification, no detection). Track A2 cita verbatim: *"QAT 320×320 + Jetson Nano deployment es una laguna en la literatura pública de repositorios"*. Mi proyecto puede llenarla → aporte para informe IEEE.

**Convergencia de 3 líneas de investigación independientes (academic + docs + repos):**

1. **NUNCA Stretch en Roboflow para entrenar YOLO/SSD.** Aunque es la opción que muchos repos comunitarios usan por inercia, introduce un *train/inference mismatch* sistemático porque Ultralytics aplica letterbox interno automático en `val` y `predict` (defaults `rect=True`). El usuario `@zxq309` documentó **4-5 pp de caída en P/R/mAP** en YOLOv5 issue #7454 al combinar Roboflow stretch + Ultralytics letterbox interno. Para SSD/TF OD API hay matiz adicional: `fixed_shape_resizer` requiere shapes estáticos para TFLite INT8 — pero esto se cumple igual con Fit-black 320×320 (la imagen sale cuadrada con padding negro).

2. **Decisión final: Roboflow "Fit (black edges) in" — versiones separadas por target.** Es el equivalente exacto al `LetterBox` de Ultralytics (padding 114 ≈ negro 0 difieren en gris medio, diferencia menor; cita Glenn Jocher #1279: "114 is ImageNet mean averaged over RGB"). Concilia A1 (académico, evita mismatch), A2 (docs oficiales), y producción (cámara fija UVC sobre banda nunca producirá imágenes pre-distorsionadas). Una versión a 320×320 (Track A) y otra a 416×416 (Track B). El costo administrativo (2 versiones Premium en lugar de 1) es trivial.

3. **NO class weights, NO oversampling. SÍ Mosaic + Mixup vía Ultralytics interno.** Crasto 2024 (arXiv:2403.07113), tabla 1: en YOLOv5 single-stage con foreground-foreground imbalance long-tailed, sampling-CAS = **-0,6 pp**, loss weights tuned = **-1,9 pp**, mientras mosaic = **+8,0 pp** y mosaic+mixup(p=0,3) = **+11,3 pp mAP50**. Mi desbalance 5,15× (plastic 7.128 / paper 1.384) es moderado-bajo según ese estudio. Mosaic+Mixup ya están como defaults en Ultralytics — NO los toco.

4. **NUNCA Mosaic en Roboflow para YOLOv8.** Cita verbatim de docs.roboflow.com: *"Many modern models apply Mosaic as an online augmentation during training; applying it twice can cause undesirable results. We do not recommend using this with Roboflow 3.0 or YOLOv8."* Doble-mosaic destruye anotaciones por superposición de cuadrantes.

5. **Cutout (Premium): no aplicar globalmente.** Random Erasing (Zhong et al. AAAI 2020) +1,4 pp mAP en Fast-RCNN/PASCAL VOC. Pero Roboflow Cutout es image-level (no preserva bboxes per-clase), y mi clase paper (1.384 bbox) es la minoritaria — Cutout agresivo puede borrar el único objeto-paper en una imagen. Object-Oriented Cutout (Yim et al. 2025) preserva bboxes pequeños pero NO existe en Roboflow; lo más cercano es Albumentations CoarseDropout vía Ultralytics. **Decisión: NO Cutout en Roboflow Version 1; aplicar en augmentations Basic = Flip H, Rot ±15°, Brightness ±25%, Exposure ±15%, Blur 1.5px, Noise 5%, MotionBlur leve. Multiplicador 3x.** Si la primera Version necesita más diversidad post-eval, generar Version 2 con Cutout solo si paper class no es la dominante en false negatives.

6. **Auto-Orient: ON.** Verbatim Roboflow Docs: *"Roboflow recommends defaulting to leaving this on."* Safe para mi caso porque OpenCV `VideoCapture` en Jetson Nano entrega frames sin EXIF (no hay nada que strip en producción).

7. **Modify Classes (delete cardboard, miscellaneous, organic, metal) + Filter Null EXPLÍCITO.** Modify Classes solo elimina bboxes, no imágenes. Verbatim doc: *"Images marked as null annotation, or 'unannotated' after applying the Modify Classes tool, are the only ones affected when using Filter Null."* Sin Filter Null, las ~1.119 imágenes que solo tenían cardboard/etc. quedarían como background negatives (lo que puede ayudar al modelo si son pocas, pero con 1.119 imgs vacías sobre 11.558 = 9,7% del dataset es ruido excesivo).

8. **Brecha de investigación documentada:** No existe paper publicado con ablación letterbox-vs-stretch sobre dataset de waste sorting con aspect ratios mixtos. Documentar empíricamente el delta de mi pipeline puede ser un aporte para el informe IEEE.

---

## (1) Resize: análisis comparativo

### 1.1 Tabla de opciones Roboflow (verbatim docs.roboflow.com)

| Opción | Comportamiento (cita verbatim) | Aspect ratio | Cuadrada | Bbox preservado | Compatibilidad TFLite INT8 estático | Compatibilidad Ultralytics |
|---|---|---|---|---|---|---|
| **Stretch to** | "Stretch your images to a preferred pixel-by-pixel dimension. Annotations are scaled proportionally. Images are square, distorted, but no source image data is lost." | NO preservado | sí | sí (escalados) | ✅ | ⚠️ mismatch con letterbox interno |
| **Fill (with center crop)** | "The generated image is a centered crop... The aspect ratio is maintained, but source image data is lost." | sí | sí | sí (puede recortar) | ✅ | ✅ |
| **Fit within** | "Image aspect ratios and original data are maintained, but they are not square." | sí | **NO** | sí | ❌ shapes variables | ❌ fuerza segundo letterbox |
| **Fit (reflect edges)** | "...newly created padding is a reflection of the source image. Notably, **Roboflow also reflects annotations by default**." | sí | sí | ⚠️ bboxes artificiales en padding | ✅ | ✅ pero ruido en bbox |
| **Fit (black edges)** ⭐ | "...newly created padding is black area. Images are square, black padded, and aspect ratios plus original data are maintained." | sí | sí | sí (intactos) | ✅ | ✅ ≈ LetterBox padding 114 |
| **Fit (white edges)** | "...newly created padding is white area." | sí | sí | sí | ✅ | ⚠️ alta intensidad confundible con paper |

### 1.2 Evidencia académica clave

**Crasto 2024 (arXiv:2403.07113) — paper ORO:** ablación foreground-foreground imbalance YOLOv5s sobre COCO-ZIPF.

| Estrategia | mAP50 | Δ baseline |
|---|---|---|
| Baseline sin augmentation | 35,8% | — |
| + Mosaic | 43,8% | **+8,0 pp** |
| + Mosaic + Mixup (p=0,3) | 47,1% | **+11,3 pp** |
| + Class-Aware Sampling (CAS) | 35,2% | -0,6 pp |
| + Loss weights tuned | 33,9% | -1,9 pp |

**Conclusión verbatim:** *"sampling and loss reweighing methods do not translate as effectively in improving YOLOv5's performance"* (foreground-foreground).

**Lin et al. 2017 (Focal Loss):** RetinaNet-101 + FL = 37,8% AP COCO. Diseñada para fg-bg imbalance ratio **1:100.000** anchors negativos. Para mi 5,15× class imbalance, ganancia marginal. Disponible en TF OD API como `weighted_sigmoid_focal_cross_entropy_loss` para Track A si experimento.

**Zhong et al. AAAI 2020 (Random Erasing):** Fast-RCNN PASCAL VOC 2007: **74,8% → 76,2% mAP (+1,4 pp)** con sl=0,02, sh=0,4, p=0,5. CIFAR-10 WRN-28-10: top-1 error 3,72% → 3,08% (-0,64 pp).

**Yim et al. 2025 (Object-Oriented Cutout):** *"OOC achieves a 0.6% improvement in mAP compared to the YOLOv5 baseline"* en VisDrone2019 (tiny objects). Cutout standard puede destruir información semántica de objetos pequeños cuando el rectángulo borrado coincide con un único bbox.

**Bochkovskiy et al. 2020 (YOLOv4):** ablation table COCO 416px. Sin Mosaic = 41,2% AP. Con Mosaic = 43,5% AP. **+2,3 pp.** *"Mosaic represents a new data augmentation method that mixes 4 training images."*

**ZeroWaste (Bashkirova et al. CVPR 2022):** dataset 1.800+ frames banda industrial real. Augmentations descritas: standard photometric + RandomResizedCrop. Sin ablación numérica de preprocessing individual publicada — **brecha de investigación confirmada.**

**Glenn Jocher / Ultralytics — issues GitHub:**
- **#7454 (zxq309)**: *"P, R, mAP decline about 4~5%"* aplicando letterbox manual en Roboflow + letterbox de YOLOv5 (doble procesamiento).
- **#1279**: *padding 114 = "ImageNet mean averaged over RGB"*, no negro (0) ni blanco (255), para minimizar sesgo de contraste.
- **#7053 (ProBroSam)** (caso casi idéntico al mío, mixed aspect ratio + Roboflow + YOLOv8): respuesta verbatim Glenn Jocher: *"YOLOv8 is designed to work with mixed aspect ratios... you don't have to choose between 'Stretch to' or 'Fit (white edges) in.'"* — pero esa respuesta es para el caso de NO forzar resize en Roboflow; si decides forzar resize, **debes** usar Fit-black para alinear con letterbox interno.

### 1.3 Evidencia de repos waste detection (B2)

| Repo | Modelo | imgsz | Resize Roboflow | Augmentations |
|---|---|---|---|---|
| boss4848/waste-detection (49⭐) | YOLOv8n | default 640 | (no documentado) | Ultralytics defaults |
| Gokzz-glitch/zerowastex | YOLOv8m | 640 | (no documentado) | hsv_s=0.9, hsv_v=0.6, degrees=30, scale=0.6, mosaic=1.0, mixup=0.3 (agresivo, India outdoor) |
| andresriverosb1331/Trash_Classifier (Chile, banda) | YOLO11 | 640 | (no documentado) | degrees=10, translate=0.1, scale=0.5, hsv_s=0.7, hsv_v=0.4, mosaic=1.0, mixup=0.1 |
| starwit/waste-detection (producción DVC) | YOLOv8m | 1280 | (no documentado) | freeze_backbone, fine-tune 2-fase |
| Qengineering/YoloV8-TensorRT-Jetson_Nano | YOLOv8n | (default) | n/a (eval) | n/a |
| BU EC523 / ZeroWaste | YOLOR | 640 (rechazado 416 por no múltiplo 64) | Roboflow + augment=True | 150 epochs |

**Patrón "lazy default" universal:** ningún repo público documenta la decisión Stretch vs Fit en código. Todos llaman `version.download("yolov8")` y confían en lo configurado en la web. **Mi `.md` será la única auditoría reproducible.**

**Qengineering benchmark mismo hardware (Jetson Nano):**
- YOLOv8n FP16 = **19 FPS**
- YOLOv8s FP16 = 9,25 FPS
- INT8 NO mejora FPS y degrada mAP en Maxwell sin tensor cores INT8.
→ Refuerza decisión Track B = FP16, no INT8 (ya investigado en `dual-track-yolov8-vs-ssd.md`).

### 1.4 Implicación específica para mi dataset (median 512×640 portrait)

Median image size 512×640 → **portrait/tall**. Aspect ratio mediano = 0,8 (ancho/alto).

**Track B (target 416×416):**
- **Fit-black 416×416**: alto = 416, ancho = 416×(512/640) = 332,8 ≈ 333 px. Padding negro = 416-333 = 83 px (41 px a cada lado). 80% de la imagen = contenido real.
- **Stretch 416×416**: comprime ancho 18,7%. Botellas (plastic) verticales se vuelven proporcionalmente más anchas. Cita Roboflow staff Leo en discuss.roboflow.com #6892: *"Detecting different cardboard box shapes—stretching would distort their proportions, making classification harder."* Aplica directamente a botellas de plástico (alargadas verticalmente) y frascos de vidrio (cuadrados o redondos).

**Track A (target 320×320):**
- **Fit-black 320×320**: alto = 320, ancho = 320×(512/640) = 256 px. Padding = 64 px. 80% contenido real.
- **`fixed_shape_resizer { 320, 320 }` recibe imagen ya 320×320 cuadrada con padding negro → no aplica nada adicional.** TFLite INT8 estático compatible (shape fija).

### 1.5 Consistencia entrenamiento ↔ inferencia

En producción Jetson Nano, mi pipeline será:
```python
# Track B inferencia (Ultralytics PyTorch o TRT runtime)
frame = cv2.VideoCapture(0).read()    # 720x1280 BGR
# Ultralytics aplica LetterBox(416) interno al modelo .pt
# O TRT engine espera input pre-procesado: aplicar letterbox manual con padding 114
```

Si entreno con **Roboflow Fit-black** y produzco con **Ultralytics letterbox interno (padding 114)**, la única diferencia es el color de padding (0 vs 114) — diferencia menor pero presente. **Mitigación opcional:** post-process pipeline en Jetson para usar `cv2.copyMakeBorder(..., value=(114,114,114))` en lugar de Ultralytics default. Para el MVP no es crítico; documentarlo como observación.

Si entreno con **Roboflow Stretch** y produzco con **Ultralytics letterbox**, hay deformación-vs-padding asimétrica → **caída 4-5 pp documentada en #7454**. **NO ACEPTABLE.**

---

## (2) Auto-Orient

**Cita verbatim Roboflow Docs (`docs.roboflow.com/datasets/dataset-versions/image-preprocessing`):**

> Auto-orient strips your images of their EXIF data so that you see images displayed the same way they are stored on disk. EXIF data determines the orientation of a given image. [...] Roboflow recommends defaulting to leaving this on and checking how your images in inference are being fed to your model.

**Para mi caso:**
- Dataset arshnoor7389: capturado con smartphones (mixto). EXIF rotation puede estar presente. Auto-Orient lo strip → orientación canónica.
- Producción Jetson Nano + OpenCV `VideoCapture(0, cv2.CAP_V4L2)` → frames sin EXIF (es video stream, no archivos).
- **Conclusión:** Auto-Orient ON sin riesgo. La inferencia OpenCV ya entrega "como están en disco", consistente con training post-strip.

---

## (3) Modify Classes + Filter Null

### 3.1 Comportamiento exacto

**Modify Classes (verbatim docs):**
> A preprocessing tool used to omit specific classes or remap (rename) classes when generating a new version of your dataset. These changes only apply to the version you generate. No changes will be made to your underlying dataset.

**Filter Null (verbatim docs):**
> The Filter Null transformation allows users to require a share of images in a dataset to be annotated. Images marked as null annotation, or 'unannotated' after applying the Modify Classes tool, are the only ones affected when using Filter Null.

**Workflow inferido (no automático):**
1. Modify Classes elimina bboxes de clases marcadas como delete (cardboard, miscellaneous, organic, metal). Las imágenes permanecen.
2. Imágenes con bboxes mixtas (ej. plastic + cardboard): conservan solo bboxes de plastic. **NO se descartan.**
3. Imágenes que SOLO tenían cardboard/miscellaneous/organic/metal: quedan sin bboxes = "null annotation".
4. **Filter Null debe activarse explícitamente** para eliminar esas imágenes-null. Sin él, quedan como background negatives.

### 3.2 Aplicación a mi dataset

Distribución por clase actual (post-upload):
- plastic: 7.128 bbox
- glass: 1.927 bbox
- metal: 1.801 bbox (delete)
- paper: 1.384 bbox
- organic: 760 bbox (delete)
- miscellaneous: 475 bbox (delete)
- cardboard: 400 bbox (delete)

Total bboxes: 13.875. Tras delete: 7.128 + 1.927 + 1.384 = **10.439 bboxes** (-3.436 bboxes, -24,8%).

**Estimación imágenes resultantes:** No conozco la matriz de co-ocurrencia exacta. Asumiendo distribución bbox/img ≈ 1,2 (13.873/11.558), las imágenes que tienen SOLO clases-delete son ~⌈3.436/1,2⌉ ≈ 2.863 imágenes en peor caso si todas son single-bbox de clases-delete. Probablemente menos por co-ocurrencia.

**Decisión:** ACTIVAR Filter Null para evitar contaminar dataset con 1.000-3.000 imágenes background.

---

## (4) Augmentations Roboflow Version 1

### 4.1 Restricciones Roboflow

**Verbatim docs.roboflow.com/datasets/dataset-versions/image-augmentation:**
> Enhanced Augmentations and Bounding Box Augmentations are premium features.
> We recommend starting a project with no augmentations. This allows you to evaluate the quality of your raw dataset.

**Tabla bbox-aware (verbatim):**
- **Preserve bboxes:** Flip, 90° Rotate, Crop, Rotation, Shear, Brightness, Exposure, Blur, Noise, Camera Gain, Motion Blur.
- **Image-level only (no bbox support):** Grayscale, Hue, Saturation, Cutout, Mosaic.

**Premium Trial (14 días, $60 créditos):** desbloquea Enhanced (incluye Mosaic; Cutout no confirmado en doc).

**Aviso explícito Roboflow re: Mosaic con YOLOv8:**
> Many modern models apply Mosaic as an online augmentation during training; applying it twice can cause undesirable results. We do not recommend using this with Roboflow 3.0 or YOLOv8.

### 4.2 Decisión por augmentation

| Augmentation | Decisión Version 1 | Justificación |
|---|---|---|
| Flip Horizontal | **ON** | Banda transportadora simétrica izq↔der. Validado VisDA 2022 (memoria mnemon `a6ff011a`). |
| Flip Vertical | **OFF** | Residuos tienen orientación gravitacional natural sobre banda. Validado VisDA 2022. |
| Rotation ±15° | **ON** | Cámara diagonal sobre banda → tolerancia leve a yaw del objeto. Trash_Classifier (Chile) usa degrees=10. ZeroWaste no especifica. Mantener ±15° por ser margen razonable sin destruir aspect ratio. |
| 90° Rotate | **OFF** | Conflicto con orientación gravitacional. |
| Shear ±2° | **ON (leve)** | Proxy de PerspectiveWarp real (no disponible en Roboflow). Validado en Leibniz Hannover 2025 (`datasets-deteccion-residuos.md` sec 105). |
| Brightness ±25% | **ON** | Iluminación industrial varía con hora del día. VisDA 2022 + Trash_Classifier hsv_v=0.4 ≈ ±25%. |
| Exposure ±15% | **ON** | Complemento brightness. Cubre overexposure típico de superficie metálica brillante (latas → glass mislabel). |
| Hue shift ±10° | **OFF** | Cita memoria mnemon: tinte falso de papel puede confundir con cartón. Hue es image-level (no preserve bbox). |
| Saturation ±20% | **ON moderado** | Vidrio transparente vs plástico transparente: vidrio tiene microreflexiones más coloreadas. Saturación moderada ayuda generalización sin invertir distinción. |
| Blur 1.5px | **ON** | Frames de cámara UVC en banda en movimiento → motion blur ligero. Validado VisDA 2022. |
| Noise 5% | **ON** | Ruido de cámara low-light Jetson. Validado VisDA 2022. |
| MotionBlur leve | **ON** | Banda en movimiento. |
| **Cutout** (Premium) | **OFF en Version 1** | Image-level (no bbox-aware en Roboflow). Riesgo de borrar único bbox de paper (1.384 bbox = clase minoritaria). Object-Oriented Cutout (Yim 2025) preserva bbox pequeños pero NO está en Roboflow. **Re-evaluar para Version 2 si modelo confunde clases por oclusión real en banda.** |
| **Mosaic** (Premium) | **OFF en Version 1** | Roboflow oficial: "do not recommend with YOLOv8" (doble-mosaic). Track B: Ultralytics aplica `mosaic=1.0` interno. Track A SSD: TF OD API NO tiene mosaic — si necesito, aplico offline post-export. |
| Crop (Bounding Box) | **OFF** | Random Crop puede dejar imagen sin objetos de la clase positiva, especialmente para paper minoritaria. |
| Grayscale | **OFF** | Plastic/glass/paper diferenciables en parte por color (paper blanco, plastic transparente/coloreado, glass marrón/verde/transparente). |
| Tile | **OFF** | Para small object detection — mis residuos ocupan 5-15% imagen, no son small. |

**Multiplicador:** **3x** (Roboflow Public Plan default; Premium Trial desbloquea hasta 10x).

Razón para 3x y no 5x/10x:
- Crasto 2024 confirma que augmentation excesiva no monotonic con benefit.
- 11.558 imgs × 3 = ~34.674 imágenes train-aug → cubre la masa requerida sin gastar 9× créditos en augmentations marginales.
- Versión 1 = baseline. Si gana benchmark dual-track, no hay que regenerar. Si pierde, generar Version 2 con 5x + Cutout.

### 4.3 Comparación con repos referencia

| Parámetro | zerowastex (India outdoor) | Trash_Classifier (Chile banda) | DWaste (paper edge) | **Mi Version 1** |
|---|---|---|---|---|
| Mosaic Roboflow | n/a | n/a | n/a | OFF |
| Mosaic Ultralytics | 1.0 | 1.0 | 1.0 | 1.0 (default) |
| Mixup | 0.3 | 0.1 | n/a | 0.0 (Track B notebook) → considerar 0.15 |
| degrees | 30 | 10 | n/a | ±15° Roboflow |
| hsv_s | 0.9 | 0.7 | n/a | Saturation ±20% Roboflow |
| hsv_v | 0.6 | 0.4 | n/a | Brightness ±25% Roboflow |
| fliplr | 0.5 | 0.5 | n/a | Flip H ON |
| translate | 0.2 | 0.1 | n/a | OFF (Roboflow Crop OFF) |
| scale | 0.6 | 0.5 | n/a | OFF |
| Class weights | NO | NO | **YES** (computed weights) | NO (Crasto: contraproducente single-stage) |

**Nota DWaste:** ellos usaron class weights para detection. Crasto 2024 contradice esa estrategia para single-stage. Mi decisión sigue Crasto.

---

## (5) Especificación EXACTA Roboflow Version 1 (copy-pasteable)

### 5.1 Versión común (compartida por ambos tracks via Modify Classes)

```
Roboflow Workspace: embebidos3
Project: waste-3class-lwld8
Generate Version 1:
  ├─ Source Images: 11.558 (todas las uploaded del batch arshnoor-base)
  ├─ Train/Test Split: 70 / 20 / 10  (estratificado por clase, default Roboflow)
  │
  ├─ PREPROCESSING:
  │   ├─ Auto-Orient: ENABLED
  │   ├─ Modify Classes:
  │   │     ├─ Delete: cardboard
  │   │     ├─ Delete: miscellaneous
  │   │     ├─ Delete: organic
  │   │     └─ Delete: metal
  │   │     (Keep: plastic, glass, paper)
  │   ├─ Filter Null: ENABLED
  │   ├─ Resize: → DECIDIR PER-TRACK abajo
  │   ├─ Tile: DISABLED
  │   ├─ Static Crop: DISABLED
  │   ├─ Grayscale: DISABLED
  │   └─ Auto-Adjust Contrast: DISABLED
  │
  ├─ AUGMENTATIONS (3x multiplier, Roboflow Basic — Premium Trial activo):
  │   ├─ Image-level:
  │   │     ├─ Flip Horizontal: ENABLED
  │   │     ├─ Flip Vertical: DISABLED
  │   │     ├─ 90° Rotate: DISABLED
  │   │     ├─ Rotation: ±15°
  │   │     ├─ Shear: ±2° H, ±2° V
  │   │     ├─ Hue: DISABLED
  │   │     ├─ Saturation: ±20%
  │   │     ├─ Brightness: ±25%
  │   │     ├─ Exposure: ±15%
  │   │     ├─ Blur: up to 1.5px
  │   │     ├─ Noise: up to 5% pixels
  │   │     └─ Mosaic: DISABLED  (Roboflow oficial NO recomienda con YOLOv8)
  │   ├─ Bounding-Box-level:
  │   │     ├─ Bbox-Crop: DISABLED  (riesgo de eliminar paper class)
  │   │     ├─ Bbox-Rotation: DISABLED  (redundante con Image-level Rotation)
  │   │     └─ Cutout: DISABLED en Version 1 (Premium-only + image-level + riesgo paper minoritaria)
```

### 5.2 Track A — versión separada para SSD MobileNet v2 FPNLite 320

```
Roboflow Generate Version 1-A (track-a-ssd-320):
  ├─ Same Preprocessing as 5.1 (Auto-Orient, Modify Classes, Filter Null)
  ├─ Resize: Fit (black edges) in 320×320
  └─ Augmentations: idénticas a 5.1
```

### 5.3 Track B — versión separada para YOLOv8n 416

```
Roboflow Generate Version 1-B (track-b-yolov8-416):
  ├─ Same Preprocessing as 5.1 (Auto-Orient, Modify Classes, Filter Null)
  ├─ Resize: Fit (black edges) in 416×416
  └─ Augmentations: idénticas a 5.1
```

### 5.4 Exports requeridos

- Version 1-A → exportar formato `tfrecord` para `train_track_a_ssd.py`.
- Version 1-B → exportar formato `yolov8` para `train_track_b_yolov8.py`.

### 5.5 Coste estimado Premium Trial

- 2 versiones × ~11.558 imgs source × 3x augmentation = ~69.348 imágenes-versión generadas. Bien dentro del límite Premium Trial (250.000 imágenes/workspace).

---

## (6) Diferencias con propuesta inicial

Mi propuesta original fue: "Fit within 640x640 white edges + Modify Classes + Aug 5x con Cutout".

Cambios fundamentados:

| Aspecto | Propuesta original | Decisión final | Razón |
|---|---|---|---|
| Resize tipo | Fit within (no cuadrada) | **Fit (black edges) in** | Fit within produce shapes variables → forces second resize en Ultralytics y rompe TFLite INT8 estático. Black edges = match con LetterBox padding 114. |
| Resize color | white edges | **black edges** | White (255) confundible con paper class blanco; black (0) cerca de padding 114 ImageNet mean. |
| Target size | 640×640 universal | **320×320 (A) + 416×416 (B), versiones separadas** | Roboflow Docs sin entry SSD MV2 FPNLite. Resize en Roboflow al target real evita resize en cascada (640 → 416 → letterbox + 640 → 320 stretch). Cada versión cuesta solo ~3x storage Premium Trial. |
| Aug multiplicador | 5x con Cutout | **3x sin Cutout** | Cutout image-level + paper minoritaria (1.384 bbox) = riesgo. 5x no es monotonic en benefit. 3x ya cubre masa. Re-evaluar Cutout en Version 2 post-eval. |
| Class imbalance | (no especificado) | **NO class weights** | Crasto 2024 demostró -1,9 pp loss weights single-stage. Mosaic+Mixup interno = +11,3 pp. |

---

## (7) Hallazgos cualitativos para informe IEEE

1. **Brecha de investigación documentada:** No existe paper publicado con ablación letterbox-vs-stretch en waste sorting con aspect ratios mixtos. Mi MVP puede aportar el delta empíricamente comparando Version 1-B (Fit-black 416) vs Version 1-B-alt (Stretch 416). Coste: 1 versión Premium Trial extra.

2. **Validación Crasto 2024 en dominio waste:** publicar comparación class weights vs mosaic+mixup en mi pipeline confirma o refuta su tesis sobre foreground-foreground imbalance en un dominio nuevo (waste sorting industrial Latam vs COCO-ZIPF).

3. **Cita Roboflow staff Leo (`discuss.roboflow.com #6892`):** *"use 'Stretch to' resizing option to maximize the limited input space"* (regla general) **EXCEPTO** *"the aspect ratio of the object (annotations) you are trying to detect matters for identifying it"* (excepción explícita). Plastic-bottle vs glass-jar es justamente el caso de excepción → Fit-black está justificado por su propio razonamiento.

4. **Discrepancia mediática:** la respuesta de Glenn Jocher en #7053 ("YOLOv8 handles mixed aspect ratios; you don't have to choose") asume que NO se fuerza resize en Roboflow. Si se fuerza (como hago yo para tener export determinístico a tamaño fijo), la elección Stretch vs Fit-black SÍ importa. Esta sutileza no está en docs oficiales.

---

## (8) Próximos pasos accionables

1. Generar Version 1-A en Roboflow UI con la spec sec. 5.2.
2. Generar Version 1-B en Roboflow UI con la spec sec. 5.3.
3. Verificar count de imágenes post-Filter Null: esperado ~10.439 imgs / ~10.439 bboxes (tras 3x aug → ~31.317 imágenes train-aug). Si Filter Null elimina mucho más de lo esperado (>3.000 imgs), revisar matriz de co-ocurrencia clase-imagen.
4. Exportar Version 1-A formato `tfrecord`, Version 1-B formato `yolov8`. Guardar zip md5 para reproducibilidad.
5. Lanzar Track A (Colab T4) y Track B (Kaggle T4) en paralelo con notebooks ya escritos.
6. Tras eval inicial, decidir si necesito Version 2 con Cutout o sin él.
7. **(Aporte IEEE)** Considerar generar Version 1-B-alt-stretch (Stretch 416) como ablación letterbox-vs-stretch publicable.

---

## Ronda 2 — 2026-05-05 (alto): Validación hardware-constrained

> **Pregunta motriz:** ¿los hallazgos de la ronda 1 se sostienen cuando se filtra la evidencia por hardware con limitaciones comparables a Jetson Nano B01? (Maxwell GPU 128 CUDA cores SIN tensor cores INT8, 4 GB unificada, JetPack 4.6.x, TRT 8.0/8.2, Cortex-A57 + NEON SIMD)

### Track A — papers académicos edge ML

#### Hallazgos cuantitativos clave

**Karimov et al. 2025 (arXiv:2508.19600, "Quantization Robustness to Input Degradations for Object Detection"):**

> "YOLO12n loses 7.2 pp mAP50-95 vs FP32 under Static INT8 TensorRT, while YOLO12x loses only 3.1 pp. Static INT8 models consistently showed increased sensitivity to noise."

→ Modelos escala nano son los MÁS penalizados por INT8. Para Track B INT8 (si se usara), esperar 5-8 pp degradación. Pero: **mi Track B NO usa INT8** (es FP16 TRT en Maxwell). Y mi Track A es TFLite INT8 con QAT (no PTQ simple).

> "The choice of calibration data (Clean vs. Mixed) for Static INT8 models had a minimal effect on both accuracy and speed when evaluated on this clean dataset, with only YOLO12x showing a slightly more pronounced mAP drop with mixed calibration."

→ **Refuta la hipótesis de que augmentations en repr dataset mejoren calibration.** Para repr dataset INT8: usar imágenes limpias del val split, NO aplicar Mosaic ni Cutout post-hoc.

**Yun & Wong CVPR 2021 (arXiv:2104.11849, "Do All MobileNets Quantize Poorly?"):**

> "MobileNet-V1 in ImageNet drops from 71.04% FP32 to 3.00% QUINT8 under simple PTQ (95.78% accuracy drop). The mechanism is dynamic range fluctuation between depthwise and pointwise layers, incompatible with per-layer single scale factor."

→ **Hallazgo más crítico de la ronda:** SSD MobileNet v2 FPNLite (mi Track A) usa depthwise separable convolutions. Sin QAT, PTQ simple PUEDE FALLAR catastróficamente. Mi `pipeline_custom.config` ya tiene `quantization_aware_training: true` con `delay: 2000` — esa decisión era arquitectónicamente correcta, ahora confirmada con evidencia cuantitativa.

**Jacob et al. CVPR 2018 (arXiv:1712.05877, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"):**

> "Our integer-only quantization scheme allows MobileNet SSD to achieve face detection at 36 FPS on a single big core (Snapdragon 835), comparable to floating-point models while reducing latency by up to 50%."

→ Validación QAT para MobileNet SSD. Mi diseño Track A se valida.

**Alqahtani et al. 2024 (arXiv:2409.16808, "Benchmarking Deep Learning Models for Object Detection on Edge Computing Devices"):**

> "When YOLOv8 was scaled down from 640×640 to 320×320 for Coral Edge TPU compatibility, mAP dropped from 44 to 16 — a loss of 28 absolute mAP points."

→ **Resoluciones cuadráticamente afectan mAP, no solo FPS.** Saltos grandes (640→320) son catastróficos. 416 (mi target Track B) es middle-ground razonable.

**Chiam et al. 2025 ("Energy Optimized YOLO: Quantized Inference for Real-Time Edge AI Object Detection"):**

> "The quantized YOLOv7-tiny model with FP16 quantization and GPU acceleration achieves a processing speed of 38 FPS with mAP of 46.3%, while maintaining a low power consumption of 5.1W [Jetson Nano]."

→ Validación viabilidad YOLO-tiny FP16 GPU Maxwell. Mi target ≥10 FPS sostenido tiene margen 4× para concurrencia con I/O servos.

**Boddu & Mukherjee 2025 (arXiv:2506.09300, "Efficient Edge Deployment of Quantized YOLOv4-Tiny for Aerial Emergency Object Detection on Raspberry Pi 5"):**

> "A representative dataset of 100 aerial images was selected from the training set. TensorFlow Lite's representative dataset gen API was used to calibrate the quantization ranges. Inference time: 28.2 ms (vs 262 ms FP32) — 9.3× speedup."

→ Confirma que 100 muestras pueden ser suficientes para repr dataset si están bien distribuidas. Mi 200 era razonable; subir a 300-500 es defensa adicional.

**Swaminathan et al. 2024 (arXiv:2406.17749, "Benchmarking Deep Learning Models on NVIDIA Jetson Nano for Real-Time Systems"):**

> "JetPack 4.6.1, TensorRT (L4T R32.7.1), Maxwell GPU 128 CUDA cores. Average TensorRT speedup: 7.01×. MobileNet-V2: up to 16.7× speedup post-TensorRT."

→ **Plataforma EXACTA confirmada (mi setup B01 + JetPack 4.6.1).** Speedup TRT 7-16× valida la decisión de TensorRT para Track B.

#### Tabla de validación final por decisión

| # | Decisión .md ronda 1 | Status post-ronda 2 | Evidencia clave | Acción refinada |
|---|---|---|---|---|
| 1 | imgsz=416 Track B | **VALIDADA cuantitativamente** | Nature 2024 Table 4: 416=30 FPS, 640=24 FPS (TRT FP16); Špeh 2023 YOLOv7-tiny: 416=17, 640=9 (1.9×); Alqahtani 2024: 320=catastrófico -28pp mAP | Mantener 416. Programar ablation propia 416 vs 640 sobre dataset waste-3class para confirmar el delta en mi domain. |
| 2 | NO class weights | **GAP en edge, pero validada por evidencia desktop + Ultralytics #20259** | Crasto 2024 desktop: -1.9 pp; Ultralytics #20259: mosaic dificulta class weights mecánicamente; Karimov 2025: INT8 amplifica diferencias entre clases con dist. confianza distinta | Mantener decisión. Ablation post-eval con/sin para validar empíricamente. |
| 3 | Repr dataset 200 muestras | **REFINADA: subir a 300-500** | Google LiteRT: 100-500 ok; Glenn Jocher #14121: 1000+ recomendado; Boddu&Mukherjee 2025 Pi 5: 100 ok; Yun&Wong: variación calibration matters en DWSCNN | **Cambio**: usar 300-500 muestras del val split distribuidas proporcionalmente entre plastic/glass/paper. Verificar `ultralytics > 8.4.31`. |
| 4 | Resize Fit-black 320/416 | **VALIDADA con matiz**: padding=0 vs Ultralytics 114 = mismatch teórico no medido (GAP) | Roboflow staff Leo (cardboard ejemplo); Ultralytics #7053; PR #21652 ago 2025 padding configurable; razonamiento teórico Krishnamoorthi 2018 | Mantener Fit-black. Mitigación opcional inferencia: documentar elección padding=0 vs 114. Aporte IEEE: ablation propia sobre delta. |
| 5 | NO Mosaic Roboflow + SÍ Mosaic Ultralytics | **VALIDADA cuantitativamente** | Roboflow doc oficial verbatim "do not recommend with YOLOv8"; AliHamzaAzam TACO +17% mAP con mosaic; techishthoughts "essential <891 imgs"; Karimov 2025 mixed-calib NO mejora pero NO daña | Sin cambios. Mosaic=1.0 (default Ultralytics) en Track B; offline para Track A si necesario. |
| 6 | NO class weights single-stage | (consolida #2) | Ver #2 | Ver #2 |
| 7 | NO Cutout V1 | **VALIDADA defensiva** (gap edge confirmado) | Yim 2025 OOC en VisDrone (no edge); Karimov 2025 mixed-calib refuta beneficio; mecánica análoga a letterbox padding (sesgo histograma activations) | Sin cambios. Re-evaluar V2 si false negatives clase paper post-eval real. |
| 8 | QAT delay=2000 Track A | **CONFIRMADA OBLIGATORIA** | Yun&Wong 2021: MobileNet sin QAT cae 71%→3% (95.78% drop) en ImageNet QUINT8; Jacob 2018 CVPR: QAT recupera precisión FP32; Sander 2025: "PTQ often unacceptable degradation in quantization-sensitive tasks" | Mantener `quantization_aware_training: true` y `delay: 2000` activo en `pipeline_custom.config`. Es CRÍTICO, no nice-to-have. |
| 9 | Track A INT8 hace sentido | **VALIDADA con reframing** | Maxwell sin tensor cores INT8 (Qengineering verbatim, Pascal+ tienen DP4A, Turing+ tienen INT8 VNNI); pero TFLite INT8 corre en CPU XNNPACK + NEON SIMD ARM Cortex-A57 que SÍ acelera INT8 vía instrucciones SIMD | **Reframing dual-track:** NO es "INT8 vs FP16", es **"backend CPU+SIMD vs GPU+CUDA"**. Track A vive en CPU (TFLite INT8), Track B en GPU (TensorRT FP16). Comparación de hardware backend. |

### Track B — repos benchmarks edge

#### Tabla benchmarks Jetson Nano B01 confirmados

| Fuente | Modelo | Hardware | Backend | imgsz | Latencia/FPS | mAP |
|---|---|---|---|---|---|---|
| **Nature Sci Rep 2024** (DOI 10.1038/s41598-024-74798-3 Tabla 4) | YOLOv8n | Jetson Nano | TensorRT FP16 | 416 | **30 FPS** | n/r |
| Idem | YOLOv8n | Jetson Nano | TensorRT FP16 | 512 | 29 FPS | n/r |
| Idem | YOLOv8n | Jetson Nano | TensorRT FP16 | 640 | 24 FPS | n/r |
| Idem | YOLOv8n | Jetson Nano | PyTorch | 416 | 16 FPS | n/r |
| Idem | YOLOv8n | Jetson Nano | PyTorch | 640 | 13 FPS | n/r |
| **Qengineering** (`src/main.cpp:14`) | YOLOv8n | Jetson Nano B01 | TRT FP16 (C++) | 640 | 19 FPS | n/r |
| Idem | YOLOv8s | Jetson Nano B01 | TRT FP16 (C++) | 640 | 9.25 FPS | n/r |
| **NobuoTsukamoto/benchmarks** | SSD MobileNet v2 | Jetson Nano JP4.6.1 | TRT FP16 | 320 | 41.1 FPS (24.31 ms) | 20.18 |
| Idem | **SSD MobileNet v2 FPNLite** | Jetson Nano JP4.6.1 | TRT FP16 | 320 | **26.3 FPS (37.99 ms)** | **21.97** |
| Idem | EfficientDet-Lite0 | Jetson Nano JP4.6.1 | TRT FP16 | 320 | 24.6 FPS | mAP@50=39.26 |
| Idem | EfficientDet-Lite1 | Jetson Nano JP4.6.1 | TRT FP16 | 384 | 13.4 FPS | n/r |
| Idem | EfficientDet-Lite2 | Jetson Nano JP4.6.1 | TRT FP16 | 448 | 9.3 FPS | n/r |
| **Tony607** | SSD MV2 | Jetson Nano JP4.3 | TF-TRT | 300 | 20+ FPS | n/r |
| **jkjung-avt/tensorrt_demos** | SSD MobileNet v1 | Jetson Nano | TRT async | 300 | 27-28 FPS | n/r |
| Idem | YOLOv4-tiny | Jetson Nano | TRT | 416 | 4.6 FPS | n/r (modelo más pesado) |
| **Špeh Medium 2023** | YOLOv7-tiny | Jetson Nano | TRT FP16 | 416 | 17 FPS | n/r |
| Idem | YOLOv7-tiny | Jetson Nano | TRT FP16 | 640 | 9 FPS | n/r |
| **Chiam et al. 2025** | YOLOv7-tiny | Jetson Nano | FP16 GPU | n/r | 38 FPS, 5.1 W | mAP 46.3% |
| **imnuman/jetson-object-detection** | YOLOv8n | Jetson Nano 4GB | TRT FP16 | 640 | 28 FPS | n/r |
| **ReadyTensor 2024** | YOLOv8n | Jetson Nano | PyTorch CUDA | n/r | 6 FPS (163-170ms) | n/r |

**Patrones identificados:**

- **Maxwell INT8 NO acelera GPU.** Confirmado por Qengineering verbatim: *"The int8 models don't give any increase in FPS, while, at the same time, their mAP is significantly worse."* Mecanismo: Maxwell carece de unidades INT8 VNNI (introducidas desde Turing/Ampere). TRT compila pero no acelera.
- **Pero TFLite INT8 SÍ acelera CPU.** ARM Cortex-A57 con NEON SIMD acelera operaciones INT8 packed. Mi Track A es viable.
- **FPNLite TRT engine en JP4.6 requiere fix:** Issue NVIDIA/TensorRT#1994 documenta workaround. NO me afecta porque Track A va TFLite (no TRT), pero documentar.
- **Inferencia pura vs pipeline completo:** Qengineering mide solo `Infer()` (kernel inference). Pipeline real (captura + letterbox + post-processing) reduce 30-50% FPS. Mi target real ≥10 FPS necesita modelo con bench FPS ≥15-20.

#### Implementaciones de referencia para Track A

- **`tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v2_fpnlite_quantized_shared_box_predictor_256x256_depthmultiplier_75_coco14_sync.config`** — config oficial QAT (256×256, depth_mult=0.75, mAP 20.0 COCO14 minival). Template directo para adaptar a mi 320×320 sin depth_mult.
- **`tensorflow/models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config`** — config base 320×320 (mAP 22.2 COCO17, sin QAT integrado). Mi `pipeline_custom.config` debe combinar este 320×320 + el bloque QAT del 256×256.

### Track C — discusiones comunidad

**Ultralytics #14121 (Glenn Jocher CEO):**

> "Generally, for INT8 calibration, using at least 1000 images from your dataset is advised to minimize any significant drop in accuracy. When you specify a dataset YAML file for INT8 calibration, the calibration process typically uses the validation set (val) defined in your YAML file."

**Ultralytics #20259 (waste detection 15 clases imbalance Bottle:719 vs Straw:1):**

> "@Jordan-Pierce I think that would require deeper modifications in several places to make it work. It's also complicated by mosaic augmentation which mixes 4 images into one."

→ Class weights con mosaic activo es no-trivial mecánicamente.

**Ultralytics #6478 (caso real degradación):**

> mAP ONNX 0.678 → TFLite INT8: 0.318 (caída 53% en SKU-110K)

Causa documentada en PR #1695 (jul 2023, merged): normalización ImageNet aplicada incorrectamente al repr dataset. **Bug ya fixed**. Confirmar versión Ultralytics actualizada.

**Ultralytics PR #21652 (ago 2025, merged):**

> "Support `padding_value` and `interpolation` in `LetterBox` for better compatibility. Default remains 114."

Hasta ago 2025 padding=114 era hardcoded. Configurable desde entonces.

**Ultralytics PR #24028 (mar 2026, merged):**

> "Updated INT8 calibration dataloader creation in `ultralytics/engine/exporter.py` to use `max(self.imgsz)` instead of only `self.imgsz[0]`. Added logic to preserve the full target image shape for `LetterBox` transforms when exporting with non-square image sizes."

→ Bug fix reciente. Verificar `ultralytics > 8.4.31` para evitarlo.

**ultralytics/yolov5 #8762 (degradación TFLite INT8 sin causa identificada):**

- YOLOv5m: mAP 0.831 → TFLite INT8 0.633 (-24%)
- YOLOv5s: mAP 0.809 → TFLite INT8 0.681 (-16%)

Sin respuesta técnica oficial. Mi Track A debe asumir 15-25% mAP gap si NO usa QAT. **Con QAT (mi caso), Jacob 2018 garantiza recuperar a comparable-FP32.**

### Cambios concretos a la spec Generate Version 1 post-ronda 2

| Sección | Sin cambio | Cambio |
|---|---|---|
| Auto-Orient | ✅ ON | — |
| Modify Classes + Filter Null | ✅ ON | — |
| Resize Track A | ✅ Fit-black 320×320 | — |
| Resize Track B | ✅ Fit-black 416×416 | — |
| Augmentations 3x | ✅ Flip H, Rot ±15°, Shear ±2°, Brightness ±25%, Exposure ±15%, Saturation ±20%, Blur 1.5px, Noise 5%, MotionBlur leve | — |
| Hue, Cutout, Mosaic, Bbox-Crop, Tile, Grayscale | ✅ OFF | — |
| Train/Test Split | ✅ 70/20/10 estratificado | — |
| Repr dataset INT8 (Track A) | — | **AJUSTE**: 300-500 muestras del val split (no 200), distribuidas proporcionalmente entre las 3 clases. Mantener QAT delay=2000 en `pipeline_custom.config`. |
| **Padding inferencia Jetson** | — | **NUEVO**: documentar elección padding value en pipeline inferencia. Por consistencia con Roboflow Fit-black=0, aplicar `cv2.copyMakeBorder(value=(0,0,0))` en el preprocesamiento del cliente Jetson. **No** sobre-escribir Ultralytics LetterBox interno (que usa 114) si se usa Ultralytics directo en Jetson. **Sí** alinear si se usa TRT engine custom. |
| **Versión `ultralytics`** | — | **NUEVO**: instalar `ultralytics > 8.4.31` para evitar bug PR #24028 INT8 calibration con imgsz no-cuadrado. |
| **Reframing narrativa informe IEEE** | — | **NUEVO**: dual-track = "TFLite INT8 (CPU XNNPACK + NEON SIMD ARM) vs TensorRT FP16 (GPU Maxwell)". NO "INT8 vs FP16". |

### Ablations propuestas para llenar gaps confirmados (aporte IEEE)

Tres mini-experimentos sobre mi propio dataset y hardware, ordenados por valor incremental:

**Ablation #1 — imgsz 416 vs 640 en Track B sobre dataset waste-3class:**
- Generar dos versiones Roboflow: 1-B (416×416 Fit-black) y 1-B-alt-640 (640×640 Fit-black).
- Entrenar YOLOv8n con `imgsz=416` y otra corrida con `imgsz=640` sobre la misma versión.
- Medir mAP@0.5 y mAP@0.5:0.95 sobre test split idéntico.
- Medir FPS Jetson Nano TRT FP16 con `bench_jetson.py`.
- Esperado (consistente con Nature 2024): 416 → ≥30 FPS, 640 → ≥24 FPS, mAP gap <5 pp.
- Coste: 1 versión Roboflow extra (Premium Trial OK) + 2 entrenamientos paralelos Kaggle T4.

**Ablation #2 — class weights ON vs OFF en Track B:**
- Mismo dataset Version 1-B.
- Corrida 1: `model.train(..., class_weights=None)` (default).
- Corrida 2: `model.train(...)` con custom dataloader que pondera plastic/glass/paper inversamente proporcional. Implementar el dataloader custom (Ultralytics no soporta nativo con mosaic activo).
- Medir mAP por clase, especialmente paper (clase minoritaria).
- Esperado (Crasto 2024): class weights -0.6 a -1.9 pp mAP global. Si en mi waste domain confirma, descartar definitivamente. Si invierte (mejora paper class), DWaste 2025 vindicado.

**Ablation #3 — letterbox vs stretch (aporte IEEE letterbox-vs-stretch waste detection):**
- Generar Version 1-B (Fit-black 416) y Version 1-B-alt-stretch (Stretch 416).
- Entrenar YOLOv8n con `imgsz=416` ambos sobre splits idénticos.
- Medir mAP@0.5 sobre test split.
- Esperado (consistente con Ultralytics #7454): letterbox > stretch por 4-5 pp.
- Coste: 1 versión Roboflow extra + 1 entrenamiento. Documentar como minor contribution publicable: brecha de literatura confirmada en ronda 1 (no existe ablation publicada en waste sorting con aspect ratios mixtos).

### Brechas residuales nuevas (post-ronda 2)

1. **GAP confirmado a literatura:** ablation imgsz 416 vs 640 específicamente sobre YOLOv8n + Jetson Nano B01 + Maxwell. Nature 2024 lo cubre pero sin especificar JetPack/TRT version. Mi ablation #1 cierra el gap.
2. **GAP confirmado:** efecto del padding value (0 vs 114) sobre INT8 calibration. Sin paper publicado. No crítico para Track B (FP16, padding no afecta tanto). Para Track A QAT debería ser robusto al padding via training samples diversos.
3. **GAP confirmado:** class weights post-INT8 quantization en edge. No crítico para Track A (QAT learn quantization-friendly weights). Mi ablation #2 es tentative.
4. **GAP confirmado:** Cutout/Random Erasing en repr dataset INT8. Mi decisión de NO Cutout V1 sigue siendo defensiva razonable.
5. **Validado closure:** la combinación QAT 320×320 + Jetson Nano B01 + waste detection + Roboflow es laguna en literatura pública (Track A2 confirma verbatim). Mi proyecto puede llenarla = aporte IEEE confirmado.

---

## Fuentes consultadas (acumulado ronda 1)

| # | Título | URL | Tipo | Confianza |
|---|---|---|---|---|
| 1 | Roboflow Docs: Preprocess Images | https://docs.roboflow.com/datasets/dataset-versions/image-preprocessing | Doc oficial | Alta |
| 2 | Roboflow Docs: Image Augmentation | https://docs.roboflow.com/datasets/dataset-versions/image-augmentation | Doc oficial | Alta |
| 3 | Roboflow Docs: Mosaic Augmentation | https://docs.roboflow.com/datasets/dataset-versions/image-augmentation/augmentation-types/mosaic-augmentation | Doc oficial | Alta |
| 4 | Roboflow Docs: Training Resolutions by Model Type | https://docs.roboflow.com/train/training-resolutions-by-model-type | Doc oficial | Alta |
| 5 | Roboflow Docs: Premium Trial | https://docs.roboflow.com/billing/premium-trial | Doc oficial | Alta |
| 6 | Roboflow Blog: What Is Image Preprocessing and Augmentation? | https://blog.roboflow.com/why-preprocess-augment/ | Blog oficial | Media |
| 7 | Roboflow Blog (Joseph Nelson, CEO): You Might Be Resizing Your Images Incorrectly (Nov 2025) | https://blog.roboflow.com/you-might-be-resizing-your-images-incorrectly/ | Blog oficial | Alta |
| 8 | Roboflow Inference: letterbox_image() helper | https://inference.roboflow.com/reference/inference/core/utils/preprocess | Doc + código | Alta |
| 9 | Roboflow staff Leo en discuss.roboflow.com #6892 (best resize method YOLOv8) | https://discuss.roboflow.com/t/selecting-the-best-image-resizing-method-in-roboflow-for-training-a-yolov8/6892 | Foro oficial | Alta |
| 10 | discuss.roboflow.com #7651: Roboflow Stretch usa cv2.INTER_AREA | https://discuss.roboflow.com/t/how-does-roboflow-resize-stretch-its-images/7651 | Foro comunidad | Media |
| 11 | Ultralytics Docs: Configuration (cfg/) | https://docs.ultralytics.com/usage/cfg/ | Doc oficial | Alta |
| 12 | Ultralytics default.yaml (rect, mosaic, imgsz) | https://github.com/ultralytics/ultralytics | Código fuente | Alta |
| 13 | Ultralytics LetterBox class (data/augment.py) | https://github.com/ultralytics/ultralytics | Código fuente | Alta |
| 14 | Ultralytics Issue #7053 (ProBroSam, mixed aspect ratio + Roboflow) | https://github.com/ultralytics/ultralytics/issues/7053 | Issue oficial | Alta |
| 15 | YOLOv5 Issue #7454 (zxq309, letterbox doble -4-5pp) | https://github.com/ultralytics/yolov5/issues/7454 | Issue oficial | Media |
| 16 | YOLOv5 Issue #1279 (Glenn Jocher, padding 114 = ImageNet mean) | https://github.com/ultralytics/yolov5/issues/1279 | Issue oficial | Alta |
| 17 | YOLOv5 Issue #8590 (rect val.py half-stride border) | https://github.com/ultralytics/yolov5/issues/8590 | Issue oficial | Media |
| 18 | TF OD API: image_resizer.proto | https://github.com/tensorflow/models/blob/master/research/object_detection/protos/image_resizer.proto | Código fuente | Alta |
| 19 | TF OD API: ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config | https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config | Config oficial | Alta |
| 20 | Lin et al. 2017, "Focal Loss for Dense Object Detection" | https://arxiv.org/abs/1708.02002 | Paper ICCV | Alta |
| 21 | Zhong et al. 2017/AAAI 2020, "Random Erasing Data Augmentation" | https://arxiv.org/abs/1708.04896 | Paper AAAI | Alta |
| 22 | DeVries & Taylor 2017, "Cutout" | https://arxiv.org/abs/1708.04552 | Preprint | Media |
| 23 | Bochkovskiy et al. 2020, "YOLOv4" (mosaic +2.3pp) | https://arxiv.org/abs/2004.10934 | Paper | Alta |
| 24 | Wang et al. 2022/CVPR 2023, "YOLOv7" | https://arxiv.org/abs/2207.02696 | Paper CVPR | Alta |
| 25 | Buslaev et al. 2018, "Albumentations" | https://arxiv.org/abs/1809.06839 | Paper | Alta |
| 26 | Bashkirova et al. 2022, "ZeroWaste" CVPR | https://arxiv.org/abs/2106.15279 | Paper CVPR | Alta |
| 27 | **Crasto 2024 (PAPER ORO)**, "Class Imbalance in Object Detection: An Experimental Diagnosis" | https://arxiv.org/abs/2403.07113 | Technical report | Alta |
| 28 | Yim et al. 2025, "Adaptive Grid Selection Training Strategy for Tiny Object Detection" (Object-Oriented Cutout) | https://doi.org/10.1109/ACCESS.2025.3529234 | Paper IEEE Access | Media |
| 29 | Gong et al. CVPR 2021, "KeepAugment" | https://arxiv.org/abs/2011.11786 | Paper CVPR | Alta |
| 30 | Kunwar 2025, "DWaste: Greener AI for Waste Sorting using Mobile and Edge Devices" | https://arxiv.org/abs/2510.18513 | Preprint | Media |
| 31 | Demetriou et al. 2023, "Real-time Construction Demolition Waste Detection" | Semantic Scholar / Waste Management (Elsevier) | Paper Q1 | Media |
| 32 | Korkmaz 2026, "A Systematic Evaluation of Photometric Data Augmentation Combinations" | Semantic Scholar | Paper revista | Media |
| 33 | boss4848/waste-detection (49⭐) | https://github.com/boss4848/waste-detection | Repo GitHub | Media |
| 34 | Gokzz-glitch/zerowastex (data.yaml) | https://github.com/Gokzz-glitch/zerowastex/blob/main/ml/dataset.yaml | Repo GitHub | Media |
| 35 | andresriverosb1331/Trash_Classifier (Chile, banda) | https://github.com/andresriverosb1331/Trash_Classifier | Repo GitHub | Media |
| 36 | starwit/waste-detection (DVC pipeline) | https://github.com/starwit/waste-detection | Repo GitHub | Media |
| 37 | Qengineering/YoloV8-TensorRT-Jetson_Nano (benchmarks 19 FPS FP16) | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo GitHub | Alta |
| 38 | ecoCrafters/waste-detection (SSD FPNLite 320 + TACO) | https://github.com/ecoCrafters/waste-detection | Repo GitHub | Baja (1⭐, sin Roboflow) |
| 39 | dbash/zerowaste (CVPR 2022) | https://github.com/dbash/zerowaste | Repo GitHub | Alta |
| 40 | pedropro/TACO (729⭐) | https://github.com/pedropro/TACO | Repo GitHub | Alta |
| 41 | MeetShroff/YOLOv8-Based-Waste-Detection-System-for-Recycling-Plants | https://github.com/MeetShroff/YOLOv8-Based-Waste-Detection-System-for-Recycling-Plants | Repo GitHub | Baja |
| 42 | Boston-University-Projects/EC523_DL_CV_Project (YOLOR + Roboflow) | https://github.com/Boston-University-Projects/EC523_DL_CV_Project | Repo GitHub | Media |
| 43 | Someshvar2408/Waste-detection-on-railway-platforms-using-YOLOv8 | https://github.com/Someshvar2408/Waste-detection-on-railway-platforms-using-YOLOv8 | Repo GitHub | Baja |
| 44 | BaraaLazkani/trash-detection-yolov8 | https://github.com/BaraaLazkani/trash-detection-yolov8 | Repo GitHub | Baja |
| 45 | AtenVisarut/Smartbin_TrashDetection-yolov8 | https://github.com/AtenVisarut/Smartbin_TrashDetection-yolov8 | Repo GitHub | Baja |
| 46 | Roboflow/supervision Issue #505 (letterbox feature request) | https://github.com/roboflow/supervision/issues/505 | Issue oficial | Media |
| 47 | Ultralytics Issue #14983 (letterbox auto=false) | https://github.com/ultralytics/ultralytics/issues/14983 | Issue oficial | Media |
| 48 | Ultralytics Issue #3344 (Roboflow 640 + YOLOv8 manual resize question) | https://github.com/ultralytics/ultralytics/issues/3344 | Issue oficial | Media |
| 49 | Roboflow/rf-detr augmentations docs | https://github.com/roboflow/rf-detr/blob/a77670be/docs/learn/train/augmentations.md | Doc/código | Media |
| 50 | Roboflow/rf-detr Issue #414 (preprocessing stretch vs padding) | https://github.com/roboflow/rf-detr/issues/414 | Issue | Media |
| 51 | albumentations-team/albumentations (LongestMaxSize + PadIfNeeded) | https://github.com/albumentations-team/albumentations | Repo | Alta |
| 52 | zhunzhong07/Random-Erasing | https://github.com/zhunzhong07/Random-Erasing | Repo | Alta |
| 53 | dergipark.org.tr — solid waste 10-class YOLOv8 + Roboflow | https://dergipark.org.tr/en/download/article-file/5020373 | Paper revista TR | Media |
| 54 | Eva Urankar 2025 — Waste Detection Mobile Devices Performance | International Journal of Science and Research Archive 15(1) | Paper revista | Media |
| 55 | Toribio et al. 2026 — e-YOLOv6 Sign Language (adaptive letterbox +12.5pp) | (sin DOI) Intl. J. for Multidisciplinary Research | Paper revista | Baja |

### Fuentes nuevas ronda 2 (hardware-constrained edge)

| # | Título | URL | Tipo | Confianza |
|---|---|---|---|---|
| 56 | **Karimov et al. 2025**, "Quantization Robustness to Input Degradations for Object Detection" | https://arxiv.org/abs/2508.19600 | arXiv preprint | Media-Alta |
| 57 | **Yun & Wong CVPR 2021**, "Do All MobileNets Quantize Poorly? Insights into Effect of Quantization on DWSCNN" | https://arxiv.org/abs/2104.11849 | Paper CVPR Workshop | Alta |
| 58 | **Jacob et al. CVPR 2018**, "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" | https://arxiv.org/abs/1712.05877 | Paper CVPR | Alta |
| 59 | Krishnamoorthi 2018 (Google), "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper" | https://arxiv.org/abs/1806.08342 | Whitepaper | Alta |
| 60 | **Alqahtani et al. 2024 (ICSOC)**, "Benchmarking Deep Learning Models for Object Detection on Edge Computing Devices" | https://arxiv.org/abs/2409.16808 | Paper ICSOC | Alta |
| 61 | Swaminathan et al. 2024, "Benchmarking Deep Learning Models on NVIDIA Jetson Nano for Real-Time Systems" | https://arxiv.org/abs/2406.17749 | Paper Procedia CS | Alta |
| 62 | Chiam et al. 2025, "Energy Optimized YOLO: Quantized Inference for Real-Time Edge AI Object Detection" | Semantic Scholar | Paper revista | Alta |
| 63 | Boddu & Mukherjee 2025, "Efficient Edge Deployment of Quantized YOLOv4-Tiny for Aerial Emergency on RPi 5" | https://arxiv.org/abs/2506.09300 | arXiv preprint | Media |
| 64 | Tariq & Javed 2025, "Small Object Detection with YOLO: Performance Analysis Across Versions and Hardware" | https://arxiv.org/abs/2504.09900 | arXiv preprint | Media |
| 65 | Sander et al. 2025, "On Accelerating Edge AI: Optimizing Resource-Constrained Environments" | https://arxiv.org/abs/2501.15014 | arXiv preprint | Media |
| 66 | Guerrouj et al. 2025 (IJACSA), "Quantized Object Detection for Real-Time Inference on Embedded GPU Architectures" | Semantic Scholar | Paper revista | Alta |
| 67 | Mittal et al. 2024 (Springer AI Review), "Comprehensive Survey of Lightweight Object Detection for Edge" | https://link.springer.com/article/10.1007/s10462-024-10877-1 | Survey Q1 | Alta |
| 68 | **Nature Scientific Reports 2024 (Tabla 4)**, "Algorithm for detecting surface defects in wind turbines based on lightweight YOLO" — única tabla pública YOLOv8n imgsz vs FPS Jetson Nano | https://www.nature.com/articles/s41598-024-74798-3 | Paper Q1 | Alta |
| 69 | **NobuoTsukamoto/benchmarks** EfficientDet + SSD MV2 FPNLite Jetson Nano JP4.6.1 TRT FP16 | https://github.com/NobuoTsukamoto/benchmarks/blob/main/tensorrt/jetson/detection/README.md | Repo benchmarks | Alta |
| 70 | Špeh Medium 2023, "YOLOv7 with TensorRT on Jetson Nano" — 17 FPS@416 vs 9 FPS@640 | https://medium.com/@jurespeh/yolov7-with-tensorrt-on-jetson-nano-with-python-script-example-63099fa7c8a5 | Blog técnico | Media |
| 71 | imnuman/jetson-object-detection (28 FPS YOLOv8n@640 Jetson Nano 4GB TRT FP16) | https://github.com/imnuman/jetson-object-detection | Repo GitHub | Media |
| 72 | Tony607/jetson_nano_trt_tf_ssd (SSD MV2 TF-TRT 20+ FPS Jetson Nano JP4.3) | https://github.com/Tony607/jetson_nano_trt_tf_ssd | Repo GitHub | Media |
| 73 | jkjung-avt/tensorrt_demos (SSD MV1 27-28 FPS async Jetson Nano TRT) | https://github.com/jkjung-avt/tensorrt_demos | Repo GitHub | Alta |
| 74 | dusty-nv/jetson-inference (SSD MobileNet v1/v2 TRT, oficial NVIDIA) | https://github.com/dusty-nv/jetson-inference | Repo NVIDIA | Alta |
| 75 | Anand 2024 (ReadyTensor), "Accelerating Edge Vision: YOLOv8 Object Detection on Jetson Nano" — 6 FPS PyTorch baseline | https://app.readytensor.ai/publications/accelerating-edge-vision-yolov8-object-detection-on-jetson-nano-4D88m4ggztQt | Blog/reporte | Media |
| 76 | Li & Grammenos UCL 2022 (arXiv:2210.00448), "Smart Recycling Bin on Jetson Nano" — caso más cercano (MobileNet V3 classification) | https://arxiv.org/pdf/2210.00448 | Paper arXiv | Alta |
| 77 | Prometeo.blog 2025, "Recyclables classifier with TensorRT Jetson Nano 4GB" — ~20 FPS, 80ms | https://prometeo.blog/en/practical-case-recyclables-classifier-with-tensorrt/ | Blog técnico | Media |
| 78 | AliHamzaAzam/vision-dl-waste-detection (TACO + YOLOv8s, mosaic +17% mAP) | https://github.com/AliHamzaAzam/vision-dl-waste-detection | Repo GitHub | Media |
| 79 | Foro NVIDIA: "Object detection on Nano with yolov8 model" (>30fps YOLOv5, JetPack4 EOL) | https://forums.developer.nvidia.com/t/object-detection-on-nano-with-yolov8-model/275370 | Foro oficial NVIDIA | Alta |
| 80 | Ultralytics issue #14530 (Ahelsamahy: imgsz reduce latencia en Nano) | https://github.com/ultralytics/ultralytics/issues/14530 | Issue GitHub | Media |
| 81 | Ultralytics issue #14121 (Glenn Jocher: INT8 calibration usa val, recomienda 1000 imgs) | https://github.com/ultralytics/ultralytics/issues/14121 | Issue GitHub | Alta |
| 82 | Ultralytics issue #20259 (waste imbalance + mosaic dificulta class weights) | https://github.com/ultralytics/ultralytics/issues/20259 | Issue GitHub | Alta |
| 83 | Ultralytics issue #6478 (SKU-110K mAP 0.678→0.318 TFLite INT8) | https://github.com/ultralytics/ultralytics/issues/6478 | Issue GitHub | Media |
| 84 | Ultralytics PR #1695 (jul 2023, fix normalización ImageNet en calibration) | https://github.com/ultralytics/ultralytics/pull/1695 | PR GitHub | Alta |
| 85 | **Ultralytics PR #21652** (ago 2025, padding_value configurable en LetterBox) | https://github.com/ultralytics/ultralytics/pull/21652 | PR GitHub | Alta |
| 86 | **Ultralytics PR #24028** (mar 2026, fix INT8 calibration con imgsz no-cuadrado) | https://github.com/ultralytics/ultralytics/pull/24028 | PR GitHub | Alta |
| 87 | Ultralytics yolov5 issue #8762 (YOLOv5m mAP 0.831→0.633 INT8 sin causa documentada) | https://github.com/ultralytics/yolov5/issues/8762 | Issue GitHub | Media |
| 88 | NVIDIA/TensorRT issue #1994 (SSD MV2 FPNLite TRT engine Jetson Nano JP4.6 fix) | https://github.com/NVIDIA/TensorRT/issues/1994 | Issue GitHub | Alta |
| 89 | google-coral/pycoral benchmarks RPi 4 (SSD MV1 15.1ms, MV2 4.5ms con USB Coral) | https://github.com/google-coral/pycoral/blob/master/benchmarks/reference/inference_reference_rp4b.csv | Benchmark oficial | Alta |
| 90 | tensorflow/models — `ssd_mobilenet_v2_fpnlite_quantized_shared_box_predictor_256x256_depthmultiplier_75_coco14_sync.config` (template QAT FPNLite oficial) | https://github.com/tensorflow/models | Config oficial | Alta |
| 91 | Google LiteRT Docs: Post-training quantization (verbatim "100-500 samples sufficient") | https://ai.google.dev/edge/litert/models/post_training_quantization | Doc oficial | Alta |
| 92 | markaicode.com 2026, "TensorFlow TFLite Edge Deployment" (1.7 pp drop typical) | https://www.markaicode.com/tensorflow-tflite-edge-deployment/ | Blog técnico | Media |
| 93 | AllanK24/QRID — código replicación Karimov et al. 2025 (calibración mixed dataset) | https://github.com/AllanK24/QRID | Repo GitHub | Alta |
| 94 | Qengineering/YoloV8-ncnn-Raspberry-Pi-4 (tabla multi-hardware Jetson Nano + RPi 4) | https://github.com/Qengineering/YoloV8-ncnn-Raspberry-Pi-4 | Repo GitHub | Alta |
| 95 | Qengineering/YoloV5-ncnn-Jetson-Nano (tabla benchmark 17 modelos 320-640) | https://github.com/Qengineering/YoloV5-ncnn-Jetson-Nano | Repo GitHub | Alta |

---

## Conceptos clave

- **LetterBox:** Resize que preserva aspect ratio escalando lado largo a target size, rellenando lado corto con padding constante. Padding 114 (ImageNet mean) en Ultralytics; 0 (negro) en Roboflow Fit-black; 255 (blanco) en Fit-white. Los tres son válidos pero Fit-black es el más cercano funcionalmente a padding 114.
- **Train/inference mismatch:** Discordancia entre preprocessing en training y en inference. Causa más frecuente de degradación silenciosa de mAP. Documentado empíricamente en YOLOv5 #7454 con caída 4-5 pp.
- **Foreground-foreground class imbalance:** Desequilibrio entre clases de objeto (plastic vs paper vs glass), distinto del foreground-background imbalance. Crasto 2024 demostró que mosaic+mixup mitigan, mientras class weights y oversampling son contraproducentes en single-stage detectors.
- **Object-Oriented Cutout (OOC):** Variante de Cutout/Random Erasing que respeta bounding boxes pequeños para preservar información semántica. Yim et al. 2025 reportó +0.6 pp mAP sobre YOLOv5 baseline en VisDrone tiny objects.
- **`fixed_shape_resizer` vs `keep_aspect_ratio_resizer` (TF OD API):** El primero aplica stretch, requiere shape estática (compatible TFLite INT8). El segundo preserva aspect ratio pero produce shape variable si `pad_to_max_dimension=false` (incompatible TFLite INT8 estático).
- **`rect=True` (Ultralytics):** Modo rectangular que agrupa imágenes por aspect ratio similar en batch, padding mínimo a múltiplo de stride. Default `False` en train (cuadrado letterbox), `True` en val/predict.
- **`scale_fill` (Ultralytics LetterBox internal):** Equivalente a stretch — distorsiona aspect ratio. Se activa con `LetterBox(..., scale_fill=True)`. NO es default.

---

## Brechas residuales

1. **No hay ablación letterbox-vs-stretch publicada en waste detection** con aspect ratios mixtos. Oportunidad de aporte IEEE.
2. **Roboflow no documenta interpolation method de Stretch** públicamente; pugio descubrió cv2.INTER_AREA por reverse engineering. Si necesito replicar exact en inference, asumir INTER_AREA.
3. **Diferencia padding 114 (Ultralytics) vs 0 (Roboflow Fit-black):** menor pero presente. Mitigación opcional: aplicar `cv2.copyMakeBorder(..., value=(114,114,114))` en pipeline inference Jetson Nano si se observa caída de mAP en eval real.
4. **Cutout ¿incluido en Premium Trial Enhanced Augmentations?** Solo Mosaic explícitamente confirmado. Verificar en UI de generación de Version. Si no está, no aplica decision sec 4.2 sobre Cutout.
5. **Premium Trial $20 vs $60:** discrepancia entre lo que dice docs.roboflow.com ($60) y lo que recibió el usuario ($20). Promoción específica universitaria probable. No bloquea decisión.
