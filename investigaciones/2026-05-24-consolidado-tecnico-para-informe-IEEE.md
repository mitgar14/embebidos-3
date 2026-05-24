# Consolidado técnico del proyecto embebidos-3 — insumo para el informe IEEE

> **Propósito de este documento.** Resumir, en un solo lugar y con detalle por subsistema, **cómo está construido y cómo procede** el proyecto `embebidos-3`, para que quien redacte el informe IEEE tenga una base fiel de la arquitectura, las decisiones y los números citables. Se centra en **lo que está efectivamente desarrollado y en uso** (la mayor parte del sistema); las piezas sueltas o inconclusas se agrupan al final (§13) y se marcan claramente para que **no se afirmen como hechas** en el informe.
>
> **Fecha de consolidación:** 2026-05-24. **Base:** lectura directa del código fuente del repositorio + verificación por SSH del estado real del Nano + las investigaciones previas de la carpeta `investigaciones/`.
>
> **Convención de lectura:** ⚠️ marca datos que el redactor debería **verificar o completar** (típicamente métricas finales del modelo, que viven en HuggingFace y no en el repositorio).

---

## 1. Resumen del proyecto

`embebidos-3` es un **MVP académico** (curso de IA en sistemas embebidos, Universidad Autónoma de Occidente, UAO) que implementa **detección de objetos en tiempo real sobre una banda transportadora** para clasificar residuos en **3 clases: `glass` (vidrio), `paper` (papel) y `plastic` (plástico)**.

| Aspecto | Detalle |
|---|---|
| **Tarea** | Detección multiclase (bounding boxes) en streaming de vídeo |
| **Entrega / sustentación** | 2026-05-26 |
| **Hardware de inferencia** | Jetson Nano Developer Kit 4 GB rev. B01 |
| **Plataforma del Nano** | JetPack 4.6.1 (L4T R32.7.1, Ubuntu 18.04), Python 3.6.9, CUDA 10.2.300, **TensorRT 8.2.1.8**, GPU Maxwell `sm_53` (128 CUDA cores, **sin tensor cores INT8, sin instrucción `dp4a`**) |
| **Cámara** | Logitech C920 OG (USB UVC, FOV 78°), montada en diagonal sobre la banda |
| **Actuación (prevista)** | 3 servomotores SG90 vía driver PCA9685 (I²C) — rampas deflectoras por clase |
| **Entrenamiento** | Nube: instancias spot Vast.ai con GPU RTX 4090 (24 GB) |

La idea operativa: la cámara observa la banda, el Nano detecta y clasifica cada objeto en tiempo real, y (a futuro) tres servos desvían cada residuo hacia su contenedor según la clase.

---

## 2. Arquitectura general del sistema

El sistema se divide en **dos planos**: un **plano de entrenamiento** (en la nube, offline) que produce el modelo, y un **plano de inferencia** (en el Nano + la laptop del operador, en vivo) que lo ejecuta. Entre ambos, **HuggingFace Hub** actúa como almacén central de modelos y datasets.

```
┌──────────────────────── PLANO DE ENTRENAMIENTO (nube, offline) ────────────────────────┐
│                                                                                          │
│  Roboflow (dataset v1-B)  ──┐                                                            │
│  Fotos reales del conveyor ─┤                                                            │
│        │                    │                                                            │
│        ▼                    ▼                                                            │
│  Auto-etiquetado        build_v1c.py        Vast.ai (RTX 4090)                           │
│  (Grounding DINO)  ───►  dataset v1-c  ───► nbconvert headless ──► YOLOv8n .pt           │
│  notebook autolabel                          (bootstrap.sh)         │                    │
│                                                                     ▼                    │
│                                                              export ONNX (opset 11)      │
│                                                              Gates 3 y 4                  │
│                                                                     │                    │
└─────────────────────────────────────────────────────────────────── │ ──────────────────┘
                                                                       ▼
                                                    ┌─────────────────────────────────┐
                                                    │   HuggingFace Hub (modelos)     │
                                                    │   mitgar14/embebidos-3-models*  │
                                                    │   best.onnx + manifest.json     │
                                                    └─────────────────────────────────┘
                                                                       │
┌──────────────────────── PLANO DE INFERENCIA (en vivo) ─────────────  │ ─────────────────┐
│                                                                       ▼                  │
│   Laptop del operador                              Jetson Nano (Headscale 100.64.0.2)    │
│   ┌─────────────────────┐                          ┌──────────────────────────────────┐ │
│   │ Dashboard web        │   JPEG frame (WS bin)   │ embebidos3-server (FastAPI/WS)    │ │
│   │ (cámara getUserMedia)│ ──────────────────────► │  TRTWorker (1 hilo GPU)           │ │
│   │  640×480, q0.7       │ ◄────────────────────── │  letterbox→TRT FP16→NMS V0        │ │
│   │  canvas overlay      │   JSON detecciones      │                                   │ │
│   └─────────────────────┘                          │ Gestión de modelo:                │ │
│   ┌─────────────────────┐   HTTP REST / SSE        │  build / swap / rollback / adopt  │ │
│   │ Pestaña "modelo"     │ ──────────────────────► │  embebidos3-builder@<job> (TRT)   │ │
│   │ (build, logs, hist.) │                          └──────────────────────────────────┘ │
│   └─────────────────────┘                                                                │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Flujo de inferencia en vivo (camino crítico):** la cámara está **en la laptop** (no en el Nano). El navegador captura frames con `getUserMedia` a 640×480, los comprime a JPEG (calidad 0,7) y los envía como mensajes binarios por WebSocket al Nano. El Nano corre la inferencia y devuelve un JSON con las cajas detectadas, que el dashboard dibuja sobre un canvas superpuesto al vídeo.

---

## 3. Decisiones arquitectónicas fundamentales

Estas decisiones son el "por qué" del proyecto y son altamente citables en el informe.

### 3.1 "Track B" exclusivo: YOLOv8n → ONNX → TensorRT FP16

El proyecto evaluó dos caminos y **descartó el "Track A"** (SSD MobileNet v2 + TFLite INT8 sobre CPU) el 2026-05-13, quedándose solo con **Track B (YOLOv8n → TensorRT FP16)**. Razones:

1. **Aprovechamiento del hardware.** La GPU Maxwell de la Tegra X1 **solo rinde bien en FP16**: no tiene tensor cores INT8 ni instrucción `dp4a`. Track A correría en CPU como en una Raspberry Pi 4, desaprovechando la GPU.
2. **Plazo de entrega.** Track A exige QAT obligatorio, calibración con *representative dataset* y postproceso `TFLite_Detection_PostProcess` embebido; Track B es más predecible y el dataset ya estaba listo.

### 3.2 Precisión FP16 (no INT8)

Consecuencia directa del `sm_53` sin tensor cores INT8. El engine se compila con `trtexec --fp16`. INT8 no daría aceleración real en esta GPU.

### 3.3 Resolución de inferencia 416×416

El engine de producción se compila a **`imgsz=416`** (binding de entrada fijo `[1, 3, 416, 416]`). Es el punto de equilibrio velocidad/precisión para este hardware. Subir a 640 está estudiado (ganaría recall en objetos pequeños) pero baja el FPS; se dejó como mejora futura (§13).

### 3.4 Shapes fijos, no dinámicos

**No se usa `dynamic=True`** en el export ni en el engine: TensorRT 8.x es inestable con shapes dinámicos en Maxwell. Si hiciera falta otra resolución, se compila un engine fijo separado.

### 3.5 NMS en CPU (variante "V0")

El postproceso usa **`cv2.dnn.NMSBoxes` en CPU** (OpenCV 4.1.1) — la "ruta segura" para `sm_53`. Se investigaron dos alternativas aceleradas por GPU (V1 `EfficientNMS_TRT`, V2 `BatchedNMSDynamic_TRT`), y se **confirmó empíricamente** que el plugin `EfficientNMS_TRT` sí funciona en el binario de JetPack 4.6.1 (contradiciendo la creencia de que está roto en Maxwell — ver §12). Aun así, V0 quedó como default por compatibilidad garantizada y simplicidad.

### 3.6 Compatibilidad Python 3.6.9 como restricción transversal

Todo el código del Nano debe ser **Python 3.6-safe**. Esto prohíbe sintaxis posterior (walrus `:=`, `match`/`case`, `dict | dict`, `subprocess(capture_output=...)`, etc.) y obliga a sustituir librerías incompatibles. El caso más visible: el **SDK `huggingface_hub` no es compatible con Python 3.6**, así que se escribió un cliente REST propio (`hf_rest.py`, §9.1). También se usa Pydantic 1.9.2 (no 2.x).

---

## 4. Pipeline de datos y datasets

### 4.1 Dataset base v1-B (Roboflow)

- **Origen:** Roboflow, workspace `embebidos3`, proyecto `waste-3class-lwld8`, versión 1 ("v1-B").
- **Preprocesamiento:** *Fit (black)* a 416×416 (relleno negro, `padding=0`).
- **Tamaño:** **17.910** imágenes de entrenamiento / **1.739** de validación / **844** de test (≈ 20.493 en total), 3 clases.
- **Orden de clases:** `[glass=0, paper=1, plastic=2]` (orden canónico del proyecto).

### 4.2 El problema detectado: bajo recall y sesgo hacia `plastic`

En una captura real del 2026-05-15 (escena de laboratorio UAO: 3 piezas pequeñas de plástico sobre fondo de tela cremosa) el modelo v1-B **solo detectó 1 de 3 objetos** (`plástico 51 %`), con 2 falsos negativos. El diagnóstico fue multicausal: umbral de confianza alto, posible *mismatch* de padding (Roboflow Fit-black=0 vs. LetterBox=114 de Ultralytics), resolución 416 corta para objetos pequeños y, sobre todo, **escasa representación del dominio real** (la banda + iluminación + cámara concretas) en el dataset. Se midió además **overfit en la clase `plastic`** (factor ≈ 5,3×). Esto motivó la iteración v1-c.

### 4.3 Auto-etiquetado de datos reales (Grounding DINO)

Para inyectar dominio real sin etiquetar a mano, se montó un **sistema de auto-etiquetado** (detalle en §8):
- Se capturaron **37 fotos reales del conveyor** (2026-05-15) y se subieron a un dataset privado de HF.
- Se auto-etiquetaron con **Autodistill + Grounding DINO** (detección zero-shot guiada por texto) en una instancia Vast.ai efímera.
- **Resultado real (2026-05-20):** 37/37 imágenes etiquetadas (100 % de cobertura), **256 bounding boxes** distribuidas en `paper` 143 (56 %), `plastic` 65 (25 %), `glass` 48 (19 %). Esa composición *paper-dominante* es justo lo que el modelo necesitaba para corregir el sesgo hacia `plastic`.
- Las etiquetas se revisan/corrigen con herramientas propias (mosaico de revisión + editor interactivo de cajas), produciendo `batch1-clean`.

### 4.4 Dataset v1-c (combinado)

`build_v1c.py` construye **localmente** (sin la API de Roboflow) el dataset v1-c = **v1-B + `batch1-clean`**:
- Copia v1-B completo y añade las 37 imágenes reales (≈ 29 a train, ≈ 8 a valid). ⚠️ *Conteos exactos de v1-c conviene verificarlos en el `data.yaml` generado.*
- Aplica un **remapeo de clases obligatorio**: `batch1` venía como `[plastic=0, paper=1, glass=2]` y se remapea `{0→2, 1→1, 2→0}` para alinearlo con el orden canónico de v1-B `[glass, paper, plastic]`.
- v1-c se empaquetó como un único ZIP y se subió a HF (`mitgar14/embebidos3-dataset-v1c`), evitando descargar decenas de miles de archivos sueltos.

---

## 5. Entrenamiento en la nube (Vast.ai)

### 5.1 Infraestructura y modo de ejecución

- **Plataforma:** instancias **spot de Vast.ai con RTX 4090** (24 GB), imagen `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` (Python 3.10, CUDA 12.4).
- **Gestor de paquetes:** `uv` (entorno `/opt/venv/trackb`).
- **Ejecución headless:** el notebook se ejecuta sin interfaz con `jupyter nbconvert --execute --inplace` dentro de una sesión **tmux** (sobrevive a caídas de SSH). Para que tmux no muera al cerrar la sesión se fuerza `KillUserProcesses=no` en `logind.conf`.
- **Persistencia incremental:** `huggingface_hub.CommitScheduler` empuja `runs/`, `manifests/` y `exports/` a HF Hub **cada 10 minutos** (`squash_history=True` mantiene el repo liviano).
- **Auto-destrucción de la instancia** (control de costos): dos mecanismos. (a) Al terminar el notebook, se lanza `vastai destroy instance <ID>` tras un `sleep 30`. (b) Un **cron watchdog** mide la GPU cada 5 minutos y, si la utilización está por debajo del 5 % durante 6 muestras seguidas (≈ 30 min), destruye la instancia.

### 5.2 `bootstrap.sh` (provisión idempotente, 8 etapas)

Instala dependencias apt; instala `uv`; crea el venv `trackb` (Python 3.10); instala el stack de ML con **pins explícitos** (`torch==2.4.1`/`torchvision==0.19.1` para CUDA 12.4, `ultralytics>=8.4.46,<8.5`, `numpy<2.0`, `onnx`, `onnxslim>=0.1.34`, `onnxruntime-gpu`); registra el kernel Jupyter `trackb`; instala el CLI `vastai`; configura el cron watchdog de auto-destrucción; y arranca JupyterLab en tmux. El pin de `torch` es una decisión deliberada: sin él, `uv` resuelve a una versión de torch que exige un driver más nuevo del que tiene Vast.ai y falla.

`onstart.sh` es el `--onstart-cmd` de la instancia: descarga `bootstrap.sh` y el notebook desde HF, ejecuta el bootstrap y lanza el entrenamiento headless en tmux, todo registrado en logs.

### 5.3 Notebooks de entrenamiento

Hay **dos iteraciones**, con la misma estructura (~14 secciones) pero distinto dataset/augmentation:

| | `train_track_b_yolov8.ipynb` (v1-B) | `train_v1c_vastai.ipynb` (v1-c, iteración activa) |
|---|---|---|
| **Fuente del dataset** | Roboflow API (con cascada de 3 estrategias por un bug del SDK en `location=`) | ZIP único descargado de HF (`embebidos3-dataset-v1c`) |
| **Arquitectura** | YOLOv8n (pesos COCO `yolov8n.pt`) | igual |
| **`imgsz`** | 416 | 416 |
| **`epochs`** | 100 (early stopping `patience=20`) | igual |
| **`batch`** | 64 (Vast.ai) / 16 (local) | 32 |
| **`seed`** | 42 | 42 |
| **Augmentation** | por defecto de Ultralytics | **reforzado** (ver abajo) |
| **Destino HF** | `mitgar14/embebidos-3-models` | `mitgar14/embebidos-3-models-v1c` |

**Augmentation reforzado de v1-c** (para combatir el overfit en `plastic`): `degrees=20`, `translate=0,1`, `scale=0,5`, `hsv_h=0,015`, `hsv_s=0,7`, `hsv_v=0,6` (cubre sombras de la banda), `mixup=0,15`, **`copy_paste=0,3`** (clona objetos de las clases minoritarias sobre fondos), `mosaic=1,0`, `close_mosaic=15` (desactiva mosaic en los últimos 15 epochs), `fliplr=0,5`, `flipud=0,0` (sin volteo vertical, no es físico en una banda horizontal). **Confirmado desde `args.yaml` (la corrida real):** `optimizer=auto`, `lr0=0,01`, `lrf=0,01`, `momentum=0,937`, `weight_decay=0,0005`, `warmup_epochs=3`, `freeze=null` (backbone **no** congelado), `amp=true`, `cos_lr=false`, y los defaults de Ultralytics `auto_augment=randaugment` y `erasing=0,4`. Pesos de pérdida: `box=7,5`, `cls=0,5`, `dfl=1,5`.

### 5.4 Robustez del entrenamiento (citable como buena práctica)

- **Heartbeat estilo TRAINCHECK:** un hilo daemon escribe `runs/heartbeat.jsonl` cada 30 s con epoch, loss, grad_norm, lr, memoria GPU y ETA.
- **Manejo de señales:** `SIGTERM`/`SIGINT` se capturan con un handler *flag-based* (sin I/O en el handler); la bandera se inspecciona en el callback `on_fit_epoch_end` y dispara un **export de emergencia** del ONNX antes de cortar.
- **Gates de validación del export:**
  - **Gate 3 (ONNX):** verifica `opset==11`, `ir_version<=8` y ausencia de operadores no soportados por TRT 8.2 (`GridSample`, `MultiHeadAttention`, `RoiAlign`, `NonZero`, etc.).
  - **Gate 4 (Polygraphy, opcional):** si hay Docker, inspecciona el ONNX con el contenedor `nvcr.io/nvidia/tensorrt`. Si no, se marca *skipped* sin romper el pipeline.
- **Export ONNX:** `opset=11`, `simplify=True`, `dynamic=False`, **`nms=False`** (NMS desacoplado, se hace en el Nano), `half=False`, en CPU.
- **Manifest consolidado** (`manifests/manifest.json`): SHA256 de `.pt` y `.onnx`, métricas de evaluación, gates, descripción del target Nano y stack de versiones. Este manifest es el que luego consume el builder del Nano.

### 5.5 Estado del entrenamiento y métricas reales del modelo v1-c

El **engine activo en el Nano corresponde a v1-c**: HuggingFace `mitgar14/embebidos-3-models-v1c`, revisión `2e80e24` (commit 2026-05-23), engine FP16 de **13,5 MB**, compilado en el Nano el **2026-05-23** (verificado por SSH). El ciclo entrenamiento → export → compilación está **cerrado de punta a punta**.

**Métricas reales (extraídas de `runs/detect/train/results.csv` en HF).** Son métricas de Ultralytics sobre el **set de validación**, **globales** (las tres clases agregadas). El entrenamiento estabilizó (meseta) desde la época ~80:

| Métrica (validación, global) | Mejor época (81) | Última época (84) |
|---|---|---|
| Precision | 0,923 (92,3 %) | 0,926 (92,6 %) |
| Recall | 0,832 (83,2 %) | 0,831 (83,1 %) |
| **mAP@50** | **0,915 (91,5 %)** | 0,915 (91,5 %) |
| **mAP@50-95** | **0,713 (71,3 %)** | 0,713 (71,3 %) |

**Dato importante para el informe — el entrenamiento se cortó por timeout.** La corrida llegó a la **época 84 de 100** y se detuvo exactamente al alcanzar el **timeout de 2 h (7.200 s) del `nbconvert`** (confirmado en `runs/heartbeat.jsonl`: última muestra en época 84, `elapsed_s ≈ 7.200`). Como consecuencia, el notebook **no ejecutó las secciones finales** de evaluación en *test* ni el **export ONNX automático**; el `manifest.json` de v1-c se regeneró **post-hoc** desde la laptop (sus propias notas lo dicen: *"el training de V1c en Vast.ai no llegó a la sección 12"*) y el ONNX se exportó a mano antes de compilar el engine. Como las métricas ya estaban en meseta en la época 80, el corte en la 84 no afectó de forma apreciable el resultado.

> ⚠️ **Lo único que falta (desglose por clase):** `results.csv` solo trae métricas **globales** de validación; **no hay desglose por clase** (`glass`/`paper`/`plastic`) ni evaluación sobre el split de *test*, porque el `model.val()` final no se ejecutó. Si el informe necesita recall/AP **por clase**, hay que correr `model.val(split="test")` sobre `best.pt` (disponible en HF: `runs/detect/train/weights/best.pt`) con el dataset v1-c. Los hiperparámetros reales fueron confirmados desde `runs/detect/train/args.yaml` (ver §5.3).

---

## 6. Compilación del engine TensorRT en el Nano (el "builder")

El proyecto **operacionaliza la compilación del engine desde el dashboard**, en lugar de hacerla a mano por SSH. Esto es uno de los componentes más elaborados del sistema. El usuario pulsa "compilar" en la web → el server lanza un servicio systemd → un script bash de **12 fases** descarga el ONNX desde HF, compila el engine con `trtexec` y lo intercambia atómicamente, todo monitoreado en vivo.

### 6.1 Disparo y aislamiento (systemd + sudoers)

- `POST /model/build` genera un `job_id` (`YYYYMMDD-HHmm-<6hex>`) y ejecuta `sudo /usr/local/bin/embebidos3-builder-launch <job_id>`.
- El **wrapper `embebidos3-builder-launch`** valida el `job_id` con una regex estricta (`^[A-Za-z0-9_-]{10,40}$`, defensa contra inyección) y arranca la unidad **`embebidos3-builder@<job_id>.service`** con `systemctl start --no-block` (no bloquea durante los 15-40 min del build).
- La unidad es un **`Type=oneshot` templado**, corre como usuario `jetson`, con `TimeoutStartSec=2700` (45 min) y logs a archivo.
- **Sudoers granular:** 14 reglas `NOPASSWD` con **rutas absolutas exactas** (p. ej. `fallocate` está en `/usr/bin/`, no en `/sbin/` — gotcha real verificado con `which`).

### 6.2 Las 12 fases de `nano_build_engine.sh`

Con `set -euo pipefail`, un **lock exclusivo** (`flock` sobre `/run/embebidos3/builder.lock`) y **traps** de EXIT y SIGTERM que garantizan limpieza pase lo que pase:

| # | Fase (progreso) | Qué hace |
|---|---|---|
| 0 | `acquired_lock` (5 %) | Toma el lock exclusivo, registra el PID |
| 1 | `downloaded_manifest` (8 %) | Descarga `manifest.json` desde HF (autenticado con `HF_TOKEN`) |
| 2 | parseo del manifest | Extrae `hf_revision`, el SHA256 esperado del ONNX y la fecha del commit |
| 3 | `downloaded_onnx` (12 %) | Descarga `exports/best.onnx` de la revisión indicada |
| 4 | `verified_sha` (15 %) | Verifica el SHA256 del ONNX; si no coincide, aborta (exit 2) |
| 5 | `stopped_server` (18 %) | Libera el engine del server vía `POST /model/_internal/release-engine` (el server **sigue vivo sin engine**); fallback a `systemctl stop` |
| 6 | `prep_nano` (22 %) | Prepara memoria: para `lightdm` (libera la GPU de X11), deshabilita zram, crea/activa **swap de 8 GB** en `/mnt/swap.img`, `swappiness=100`, `drop_caches=3` |
| 7 | `trtexec_built` (25→75 %) | Lanza `tegrastats` + `trtexec --onnx=... --saveEngine=... --fp16 --workspace=512 --buildOnly --verbose` con `timeout 40m`. Maneja exit 124 (timeout) y 137 (OOM). Sanity check: engine < 1 MB ⇒ falla |
| 8 | `validated` (80→85 %) | Corre `validate_engine.py` sobre el engine recién compilado |
| 9 | `backed_up_previous` (88→92 %) | Archiva el engine anterior en `engines-archive/<ts>__<sha8>/` y sube **solo el manifest** (~1 KB) a HF |
| 10 | swap atómico (95 %) | Intercambio "A++" con centinela `.ready` y `fsync` de archivos y directorios (ver 6.4) |
| 11 | `restoring_nano` (97 %) | Reinicia `lightdm`, restaura `swappiness=60` |
| 12 | `starting_server` → `done` (99→100 %) | Recarga el engine en el server vía `POST /model/_internal/reload-engine`; fallback a `systemctl start` |

El **parser de progreso** (`parse_trtexec_progress.py`) lee la salida de `trtexec` en vivo y, por *regex* (`Finished parsing network model`, `Engine built in`, `&&&& PASSED`, líneas `Timing Runner`…), va reportando el porcentaje a `builder_state.py`, que escribe atómicamente `/run/embebidos3/job.json` con la fase, el progreso, un **heartbeat** y el PID.

### 6.3 Validación de correctitud (`validate_engine.py`)

Deserializa el engine en staging, asigna buffers con `pycuda`, y corre inferencia real sobre hasta **3 imágenes de prueba**. **Criterio de aprobación:** al menos **2 de 3** imágenes deben producir ≥ 1 detección con confianza > 0,3. La salida JSON se incrusta en el `.meta.json` del engine. *(Existe además `nano_correctness.py`, una herramienta manual de inspección vía SSH+SCP que dibuja las cajas; no se invoca desde el builder automatizado.)*

### 6.4 Swap atómico con centinela (robustez ante cortes)

El intercambio del engine es a prueba de fallos: usa un archivo centinela `best_fp16.engine.ready` (con `committed_at` y `engine_sha256`) y `fsync` explícito de archivos **y** del directorio padre (patrón inspirado en Cog y OSTree). Si el proceso muere por SIGKILL/OOM entre los dos `mv`, al reiniciar el server **reconcilia el estado** (`reconcile_engine_state`, §7.5) y restaura el engine previo. Esto evita dejar el Nano "sin modelo" por un corte a mitad de swap.

### 6.5 Metadata y trazabilidad de cada engine

`write_engine_meta.py` escribe `best_fp16.engine.meta.json` junto al engine activo, con: `engine_sha256`, `onnx_sha256`, `hf_revision`, `hf_commit_date`, `trtexec_args`, `build_completed_at`, `build_duration_s` y el bloque `validation` completo. **Sin `.meta.json`, el sistema reporta `no_model` aunque el binario exista** — esto fuerza la trazabilidad total: todo engine en uso sabe de qué ONNX/commit salió.

### 6.6 Instaladores

- `nano_install_systemd.sh` (idempotente): crea `/etc/embebidos3/secrets.env`, instala el wrapper, las 14 reglas sudoers (validadas con `visudo -cf`), las unidades systemd, la config de `tmpfiles` (limpia `logs/jobs/` con TTL de 3 días), recarga el daemon y arranca el server.
- `nano_install_inference.sh`: instala el stack Python 3.6 del Nano con versiones fijadas — **pycuda 2019.1.2** (versión recomendada para JP 4.6.1), `fastapi==0.65.3`, `uvicorn==0.13.4`, `websockets==9.1`, `starlette==0.14.2`, `pydantic<2`, sobre OpenCV 4.1.1 del sistema.

---

## 7. Servidor de inferencia (`nano_server.py`)

Es el **corazón del plano de inferencia**: una aplicación **FastAPI + Uvicorn** que recibe frames por WebSocket, ejecuta TensorRT y devuelve detecciones, además de gestionar todo el ciclo de vida del modelo.

### 7.1 Modelo de concurrencia

- **Un único hilo GPU** (`TRTWorker`, `threading.Thread` daemon). El event loop asyncio nunca toca el contexto CUDA; se comunica con el worker mediante una `asyncio.Future` resuelta con `loop.call_soon_threadsafe(...)`.
- **Backpressure:** cola de entrada `queue.Queue(maxsize=2)`. Si está llena, el server responde `{"ok": false, "error": "queue_full"}` en vez de acumular lag. Diseño correcto para tiempo real: se sacrifican frames para mantener baja la latencia.
- **Parche Python 3.6:** `asyncio.create_task = asyncio.ensure_future` (no existe `create_task` en 3.6.9).

### 7.2 Pipeline de inferencia, paso a paso

1. **Entrada:** frame JPEG binario por `/ws`. Se registra `t_recv_ms` y un `seq` incremental.
2. **Decodificación:** `cv2.imdecode` (BGR).
3. **Letterbox a 416×416:** escala manteniendo proporción (`r = 416/max(h,w)`), centra y **rellena con gris 114** (estándar YOLO).
4. **Tensor:** BGR→RGB, HWC→CHW, `float32`, normalizado a `[0,1]` (÷255).
5. **TensorRT:** copia *host→device* (`pycuda`, buffers *page-locked*), `execute_async_v2`, copia *device→host*, `stream.synchronize()`. El contexto CUDA se crea una sola vez por vida del hilo (crear/destruir contextos seguido crashea en `sm_53`).
6. **Postproceso + NMS V0:** decodifica la salida YOLOv8 `(N_anchors, 4+3)`, `argmax` de clase, filtra por `conf` (default **0,25**), `cv2.dnn.NMSBoxes` (IoU default **0,45**), y **deshace el letterbox** para devolver coordenadas en el espacio del frame original.
7. **Respuesta JSON:**
   ```json
   {"ok": true,
    "bboxes": [{"x1":…, "y1":…, "x2":…, "y2":…, "conf":…, "cls":0, "cls_name":"glass"}],
    "t_infer_ms":…, "t_recv_ms":…, "client_ts_ms":…, "seq":…}
   ```
   Las clases son `["glass","paper","plastic"]` (índices 0/1/2). El umbral `conf` se puede ajustar **en vivo** desde el cliente (`{"type":"conf","value":X}`).

**Rendimiento (verificado por SSH):** ≈ **40-43 FPS / 23-25 ms** end-to-end a `imgsz=416` FP16. Esto **supera la predicción de la literatura** (*Nature* 2024 estimaba 30 FPS para Nano) y deja un margen de 4× sobre el umbral del MVP (≥ 10 FPS).

### 7.3 Estados del modelo (derivados del filesystem)

| Estado | Condición |
|---|---|
| `building` | Hay un job activo con PID vivo y cmdline válida |
| `ready` | Existe el engine + `.meta.json` con `from_fallback=false` |
| `degraded` | Existe el engine pero el meta tiene `from_fallback=true` (se llegó por rollback) |
| `no_model` | No hay meta, o no hay binario |

### 7.4 Endpoints (lista real)

| Método | Ruta | Función |
|---|---|---|
| GET | `/health` | Temperatura GPU, RAM, FPS (ventana 1 s), latencia, conf actual, `IMGSZ`, `CLASSES` |
| WS | `/ws` | Inferencia (JPEG→JSON), ajuste de `conf`, ping/pong |
| GET | `/model/state` | Estado del modelo + metadata del engine activo/previo + job activo |
| POST | `/model/build` | Lanza build (202 + `job_id`; 409 si ya hay uno) |
| POST | `/model/check-updates` | Compara `hf_revision` **y** `onnx_sha256` local vs. HEAD de HF (sin descargar el binario) |
| POST | `/model/rollback` | Intercambio inverso con el engine previo |
| POST | `/model/rollback-to/{archive_id}` | Restaura un engine arbitrario del histórico |
| POST | `/model/adopt` | Registra meta retroactivo para un engine huérfano (binario sin meta) |
| GET | `/jobs/active`, `/jobs/{id}`, `/jobs` | Job en curso / estado de un job / histórico de engines |
| GET | `/jobs/{id}/logs` | **Stream SSE** de logs del build en vivo |
| DELETE | `/jobs/{id}` | Cancela el build (SIGTERM vía systemd) |
| POST | `/model/_internal/{release,reload}-engine` | **Solo localhost** — usados por el builder para liberar/recargar el engine sin reiniciar el server |

CORS abierto (`allow_origins=["*"]`) para que el dashboard local pueda consumir los endpoints. *(No hay autenticación: aceptable para un MVP en red privada VPN.)*

### 7.5 Recuperación al arranque

Antes de que el worker arranque, el server corre `reconcile_engine_state()` (repara un swap interrumpido, §6.4, con `fsync` de directorio) y `recover_job_state()` (si el job.json apunta a un PID muerto, lo marca `abandoned`; si el heartbeat es viejo, `stalled`). Garantiza que el sistema siempre arranca en un estado coherente.

---

## 8. Sistema de auto-etiquetado

Resuelve cómo ampliar el dataset con datos reales **sin etiquetar a mano** y de forma autónoma (cualquier integrante puede disparar un job sin coordinar con otro).

### 8.1 Arquitectura: primario + plan B

- **Primario — Vast.ai on-demand (ejecutado):** notebook `autolabel_vastai.ipynb` corrido headless en una GPU efímera; se auto-destruye al terminar. Costo real de un lote: ≈ **0,02 USD** (RTX 3060, ~20 min).
- **Plan B — WSL2 + RTX 3060 (implementado, no activado):** un server FastAPI (`scripts/labeling/server/`) como servicio systemd dentro de WSL2, expuesto al *tailnet* con `tailscale serve`. Quedó listo pero **nunca se activó** porque el primario resolvió todo.

### 8.2 Flujo end-to-end

```
Fotos (OneDrive) ─upload_batch_hf.py─► HF raw-batches ─► [Vast.ai: Grounding DINO] ─► HF labels
                                                                                          │
   dataset YOLO limpio ◄─ revisión humana (overlay_review + label_review) ◄──────────────┘
            │
            └─ build_v1c.py ─► v1-c ─► re-entrenamiento
```

### 8.3 Componentes

- **Notebook `autolabel_vastai.ipynb` (8 celdas):** instala deps (con pins críticos: `transformers<5`, `opencv-python-headless` antes de `autodistill`), define una **ontología por texto** (`"plastic bottle or plastic container" → "plastic"`, etc. — descripciones largas dan mejor precisión), descarga el lote desde HF, etiqueta con `GroundingDINO(box_threshold=0,25, text_threshold=0,175)`, aplica un **gate de calidad del 85 %** (ratio etiquetas/imágenes) y sube el resultado a HF, luego auto-destruye la instancia.
- **Revisión humana:**
  - `overlay_review.py`: genera imágenes con cajas dibujadas + un **mosaico** + un resumen de conteos por clase (revisión visual rápida, no modifica nada).
  - `label_review/app.py`: una **SPA FastAPI** para corregir cajas a mano (dibujar/mover/redimensionar/borrar con mouse y teclado, auto-guardado con debounce). Pre-filtra cajas demasiado grandes (>35 % del área) y aplica NMS. Exporta un dataset YOLO limpio (`batch1-clean`) con su `data.yaml`.
- **Server del Plan B (`server/main.py` + `worker.py`):** API con `/autolabel/job`, `/jobs`, `/jobs/{id}/state|logs|artifact`, cancelación, recuperación de jobs huérfanos al arranque, y serialización estricta (un job a la vez, porque la RTX 3060 6 GB no soporta dos Grounding DINO simultáneos).

### 8.4 Estado

Ejecutado y útil: el notebook (lote `batch1`, 37 imágenes), `upload_batch_hf.py`, `download_v1b.py`, las dos herramientas de revisión. Implementado pero no usado: el server WSL2 del Plan B y su instalador. *(Deuda técnica menor: una corrección de un bug del gate quedó en el notebook pero no se retroportó al worker del Plan B — irrelevante porque el Plan B no se usó.)*

---

## 9. Infraestructura de soporte

### 9.1 Cliente HuggingFace propio (`hf_rest.py`)

Existe porque el **SDK `huggingface_hub` no corre en Python 3.6.9**. Es un cliente REST mínimo que solo depende de `requests`. Funciones:
- `download(...)`: descarga con escritura atómica (`.tmp` + rename).
- `get_head_revision()`: devuelve el `sha` del HEAD (para comparar versiones sin descargar nada).
- `get_file_lfs_sha256(...)`: obtiene el **SHA256 (oid LFS) de un archivo sin descargarlo** (~1 KB de JSON) — clave para `/model/check-updates`.
- `upload_file_inline(...)` / `delete_file_inline(...)`: commits NDJSON para archivos pequeños (< 10 MB). **No implementa el flujo LFS completo**, por eso al archivar engines solo se sube el manifest (~1 KB) y el binario (~13,5 MB) queda en el Nano.
Autentica con `HF_TOKEN` (`Bearer`). Tiene CLI (`download`/`upload`/`delete`/`head-revision`) con validación manual de subcomando (porque `argparse` de 3.6 no soporta `required=True` en subparsers).

### 9.2 Conectividad de red: de Tailscale a Headscale self-hosted

**Problema (2026-05-22):** en la red WiFi de la UAO, el **FortiGate bloquea Tailscale** (categoriza el handshake WireGuard y/o filtra por SNI los dominios `*.tailscale.com`). La VPN funcionaba en casa pero no en el campus — crítico para la sustentación.

**Solución (2026-05-23):** se desplegó un **control plane Headscale propio**:

| Componente | Detalle |
|---|---|
| VPS | Contabo, IP `80.241.217.130`, Ubuntu 24.04 |
| Dominio | `80-241-217-130.nip.io` (nip.io evita registrar un dominio) |
| Control plane | **Headscale v0.28.0** en Docker |
| TLS / proxy | Caddy 2 (certificado Let's Encrypt automático), reverse proxy a Headscale, **todo por `443`** |
| Relay | **DERP embebido** de Headscale (región 999) en el mismo `443/tcp` + `3478/udp` (STUN) |

**Por qué atraviesa el FortiGate:** el cliente solo ve una conexión TLS saliente a un dominio `nip.io` no categorizado como VPN. Si el camino directo UDP (`41641`) se bloquea, el tráfico **cae automáticamente al DERP en `:443`**, que sí pasa.

**Nodos del *tailnet* `embebidos3`:** VPS `100.64.0.1`, **Nano `100.64.0.2`** (migrado con `--force-reauth`; el server queda en `http://100.64.0.2:8000`), laptop `100.64.0.3`. El acceso SSH al Nano se reforzó con **llaves ed25519** (ya no depende de Tailscale SSH). La migración usó un *dead-man-switch* (`systemd-run --on-active=12min`) que restauraba Tailscale si la migración fallaba.

> ✅ **Validado en el campus UAO (2026-05-23).** El bypass quedó probado **end-to-end desde la red WiFi-UAO**, sin hotspot 4G ni pasos manuales: SSH al Nano (`100.64.0.2`) OK, FastAPI `/health` OK, y el control plane `https://80-241-217-130.nip.io` alcanzable (status *pass*) — **confirmando que el FortiGate no bloquea `nip.io`**. El data plane usó el **DERP embebido sobre `nip.io:443`** (~330 ms, el camino garantizado que atraviesa el firewall) y **además** logró **conexión directa por NAT traversal** (`186.169.20.111:41641`, ~33 ms). La fricción del hotspot 4G quedó eliminada. Es decir, la solución de conectividad está **validada en el entorno real de la sustentación**.

---

## 10. Hardware de actuación (servos) — estado de validación

La **actuación física no está integrada al sistema de inferencia todavía**; lo que existe es la **validación del hardware** de servos.

- **Diseño previsto:** 3 servos **SG90** vía driver **PCA9685** (I²C, dirección `0x40`), comandados **desde el Nano por Python `smbus`** (no por Arduino) en producción.
- **Lo desarrollado:** 5 sketches Arduino de **bench-test** (en un ESP32, SDA=GPIO21/SCL=GPIO22) para validar el hardware antes de integrarlo:
  - `i2c_scanner.ino`: confirma que el PCA9685 responde en `0x40`.
  - `servo_sweep.ino`: barridos de prueba (wiggle estrecho → medio → amplio) canal por canal.
  - `servo_ch0_loop.ino`, `servos_home_zero.ino`, `servos_loop_seq.ino`: oscilación, posición *home* (0°) y secuencia.
  - Calibración PWM común: 50 Hz, pulso `150–600` *counts* (`map(deg,0,180,150,600)`), oscilador a 27 MHz. Se mueve **un servo a la vez** por la limitación de corriente del USB del ESP32 (en producción habría una fuente dedicada).

> **Importante para el informe:** estos sketches son **herramientas de validación de hardware**, no el código de control de producción. La actuación coordinada con la detección (qué servo activar y cuándo) **no está implementada** — ver §13.

---

## 11. Mapa del repositorio (dónde está cada cosa)

```
embebidos-3/
├── README.md                         Visión general (fechado 2026-05-14; subestima el avance posterior)
├── CLAUDE.md                         Notas de contexto (acceso SSH al Nano)
├── pyproject.toml / uv.lock          Proyecto uv del host (Python ≥3.10)
├── .claude/skills/embebidos3-nano/   Skill: endpoints, paths, sudoers, restricciones Py3.6 del Nano
│
├── investigaciones/                  ★ Input directo para el informe IEEE
│   ├── 2026-05-15-mejoras-modelo-deteccion-plasticos.md   Recall/SAHI/fine-tune (diagnóstico)
│   ├── 2026-05-20-infra-auto-label-remoto.md              Arquitectura de auto-etiquetado
│   ├── 2026-05-22-tailscale-bloqueo-uao.md                Diagnóstico del bloqueo de red
│   ├── 2026-05-23-headscale-deploy.md                     Deploy del control plane propio
│   ├── 2026-05-23-deteccion-proximidad-orden-caida.md     (investigación, sin código aún)
│   ├── 2026-05-23-plan-mvp-proximidad.md                  (plan, sin código aún)
│   └── 2026-05-24-consolidado-tecnico-para-informe-IEEE.md ESTE documento
│
├── notebooks/
│   ├── train_track_b_yolov8.ipynb    Entrenamiento v1-B
│   ├── train_v1c_vastai.ipynb        Entrenamiento v1-c (iteración activa)
│   └── autolabel_vastai.ipynb        Auto-etiquetado con Grounding DINO
│
├── scripts/
│   ├── server/      nano_server.py (FastAPI/WS), constants, pid_utils, recover_job_state
│   ├── builder/     nano_build_engine.sh (12 fases), validate_engine, nano_correctness,
│   │                parse_trtexec_progress, write_engine_meta/archive_manifest, builder_state
│   ├── hub/         hf_rest.py (cliente HF para Python 3.6)
│   ├── install/     nano_install_systemd.sh, nano_install_inference.sh
│   ├── training/    bootstrap.sh, onstart.sh, build_v1c.py
│   ├── labeling/    server/, label_review/, overlay_review.py, download_v1b.py, upload_batch_hf.py
│   └── dashboard/   index.html, app.js, modelo.js, ui.js, style.css (UI estática)
│
├── firmware/        5 sketches .ino de bench-test de servos (PCA9685 + SG90)
├── systemd/         embebidos3-server.service, embebidos3-builder@.service, tmpfiles
└── prototipos/      Imágenes del montaje físico
```

---

## 12. Aportes y resultados citables para el informe IEEE

1. **YOLOv8n FP16 416×416 a ~40-43 FPS / 23-25 ms en Jetson Nano B01 (JP 4.6.1)**, superando la predicción de *Nature* 2024 (Tabla 4: 30 FPS). Resultado empírico reproducible y con margen de 4× sobre el umbral del MVP.
2. **Validación empírica de que `EfficientNMS_TRT` funciona en el binario de JetPack 4.6.1** (`sm_53` Maxwell), contradiciendo la literatura dual-track que lo asume roto. Se documentan las tres rutas de NMS (V0 CPU, V1 `EfficientNMS_TRT`, V2 `BatchedNMSDynamic_TRT`).
3. **Pipeline reproducible de entrenamiento headless sobre cloud spot (Vast.ai)** con `CommitScheduler`, signal handlers *flag-based*, heartbeat estilo TRAINCHECK y auto-destrucción por watchdog. Patrón replicable para proyectos académicos con presupuesto limitado (lote de auto-etiquetado a ≈ 0,02 USD).
4. **Sistema de gestión del ciclo de vida del modelo en el edge** con swap atómico a prueba de cortes (centinela `.ready` + `fsync`), rollback, adopción de engines huérfanos y verificación de actualizaciones contra HF Hub sin descargar binarios — operado íntegramente desde un dashboard web.
5. **Auto-etiquetado zero-shot con Grounding DINO** para corregir *domain shift* y *class imbalance* con datos reales del montaje (100 % de cobertura en el lote real, composición *paper-dominante* para des-sesgar el overfit en `plastic`).
6. **Solución de conectividad sorteando DPI universitario:** control plane Headscale self-hosted con DERP embebido en `:443`, para garantizar acceso remoto a través del FortiGate de la UAO.
7. **(Propuesto como ablación)** comparativas documentadas: *imgsz* 416 vs. 640 en waste detection 3-class, letterbox vs. stretch, y P2-head para objetos pequeños.

**Métricas del modelo v1-c (validación, global):** mAP@50 = **0,915**, mAP@50-95 = **0,713**, Precision = **0,926**, Recall = **0,831** (ver §5.5). ⚠️ Falta solo el **desglose por clase** y la evaluación en *test* (el `model.val()` final no corrió por el corte de timeout); correr `model.val(split="test")` sobre `best.pt` lo produce. El **delta v1-B → v1-c** (que cuantificaría el efecto del augmentation reforzado + datos reales) requeriría evaluar ambos modelos sobre el mismo split.

---

## 13. Alcance y trabajo futuro (lo que NO está hecho)

Para que el informe no afirme de más, estas piezas están **investigadas o planificadas pero sin implementación funcional** a la fecha:

- **Detección de proximidad / orden de caída.** Hay una investigación completa y un plan de MVP (homografía/IPM + tracker NumPy + scheduler de servos) fechados 2026-05-23, pero **no hay código**: ni `proximity.py`, ni el tracker, ni la calibración de homografía. El dashboard tampoco dibuja proximidad, *next-to-fall* ni cola. Es la principal pieza pendiente.
- **Actuación de servos integrada.** Solo existen los bench-tests Arduino (§10). No hay control desde el Nano por `smbus` enganchado a las detecciones, ni lógica de "qué servo activar y cuándo".
- **Mejoras de recall investigadas pero no aplicadas:** SAHI (inferencia en tiles), re-entrenamiento a `imgsz=640`, cabeza P2 para objetos pequeños. Están cuantificadas en `2026-05-15-mejoras-modelo-deteccion-plasticos.md` como mejoras/ablaciones futuras.
- **Plan B de auto-etiquetado (server WSL2):** implementado pero nunca activado.
- **Evaluación formal del modelo:** el entrenamiento de v1-c se cortó por el timeout de 2 h en la época 84/100, antes de la evaluación en *test* y del desglose **por clase**. Solo hay métricas **globales de validación** (§5.5). Para el informe, correr `model.val(split="test")` sobre `best.pt` daría el desglose por clase. *(La conectividad de red en el campus UAO, en cambio, ya está **validada** — §9.2.)*

---

## 14. Referencias bibliográficas principales (ya identificadas)

Citables verificadas durante las investigaciones del proyecto (lista completa en los documentos de `investigaciones/`):

- *Real-time waste detection on Jetson Nano*, **Nature Scientific Reports** (oct. 2024), DOI 10.1038/s41598-024-74798-3.
- Bochkovskiy et al. (2020), *YOLOv4: Optimal Speed and Accuracy of Object Detection*, arXiv:2004.10934.
- Crasto, K. (2024), *Class Imbalance in Object Detection*, arXiv:2403.07113.
- Chakraborty et al. (2025), *Half-core utilization rule on Jetson edge devices*, arXiv:2508.08430.
- Yan, P. et al. (2025), *TRAINCHECK: practical training-time invariant checking for ML pipelines*, arXiv:2506.14813.
- Akyon, Altinuc, Temizel (2022), *Slicing Aided Hyper Inference (SAHI)*, IEEE ICIP, arXiv:2202.06934.
- Khalili, Smyth (2024), *SOD-YOLOv8: Small Object Detection in Traffic Scenes*, IEEE ICIP, arXiv:2408.04786.
- Tariq, Javed (2025), comparativa YOLOv8/v9/v10/v11 en objetos pequeños (DOTAv1.5), arXiv:2504.09900.
- Casao et al. (2024), *SpectralWaste* (waste sorting sobre banda, cámara fija), IROS, arXiv:2403.18033.
- *(Para proximidad, si se llegara a implementar)* Kujala et al. / ZenRobotics (2015), arXiv:1511.07608; Zocco et al. (2024), IEEE TIM, arXiv:2405.06821.

---

*Fin del consolidado. Cualquier número marcado con ⚠️ debe verificarse en el repositorio HuggingFace (`mitgar14/embebidos-3-models-v1c`, carpeta `manifests/`) o en el estado en vivo del Nano antes de publicarlo en el informe.*
