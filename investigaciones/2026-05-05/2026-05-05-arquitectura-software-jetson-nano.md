# Investigación: Arquitectura software Jetson Nano para clasificador de basura MVP

> **Proyecto:** `embebidos-3` — clasificador de residuos con visión por computadora.
> **Hardware:** Jetson Nano Developer Kit 4 GB rev. B01 (NVPN 776-0011-002 R5), JetPack 4.6.1 (L4T 32.7.1, Ubuntu 18.04, Python 3.6.9, CUDA 10.2, cuDNN 8.2.1, TensorRT 8.2.1, OpenCV 4.1.1).
> **Cámara:** USB UVC, objetivo realista 720p/30fps. Montada **en diagonal por encima** de la banda transportadora y la compuerta superior, apuntando hacia los residuos que estarán **acostados** sobre la banda. El ángulo diagonal es decisión de diseño para ampliar el campo de visión y capturar más superficie del objeto.
> **Tarea de visión:** **detección de objetos obligatoria** (no clasificación pura). El curso es de IA en embebidos y la prioridad de evaluación es demostrar funcionalidades de IA, así que la detección por modelo es el componente central. El sensor IR queda exclusivamente como fallback de respaldo si el modelo no resulta robusto en pruebas empíricas.
> **Actuación:** 3-4 servomotores via PCA9685 (I2C). Sin Arduino. Sin Raspberry Pi.
> **Modelo:** TF Lite cuantización INT8 (preferencia), fallback TensorRT FP16.
> **Entrega:** 2026-05-26 (3 semanas).

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-05 | alto | Arquitectura end-to-end: stack, concurrencia, modelo de visión, gotchas, frameworks |

---

## Síntesis y recomendaciones

### 1. Patrón de concurrencia recomendado

**Decisión: dos hilos Python con `threading` + `queue.Queue(maxsize=2)` con política drop-oldest, más un tercer hilo daemon para el control I2C de servos.**

Justificación combinada:

- **`Interpreter.invoke()` libera el GIL** durante la inferencia nativa. Confirmado en doc oficial Google AI Edge: *"this function releases the GIL so heavy computation can be done in the background while the Python interpreter continues. No other function on this object should be called while the invoke() call has not finished."* (research-web). Implicación: `threading` es suficiente para paralelismo real entre captura/inferencia y control. **No necesitamos `multiprocessing`**, que añade ~8× overhead de memoria (arXiv 2601.10582) y complica el ciclo de vida del intérprete.
- **Regla del medio núcleo de Chakraborty et al. (2025, arXiv 2508.08430):** en Jetson Nano (4 cores Cortex-A57), mantener ≤ 2 procesos/hilos *activos en GPU* mantiene EC duration estable (1-2 ms). Pasar a 4 procesos infla el EC ~30×, a 8 procesos ~70×. Implicación: solo un hilo debe correr inferencia; el resto debe estar mayormente bloqueado en I/O.
- **Patrón producer-consumer con cola pequeña** es el dominante en repos reales (`spehj/yolov7-counter-jetson-nano`, `msubzero2000/project-ellee-public`, research-code). Cola de tamaño 1-3 es suficiente para backpressure natural.
- **Captura + inferencia en el mismo hilo** (no hilos separados) es la práctica observada en repos serios. Razón: con GIL liberado en `invoke()` y solo una cámara, separar `cv2.read()` de `interpreter.invoke()` no compra nada — ambos liberan el GIL en sus llamadas nativas y lo único que harías es duplicar el costo de IPC entre hilos.

**Importante:** un único objeto `tflite.Interpreter` **no es thread-safe** para llamadas concurrentes. Si se necesita paralelismo de inferencia (no es el caso del MVP), una instancia por hilo. Para el MVP: una sola instancia.

### 2. Diagrama de hilos y comunicación

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PROCESO ÚNICO (Python 3.6.9, jetson_clocks, MAXN)                       │
│                                                                          │
│  ┌────────────────────────────┐         ┌─────────────────────────────┐  │
│  │ HILO 1: CaptureInferThread │  q1     │ HILO 2: ServoControlThread  │  │
│  │ ─────────────────────────  │ ──────► │ ────────────────────────    │  │
│  │ cv2.VideoCapture(0,        │ Queue   │ Lee cls de q1 (block=True)  │  │
│  │   cv2.CAP_V4L2)            │ size=2  │ Decide ángulos por FSM      │  │
│  │ MJPG 720p 30fps            │ drop-   │ Envía comandos a HILO 3 q2  │  │
│  │ preprocess (resize 320×320)│ oldest  │                             │  │
│  │ tflite.Interpreter.invoke()│         │                             │  │
│  │ argmax → cls (0/1/2)       │         │                             │  │
│  │ q1.put(cls)                │         │                             │  │
│  └────────────────────────────┘         └──────────┬──────────────────┘  │
│                                                    │ q2 size=1            │
│                                                    ▼                      │
│                                         ┌─────────────────────────────┐  │
│                                         │ HILO 3: I2CWorkerThread     │  │
│                                         │ ────────────────────────    │  │
│                                         │ ServoKit / smbus2           │  │
│                                         │ kit.servo[ch].angle = θ     │  │
│                                         │ time.sleep(settle ~250 ms)  │  │
│                                         │ q2.task_done()              │  │
│                                         └─────────────────────────────┘  │
│                                                                          │
│  Logging: logging.handlers.QueueHandler → QueueListener (no bloquea)     │
└──────────────────────────────────────────────────────────────────────────┘

       ▲ cámara USB UVC                ▼ I2C bus → PCA9685 → servos
       │ (720p MJPG @ 30fps)           (3-4 servos en canales 0..3)
```

**Justificación de separar HILO 2 de HILO 3:** el `kit.servo[].angle = θ` puede bloquear varios cientos de ms si hay congestión del bus I2C o si esperamos settle del servo (`time.sleep`). Si HILO 2 (la FSM/decisión) se bloqueara en I2C, perdería sincronía con la inferencia. Aislando el I/O en HILO 3 con su propia cola q2 (tamaño 1, drop-oldest), la FSM siempre puede descartar comandos viejos al llegar uno nuevo (ej.: "el residuo en cola ya cambió de clase, reposiciona inmediatamente").

**Política drop-oldest en `queue.Queue`:** Python no la implementa nativamente. Implementación recomendada en el productor:

```python
try:
    q1.put_nowait(cls)
except queue.Full:
    try:
        q1.get_nowait()      # descarta el más antiguo
    except queue.Empty:
        pass
    q1.put_nowait(cls)
```

### 3. Stack concreta de librerías y versiones

| Capa | Librería | Versión exacta | Notas |
|---|---|---|---|
| SO | Ubuntu 18.04, kernel 4.9 | (incluido en JetPack 4.6.1) | EOL, sin upgrade posible |
| Runtime | Python | 3.6.9 (sistema) | No instalar Python 3.10/3.11; quiebra el wheel de TF |
| Visión low-level | OpenCV | 4.1.1 (incluida en JetPack) | Compilada con CUDA y GStreamer; usar `cv2.CAP_V4L2` |
| Inferencia | TensorFlow | `2.5.0+nv21.8-cp36-cp36m-linux_aarch64.whl` (wheel oficial NVIDIA) | Index `https://developer.download.nvidia.com/compute/redist/jp/v461`; usar `tf.lite.Interpreter` desde aquí (no hay wheel `tflite_runtime` standalone para cp36 arm64) |
| Pre-deps TF | numpy 1.19.4, h5py 3.1.0 (o 2.10.0), protobuf<4 | exactos | numpy>1.19 quiebra build; protobuf>=4 crashea TF 2.5 |
| GPIO | `Jetson.GPIO` (NVIDIA fork) | última de NVIDIA/jetson-gpio | Para sensor IR opcional. **No tiene SW PWM** |
| I2C | `Adafruit-Blinka` + `adafruit-circuitpython-pca9685` o `adafruit-circuitpython-servokit` | Blinka 8.x (no 9+) | Versiones 9+ rompen detección Nano en Ubuntu 18.04 (Stack Overflow 79551320). Alternativa libre de fricción: `smbus2` directo. |
| Máquina de estados | `python-statemachine` 2.x **o** FSM hand-rolled | — | Para MVP de 3 estados básicos (IDLE / POSICIONAR / DESPACHAR), una FSM hand-rolled de ~50 líneas alcanza |
| Logging | `logging` (stdlib) + `QueueHandler`/`QueueListener` | stdlib | No bloquear el hilo de inferencia con I/O de disco |

**Variables de entorno y arranque:**

```bash
sudo jetson_clocks                     # max performance fijo
sudo nvpmodel -m 0                     # modo MAXN
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1   # workaround static TLS
# Solo si se usa Blinka:
export BLINKA_FORCEBOARD=JETSON_NANO
```

### 4. Pipeline de inferencia: decisión TFLite vs TensorRT

**Recomendación primaria: TFLite INT8 (CPU + XNNPACK delegate)**.

| Ruta | Latencia esperada (modelo MobileNetV2-SSD 320×320) | Pros | Cons |
|---|---|---|---|
| **TFLite INT8 (CPU, XNNPACK)** | ~12-30 ms (extrapolado de tildalice + Swaminathan) | Sin cadena de conversión; cuantización post-training trivial; debugging fácil; usa los 4 cores A57 | No aprovecha la GPU Maxwell |
| TFLite INT8 (GPU delegate) | Documentado como problemático en Jetson | — | TFLite GPU delegate en Jetson requiere `libnvinfer` + `LD_LIBRARY_PATH` hacks (tildalice.io); generalmente no se recomienda |
| **TensorRT FP16** | ~24 ms SSD MobileNet v2 320×320 (NobuoTsukamoto), 14 ms MobileNetV2 (tildalice) | Aprovecha Maxwell GPU; bien documentado | Cadena PyTorch/TF → ONNX → TRT engine; rebuild si cambia el modelo; ~1.2 GB RAM workspace |
| TensorRT INT8 | **PEOR que FP16 en Nano** (Chakraborty 2025) | — | Maxwell no tiene tensor cores INT8 → fallback FP32 en muchos layers. YOLOv8n: 10 img/s INT8 vs 20 img/s FP16. **Hallazgo contraintuitivo crítico** |

**Lectura clave (research-academic, Chakraborty et al. 2025):** en TensorRT sobre Maxwell, INT8 es subóptimo porque layers incompatibles caen a FP32. **Si el camino termina siendo TensorRT, usar FP16, no INT8.** En cambio, TFLite INT8 corre en CPU con XNNPACK y los kernels de enteros sí están optimizados para ARM Cortex-A57 con NEON SIMD.

**Plan B documentado:** si la latencia medida con TFLite INT8 supera 80 ms (< 12 FPS sostenidos), migrar a TensorRT FP16 vía:

```
modelo.tf  →  saved_model  →  ONNX (tf2onnx)  →  TRT engine (trtexec)
```

con `trtexec --onnx=modelo.onnx --fp16 --workspace=1024 --saveEngine=modelo.trt`.

### 5. Modelo de visión recomendado

**Tarea fija: detección de objetos** (con bounding box + clase + score). Decidida por requisito del curso (IA en embebidos: la demostración de funcionalidades de IA es prioritaria) y por la geometría de la cámara (vista diagonal desde arriba sobre objetos acostados, con campo de visión amplio donde la posición del objeto en la imagen no es fija).

**Backbone candidato: SSD MobileNet v2 320×320 (primera opción) o EfficientDet-Lite0 320×320 (alternativa con más mAP a costa de latencia).** Fine-tuned sobre dataset propio de cartón con imagen impresa, capturado con la cámara real montada en su ángulo diagonal final (no usar TrashNet u otros datasets con vista frontal/cenital sobre fondo neutro — distribución muy distinta).

| Backbone | Input | Latencia TensorRT FP16 (Nano) | Latencia TFLite INT8 (Nano, estimado) | mAP base (COCO) | Notas |
|---|---|---|---|---|---|
| **SSD MobileNet v2** | 320×320 | **24,3 ms (~41 FPS)** (NobuoTsukamoto) | ~35-50 ms (extrapolado) | 20,18 | **Primera opción.** Fine-tuneado a 3 clases debería dar >90% mAP@50 con dataset propio bien curado |
| EfficientDet-Lite0 | 320×320 | 40,5 ms (~25 FPS) | ~55-70 ms | 23,9 | Plan B si SSD MobileNet v2 no separa bien las 3 clases |
| EfficientDet-Lite1 | 384×384 | 74,4 ms (~13 FPS) | — | 29,2 | Solo si los anteriores no alcanzan accuracy. Empieza a apretar el throughput |
| YOLOv8n | 416×416 | 33 ms (TRT, Nature 2024) | — | — | Posible pero ecosistema Ultralytics añade fricción al MVP |

**Lógica anti-flicker recomendada** (porque la detección reemplaza al sensor IR):
- N frames consecutivos con `score > 0,7` y la misma clase para confirmar "objeto presente y clasificado" (N ∈ {3, 5} a definir empíricamente según FPS real y velocidad de la banda).
- M frames consecutivos sin detección o `score < 0,5` para volver a IDLE (M > N para evitar oscilación).
- Usar IoU del bounding box entre frames consecutivos para confirmar que es el mismo objeto, no detecciones independientes.

**Implicación de la cámara diagonal:**
- El objeto aparece con perspectiva oblicua → augmentación de entrenamiento debe incluir rotaciones, perspective warps y variación de escala. No vale entrenar solo con tomas cenitales.
- El bounding box no está alineado con el objeto físico (perspectiva). Si el control de servos depende de la posición del objeto en la imagen para decidir timing de apertura, hay que mapear coordenadas imagen → coordenadas banda con una calibración (homografía) o dejar el control puramente temporal (banda mueve a velocidad fija → tiempo entre detección y caída es constante).
- Iluminación: ángulo diagonal genera sombras propias del objeto sobre la banda. Considerar luz frontal difusa (LED strip) si las pruebas muestran sombras que confunden al detector.

### 6. Veredicto frameworks pesados

| Framework | Veredicto | Razón |
|---|---|---|
| **ROS2** | **NO** | Foxy EOL (jun 2023), Humble requiere Ubuntu 22.04. Compilar Foxy desde fuente sobre 18.04 = horas perdidas. Topics/nodos son overkill para 1 cámara, 3 servos, 1 caja. |
| **NVIDIA DeepStream 6.0** | **NO** | Soporta Nano + JetPack 4.6.1 pero diseñado para multi-stream + multi-modelo. Foros reportan **4-5 FPS YOLOv5s** por overhead de plugins. Ganancia marginal vs `cv2.VideoCapture` directo en este caso de uso. |
| **GStreamer** | **CONDICIONAL** | Solo si `cv2.VideoCapture(0, cv2.CAP_V4L2)` da fps drops empíricamente. Pipeline `nvv4l2decoder` puede liberar CPU pero `videoconvert` introduce su propia latencia (NVIDIA DaneLLL). Default: empezar con OpenCV directo. |

### 7. Sensor de "objeto presente": detección por modelo (primario) + IR como fallback

**Decisión: detección por modelo de visión es la única vía primaria para identificar "hay objeto presente".** El curso es de IA en embebidos y la demostración del modelo es central a la evaluación. Apoyarse en un sensor IR como entrada primaria desviaría la prioridad de la calificación.

**Mecanismo:** el mismo modelo de detección hace doble función — clasificar el residuo Y disparar el flujo de servos cuando aparece. La transición `IDLE → POSICIONAR_SERVOS` se gatilla cuando hay N frames consecutivos con detección válida (score > umbral, clase consistente, IoU consistente entre frames).

**IR break-beam queda como FALLBACK exclusivo, a evaluar solo si:**
- En la semana 2 las pruebas empíricas muestran tasa alta de falsos positivos del modelo (sombras, fondos cambiantes, residuos parcialmente entrando al campo).
- O si la latencia de detección + anti-flicker excede el tiempo que el residuo permanece en la zona útil de la banda.

Si se activa el fallback, el patrón es: IR confirma "objeto presente" → recién entonces se considera la detección del modelo para decidir clase. Es decir, el IR sirve como gating del *trigger*, pero la *clasificación* siempre la da el modelo.

**Costo computacional aceptado:** el modelo corre continuo sobre el stream completo (no solo cuando hay objeto). Esto sube el consumo y el throttling térmico se vuelve crítico → cooler activo es no negociable. La optimización empírica posible es bajar la frecuencia de inferencia cuando IDLE (ej. 5 FPS en idle, 30 FPS cuando se detecta movimiento por diferencia de frames OpenCV), pero solo si las pruebas muestran que hace falta.

### 8. Máquina de estados (formalización de tu propuesta)

```
                    ┌──────┐
                    │ IDLE │◄──────────────────────┐
                    └───┬──┘                       │
              objeto detectado (IR / visión)       │
                        ▼                          │
                ┌────────────────┐                 │
                │ POSICIONAR_SVO │                 │
                └────────┬───────┘                 │
              servos_alcanzados (settle ~250 ms)   │
                        ▼                          │
                ┌──────────────────┐               │
                │ ABRIR_COMPUERTA  │               │
                └────────┬─────────┘               │
                  delay 100-200 ms                 │
                        ▼                          │
        ┌─────────────────────────────┐            │
        │ CERRAR_COMPUERTA + BANDA_ON │            │
        └────────┬────────────────────┘            │
                 │ residuo cae a la zona de servos │
                 ▼                                 │
        ┌──────────────────────────┐               │
        │ ESPERAR_NUEVA_PREDICCION │               │
        └────────┬─────────────────┘               │
                 │                                 │
  ┌──────────────┼───────────────────┐             │
  ▼              ▼                   ▼             │
hay objeto    hay objeto          NO hay objeto    │
clase = C'    clase ≠ C           ─────────────────┘
"keep servos"  ─► POSICIONAR_SVO
   ─► ABRIR_COMPUERTA
```

Implementación pragmática: ~50 líneas con un dict de estados → handlers. No invertir en `transitions`/`python-statemachine` para una FSM tan simple.

### 9. Snippet de inicialización canónica

Adaptado de `JetsonHacksNano/ServoKit` (research-code), patrón de cola de `spehj/yolov7-counter-jetson-nano`, y guías de NVIDIA forum:

```python
import threading
from queue import Queue, Full
import cv2
import numpy as np
import board, busio
from adafruit_servokit import ServoKit
import tensorflow as tf

# --- Modelo TFLite INT8 ---
interpreter = tf.lite.Interpreter(model_path="modelo_int8.tflite", num_threads=4)
interpreter.allocate_tensors()
in_idx  = interpreter.get_input_details()[0]['index']
out_idx = interpreter.get_output_details()[0]['index']

# --- I2C / PCA9685 ---
# DECISION EMPIRICA PENDIENTE: bus 0 (pines 27/28) vs bus 1 (pines 3/5).
# Verificar con: sudo i2cdetect -y -r 0 ; sudo i2cdetect -y -r 1
# y ver dónde aparece el 0x40 del PCA9685.
i2c = busio.I2C(board.SCL_1, board.SDA_1)   # default JetsonHacks (bus 0 según ellos)
kit = ServoKit(channels=16, i2c=i2c)

# --- Cámara ---
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- Cola con drop-oldest ---
q1 = Queue(maxsize=2)

def put_drop_oldest(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try: q.get_nowait()
        except: pass
        q.put_nowait(item)

# --- Hilo 1: captura + inferencia ---
def capture_infer_loop():
    while True:
        ok, frame = cap.read()
        if not ok: continue
        crop = preprocess(frame)               # resize 320x320 + INT8 normalization
        interpreter.set_tensor(in_idx, crop)
        interpreter.invoke()                    # libera GIL durante esta llamada
        scores = interpreter.get_tensor(out_idx)
        cls = int(np.argmax(scores))
        put_drop_oldest(q1, cls)

# --- Hilo 2/3: FSM + control servos ---
SERVO_MAP = {0: 30, 1: 90, 2: 150}             # clase -> ángulo (calibrar)
def control_loop():
    while True:
        cls = q1.get(block=True)
        target = SERVO_MAP.get(cls, 90)
        kit.servo[0].angle = target            # bloquea ~ms en I2C
        # ... resto de FSM (compuerta, banda) ...
        q1.task_done()

threading.Thread(target=capture_infer_loop, daemon=True).start()
threading.Thread(target=control_loop, daemon=True).start()
```

---

## Tabla consolidada de gotchas

| # | Gotcha | Impacto | Workaround | Fuente |
|---|--------|---------|------------|--------|
| 1 | `h5py` falla al instalar TF en Python 3.6 | TF no instala | `apt install libhdf5-serial-dev hdf5-tools && pip3 install h5py==2.10.0` | NVIDIA forum 290156 |
| 2 | `protobuf>=4` rompe TF 2.5.0 | ImportError | `pip3 install 'protobuf<4'` | NVIDIA forum 178809 |
| 3 | Static TLS block al importar TF | TF no carga | `export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` | docs.nvidia.com |
| 4 | Jetson.GPIO no tiene software PWM | Sin PWM en pines genéricos | PCA9685 vía I2C (decisión arquitectónica) | github.com/NVIDIA/jetson-gpio |
| 5 | HW PWM requiere reconfigurar pinmux | Pines 32/33 no emiten PWM | `sudo /opt/nvidia/jetson-io/jetson-io.py` + reboot | seeedstudio.com |
| 6 | `adafruit-blinka` 9+ rompe detección Nano en 18.04 | `AttributeError: JH71x0` | `pip3 install 'adafruit-blinka<9'` o usar `smbus2` | Stack Overflow 79551320 |
| 7 | `cv2.VideoCapture(0)` sin `CAP_V4L2` | Latencia 200-300 ms, fps drops | Siempre `cv2.VideoCapture(0, cv2.CAP_V4L2)` | NVIDIA forum 160486 |
| 8 | YUYV a 720p baja a 10 fps | Cámara no logra 30 fps | Forzar MJPG: `cap.set(CAP_PROP_FOURCC, 'MJPG')` | NVIDIA forum 142367 |
| 9 | TFLite `Interpreter` no thread-safe | Crash o resultados corruptos en concurrencia | Una `Interpreter` por hilo si hay paralelismo (no es el caso del MVP) | ai.google.dev/edge |
| 10 | `tflite_runtime` standalone no tiene wheel arm64 cp36 para JP46 | `ModuleNotFoundError` | Usar `tf.lite.Interpreter` desde wheel TF completo de NVIDIA | NVIDIA forum 292895 |
| 11 | Throttling térmico sin cooler activo | -36% throughput en <2 min | Cooler activo (PWM 40 mm, ~$5) **obligatorio** | tildalice.io |
| 12 | TensorRT INT8 en Maxwell falla a FP32 en muchos layers | INT8 más lento que FP16 (10 vs 20 img/s YOLOv8n) | Si TensorRT, usar FP16 (no INT8). En TFLite, INT8 sí rinde porque corre en CPU XNNPACK | Chakraborty 2025 |
| 13 | TRT engine cache en `/tmp` (RAM) come 800 MB | OOM en Nano 4 GB | `trt_engine_cache_path` a almacenamiento persistente | tildalice.io |
| 14 | Multiprocessing.Pool con TFLite cuelga `invoke()` | Hangs silenciosos | No usar multiprocessing — `threading` con GIL liberado alcanza | google-coral/edgetpu#246 |
| 15 | Threading con CPU-bound puro mata FPS | 3-7 FPS reales con objetivo 15 | Solo aplica a código Python puro; TFLite invoke() es C nativo y libera GIL | ankitbko 2022 |
| 16 | TFLite + carga CPU alta en otro hilo (TF ≤ 2.4) | Output congelado, mismo resultado siempre | Resuelto en TF ≥ 2.12. NVIDIA wheel TF 2.5+nv21.8 está después de la fix | tensorflow/tensorflow#60563 |
| 17 | "Half-cores rule" en Jetson Nano | Latencia explota con >2 procesos GPU concurrentes | Mantener exactamente 1 thread de inferencia, no paralelizar inferencias | Chakraborty 2025 |

---

## Conflictos entre fuentes / decisiones empíricas pendientes

| Conflicto | Posiciones | Resolución empírica |
|---|---|---|
| **Bus I2C correcto para PCA9685** | DiamondSheep + foro NVIDIA + research-web: bus 1 (pines 3/5). research-code citando JetsonHacksNano/ServoKit: bus 0 (pines 27/28) — afirma que el default de `ServoKit` falla silenciosamente | Conectar PCA9685 y ejecutar `sudo i2cdetect -y -r 0` y `sudo i2cdetect -y -r 1`. La dirección `0x40` aparecerá en el bus correcto. Decisión final del cableado se toma con el dato real |
| **TFLite GPU delegate en Nano** | tildalice.io reporta 12 ms con "GPU delegate" pero menciona problemas con `libnvinfer`. research-academic afirma que TFLite en Nano corre en CPU; el delegado GPU de TFLite no funciona bien con Maxwell | Si el plan A (CPU XNNPACK INT8) da menos de 12 FPS, NO intentar TFLite GPU delegate — saltar directo a TensorRT FP16 |
| **Fricción real de instalación** | NVIDIA documenta paths "limpios" pero foros reportan múltiples gotchas (numpy, h5py, protobuf, LD_PRELOAD) | Asumir 1-2 días completos solo para bring-up. Reservarlo en la planificación |

## Compra del módulo PCA9685 (guía de compra)

Sección de bolsillo para la visita a la tienda de electrónica. Asume que el comprador no tiene experiencia previa con módulos de control de servos.

### Frase central para pedirlo

> "Necesito un **módulo PCA9685 de 16 canales, 12 bits, comunicación I2C**, tipo el de **Adafruit** o un genérico equivalente. Es para **controlar 3 a 4 servomotores** desde una **NVIDIA Jetson Nano** (placa de 3,3 V en el bus I2C). **Que tenga bornera de tornillos para alimentación externa de los servos** (no quiero que los servos se alimenten desde la placa de control)."

### Por qué cada requisito

| Requisito | Justificación |
|---|---|
| **PCA9685, 16 canales, 12 bits, I2C** | Chip estándar de la industria. 16 canales > los 3-4 del proyecto (margen). 12 bits = 4096 pasos PWM (resolución de sobra). I2C es el protocolo expuesto en el header J41 de la Nano. |
| **Lógica 3,3 V compatible** | La Jetson Nano saca **3,3 V** en el bus I2C (no 5 V como Arduino). El PCA9685 acepta Vcc lógico 2,3-5,5 V, así que está bien. Si el vendedor ofrece algo "solo 5 V en la lógica", descartar. |
| **Bornera de tornillo para alimentación externa de servos** | **Crítico.** El módulo tiene dos zonas de alimentación: Vcc (lógica del chip, va a 3,3 V de la Jetson) y V+ (servos, va a fuente externa 5-6 V). Si no separa ambas zonas o no tiene bornera, no sirve para esta arquitectura. |

### Preguntas al vendedor (en orden)

1. **"¿Es PCA9685 de 16 canales?"** — Si ofrece otro chip (TLC5940, PCA9635), no es el mismo y los tutoriales/librerías no aplican.
2. **"¿Trae los headers soldados o vienen sueltos?"** — Para alguien sin experiencia, **soldados** es preferible. Si vienen sueltos, requiere soldadura.
3. **"¿Tiene bornera (terminal block) verde de tornillo para conectar fuente externa a los servos?"**
4. **"¿La lógica I2C funciona a 3,3 V?"** — Respuesta esperada: sí, 2,3-5,5 V.
5. **"¿Trae fuente o adaptador de 5 V incluido para los servos, o lo compro aparte?"** — Casi siempre aparte.
6. **"¿Dirección I2C por defecto 0x40, configurable por jumpers?"** — Default `0x40` confirmado, no se necesita cambiar (un solo módulo).

### Qué NO aceptar (banderas rojas)

- Módulos de servos "Arduino-only" o con librería propietaria → no funcionan con Python/Jetson sin trabajo extra.
- "Solo 8 canales" → otro chip (PCA9635). Funciona para el proyecto, pero las guías y código asumen 16 canales del PCA9685.
- Shields de Arduino con header 2×N específico → pines incómodos para conectar a la Jetson; preferir módulo "pelado" con headers genéricos.
- Cualquier cosa que no diga **PCA9685 textual** aunque diga "compatible".

### Lista de compra completa

1. **Módulo PCA9685 16-channel 12-bit I2C** (Adafruit o genérico equivalente) con headers soldados y bornera verde.
2. **Fuente externa 5 V / ≥3 A** (3-4 servos pequeños con picos pueden pedir 1-2 A; 3 A da holgura).
3. **Cables Dupont hembra-hembra**, mínimo 6 (VCC, GND, SCL, SDA, OE opcional, repuesto).
4. **Servos** si no se tienen aún:
   - **SG90** (plástico, livianos, baratos) — alcanza para cartón liviano.
   - **MG90S** (engranajes metálicos, más durables) — opción si se quiere robustez.
   - Cantidad sugerida: **5** (3 desviadores + 1 compuerta superior + 1 repuesto).

### Frase de respaldo si apuran

> "Es el módulo PCA9685 de Adafruit o equivalente, lo voy a usar con una Jetson Nano y servos pequeños alimentados por fuente externa. ¿Cuál tiene?"

---

## Cosas a probar empíricamente (con la Nano viva)

Lista explícita de mediciones y pruebas que deben hacerse durante S1 y S2. Cada ítem tiene un criterio de aceptación (qué número o comportamiento decide la dirección) y, donde aplica, un plan B si la prueba falla.

| # | Cosa a probar | Cómo medir | Criterio de aceptación | Plan B si falla |
|---|---|---|---|---|
| 1 | Bus I2C correcto para PCA9685 | `sudo i2cdetect -y -r 0` y `sudo i2cdetect -y -r 1`, ver dónde aparece `0x40` | Aparece `0x40` en exactamente uno de los dos buses | Probar ambas opciones de cableado físico (pines 3/5 y 27/28); el bus 0 (pines 27/28) puede chocar con CSI camera, retirarla si está ocupando |
| 2 | Cámara UVC abre a 720p/30fps en MJPG | `cv2.VideoCapture(0, cv2.CAP_V4L2)` con `CAP_PROP_FOURCC=MJPG`, `WIDTH=1280`, `HEIGHT=720`, `FPS=30`. Loop con `time.perf_counter()` midiendo intervalo entre frames | FPS sostenido ≥ 25 sin caer | Bajar a 640×480; si tampoco, probar otra cámara; último recurso: pipeline GStreamer con `nvv4l2decoder` |
| 3 | Latencia end-to-end TFLite INT8 + SSD MobileNet v2 320×320 sobre la Nano | Cargar modelo INT8 cuantizado, hacer 500 inferencias sobre input dummy tras 50 de warmup. Medir mean, P50, P95 con `time.perf_counter()` | Mean < 60 ms (≥16 FPS sostenidos) | Si está entre 60-100 ms, bajar input a 224×224. Si > 100 ms, migrar ruta TFLite INT8 → TensorRT FP16 |
| 4 | TensorRT FP16 SSD MobileNet v2 (Plan B activo) | `trtexec --onnx=modelo.onnx --fp16 --workspace=1024 --avgRuns=500` | Mean < 30 ms | Cambiar a EfficientDet-Lite0 o YOLOv8n FP16 |
| 5 | TensorRT INT8 NO se usa en Maxwell | Verificar con `trtexec --int8` que algunos layers caen a FP32 (mensajes en stderr) y que es más lento que FP16 | Confirma teoría Chakraborty 2025 → no usar | (no aplica, siempre FP16 en TRT) |
| 6 | Cooler activo necesario | Correr inferencia continua 5 minutos sin cooler. Medir `tegrastats` para ver `temp` y throttling | Sin cooler la temperatura sube > 80°C en < 2 min y FPS cae > 20% | Cooler PWM 40 mm es obligatorio antes de cualquier otra prueba sostenida |
| 7 | GIL liberado durante inferencia (verificar empíricamente) | Lanzar 2 hilos: uno con `interpreter.invoke()` en loop, otro con bucle Python intensivo. Medir FPS de ambos vs solo uno | Inferencia mantiene >80% del FPS del caso single-threaded | Si baja drásticamente, considerar `multiprocessing.Process` aunque cueste memoria |
| 8 | `Interpreter.invoke()` thread-safety | Una sola `Interpreter`, dos hilos llamando `invoke()` concurrente | Falla con error o resultados corruptos (esperado) | Confirmado, una `Interpreter` por hilo si alguna vez se necesita paralelismo |
| 9 | Latencia I2C de un comando de servo | `time.perf_counter()` antes/después de `kit.servo[0].angle = θ` | < 10 ms por comando | Si es alto, considerar `smbus2` directo en vez de Blinka |
| 10 | Tiempo de settle del servo (estabilización mecánica) | Cambiar ángulo de servo de 30° a 150° y filmar a 240 fps con celular; o usar potenciómetro lineal | < 300 ms para ángulo estable ±2° | Si es muy lento, servo MG996R en vez de SG90; o reducir delta de movimiento |
| 11 | FPS sostenido del pipeline completo (captura + inferencia + cola + control) | Correr 10 minutos, log de timestamps en cada etapa, calcular FPS por etapa y latencia end-to-end (frame entra → servo se mueve) | FPS ≥ 10 sostenido, latencia end-to-end < 500 ms P95 | Bajar resolución, usar modelo más pequeño, mover anti-flicker a más frames |
| 12 | Falsos positivos del detector con cámara real en banda real | Filmar 5 min con la banda corriendo vacía, contar detecciones espurias con score > 0,5 | < 1 falso positivo por minuto | Subir umbral, aumentar N de anti-flicker, reentrenar con augmentación de escenas vacías, considerar IR fallback |
| 13 | Detección con ángulo diagonal | Capturar 50 imágenes del residuo en distintas posiciones de la banda con la cámara montada en su ángulo final. Pasar por modelo pre-trained (sin fine-tune) y por modelo fine-tuned con dataset cenital. Medir mAP@50 | Modelo cenital cae fuerte vs diagonal → confirma necesidad de dataset propio | Capturar dataset propio con la cámara en su ángulo real desde el inicio |
| 14 | Cuantización INT8 no degrada accuracy del modelo fine-tuneado | Comparar mAP@50 del modelo float vs INT8 sobre validation set | Pérdida < 3 puntos absolutos de mAP | Si cae más, usar quantization-aware training (QAT) o aceptar FP32/FP16 si latencia lo permite |
| 15 | Tiempo de bring-up real (calibración del cronograma) | Llevar registro del tiempo total invertido en S1 antes de tener un modelo dummy corriendo end-to-end | ≤ 5 días hábiles | Si supera, comprimir S2/S3, mover features no críticos a "nice-to-have" |
| 16 | Estrategia de drop-oldest funciona | Logs de "frames descartados" en el productor durante operación normal | Drops > 0% pero < 30% del tiempo | Si drops > 50%, reducir FPS de captura o subir maxsize de la cola |
| 17 | Múltiples objetos en cola en la banda | Poner 2-3 objetos juntos en la banda, ver si el modelo distingue las clases y la FSM hace lookahead correcto | Servos se reposicionan correctamente entre objetos consecutivos sin perder ninguno | Aumentar separación mínima entre objetos en banda; o serializar (un objeto a la vez como en Steinheilig) |
| 18 | Iluminación de la escena | Probar con luz natural variable, luz artificial fluorescente, LED frontal, sin luz dirigida | Modelo mantiene mAP@50 > umbral en al menos 2 condiciones | Añadir LED strip difuso al diseño; o restringir condiciones de demo |

---

## Plan de tiempo (3 semanas hasta 2026-05-26)

| Semana | Foco | Entregables verificables |
|---|---|---|
| **S1 (5-11 may)** | Bring-up: imagen JetPack 4.6.1, pip-installs, `cv2` con cámara USB, TF 2.5+nv21.8, `tf.lite.Interpreter` cargando un modelo dummy, PCA9685 detectado en `i2cdetect` | Script `s1_smoke.py` que captura frame → corre inferencia dummy → mueve servo a 90° |
| **S2 (12-18 may)** | Modelo: dataset de 3 clases en cartón, fine-tuning MobileNetV2 (clasificación pura sobre crop fijo) o SSD MobileNet (detección), cuantización INT8 post-training. FSM básica con dos hilos, drop-oldest, log a archivo. Bench de FPS sostenido | Pipeline end-to-end sobre la mesa (sin caja): cámara → modelo → servo. ≥10 FPS sostenidos. Confusion matrix en validation set ≥85% acc |
| **S3 (19-25 may)** | Integración mecánica: caja, banda, compuerta superior, calibración de ángulos de servos desviadores, IR break-beam si tiempo. Logs persistentes. Pruebas de demo en tres clases distintas | Sistema completo en caja, demo de 10+ clasificaciones consecutivas con ≥80% acc real |
| **2026-05-26** | Entrega | — |

---

## Ronda 1 — 2026-05-05 (alto)

### Track A — Resultados de los 3 agentes

#### research-web (Jetson Nano docs y gotchas)

Cubrió 7 puntos: JetPack 4.6.x exacto, TF/TFLite, Jetson.GPIO, Adafruit-CircuitPython-PCA9685, OpenCV+V4L2, GIL en TFLite, frameworks pesados.

Hallazgos numéricos clave:
- JetPack 4.6.1 imagen SD-card directa para Nano B01. 4.6.5/4.6.6 solo via SDK Manager, sin cambios AI relevantes.
- Componentes preinstalados confirmados: TensorRT 8.2.1, cuDNN 8.2.1, CUDA 10.2, OpenCV 4.1.1, Python 3.6.9, Ubuntu 18.04.
- Wheel TF: `tensorflow-2.5.0+nv21.8-cp36-cp36m-linux_aarch64.whl` desde `developer.download.nvidia.com/compute/redist/jp/v461`.
- `tflite_runtime` standalone NO tiene wheel arm64 cp36 para JP46 → usar `tf.lite.Interpreter` que viene en wheel TF completo.
- **GIL liberado en `invoke()`:** confirmado textual en doc oficial Google AI Edge.
- Jetson.GPIO sin SW PWM, solo 2 pines HW PWM (32, 33), requieren `jetson-io.py`.
- Adafruit Blinka 9+ rompe detección Nano en 18.04 (Stack Overflow 79551320, abr 2025).
- OpenCV: siempre `CAP_V4L2`, forzar MJPG.
- Veredicto frameworks pesados: ROS2 NO, DeepStream NO, GStreamer condicional.
- TFLite en Nano corre en CPU; el delegado GPU de TFLite no es viable con Maxwell.

15 fuentes citadas con URLs.

#### research-code (Repos open source con pipeline similar)

5 repos analizados con código leído directo:

| Repo | Stars | Stack |
|---|---|---|
| `spehj/yolov7-counter-jetson-nano` | ~30 | TensorRT, threading, queue.Queue, OpenCV |
| `msubzero2000/project-ellee-public` | 77 | PyTorch/TFLite, adafruit_servokit, busio, threading, queue.Queue |
| `JetsonHacksNano/ServoKit` | 62 | adafruit_servokit, busio, init canónica I2C |
| `saeth40/Garbage-Sorting-Robot-Using-Object-Detection` | 2 | Anti-patrones (single-thread con sleeps) |
| `NVIDIA-AI-IOT/jetson-multicamera-pipelines` | 190 | GStreamer DNN pipeline + ThreadPoolExecutor |

**Patrón ganador identificado:** two-thread producer-consumer con `queue.Queue(maxsize=N)`. Documentado con snippet canónico de inicialización PCA9685.

**Anti-patrones detectados (a evitar):**
1. `time.sleep(5)` dentro del main loop de inferencia.
2. `GPIO.setmode/setup/cleanup` dentro del loop.
3. Skip de frames con contadores en el productor (preferir `put_nowait` con drop-oldest).
4. Servo controller con variables compartidas sin queue (no portable).
5. Captura+inferencia en mismo thread sin buffer/drop de frames viejos.

**Hallazgo importante (conflicto pendiente):** sobre qué bus I2C usa PCA9685 en Nano B01, research-code citando código real dice bus 0 (pines 27/28), contradiciendo a research-web y otras fuentes que dicen bus 1 (pines 3/5).

#### research-academic (Papers sobre scheduling, modelos y FSM)

5 papers/recursos con datos cuantitativos:

| # | Cita | Hallazgo numérico |
|---|---|---|
| 1 | Chakraborty et al. 2025 (arXiv 2508.08430) | "Half-cores rule" en Jetson Nano: ≤2 procesos GPU mantiene EC duration 1-2 ms; 4 procesos infla 30×; 8 procesos 70× |
| 2 | Swaminathan et al. 2024 (arXiv 2406.17749) | TensorRT speedup 2,72-16,77× sobre PyTorch baseline en Nano. MobileNetV2 16,7×. ShuffleNet V2 13,6× |
| 3 | Lew et al. 2025 (IJRAS) | MobileNetV3-Large 4,21M params: 76,8% acc en TrashNet (6 clases), 0,72 ms en GPU datacenter |
| 4 | Aral et al. 2018 (IEEE Big Data) | Paper original benchmark TrashNet 6 clases |
| 5 | Mardiati et al. 2024 (JRC) | FSM + fuzzy logic para gripper robótico; precedente del patrón FSM + servo |

**Hallazgo contraintuitivo crítico:** TensorRT INT8 en Maxwell es PEOR que FP16 porque Maxwell no tiene tensor cores INT8 → fallback a FP32 en muchos layers. YOLOv8n: 10 img/s INT8 vs 20 img/s FP16 en Nano. **Por tanto, si la ruta es TensorRT, ir a FP16, no INT8.** TFLite INT8 sí rinde porque corre en CPU con kernels enteros optimizados (XNNPACK + NEON).

**Recomendación de literatura:** 2 hilos `threading` con `queue.Queue(maxsize=2)` drop-oldest. TFLite INT8 alcanza para MVP. MobileNetV2-SSD o EfficientDet-Lite0 como backbone.

**Gaps no respondidos (medir empíricamente):**
1. Latencia end-to-end TFLite INT8 + EfficientDet-Lite0/MobileNetV2-SSD en Nano JetPack 4.6.x.
2. Latencia real de `VideoCapture.read()` con cámara USB UVC 720p en Nano.
3. Drop-oldest vs drop-newest en Python para este caso.
4. Precisión sobre cartón con imagen impresa (TrashNet usa fondos neutros).
5. Overhead I2C de PCA9685 co-ejecutando con inferencia GPU.

### Track B — Lectura activa de fuentes descubiertas

Tres queries de `discover.py` ejecutadas (90 URLs descubiertas en total). Lectura activa de 13 URLs prioritarias.

**Lecturas más relevantes:**

1. **arXiv 2406.17749 (Swaminathan)** — Pipeline canónico PyTorch → ONNX → TensorRT en Jetson Nano JetPack 4.6.1. Tabla 1 con tiempos pre-opt/post-opt: AlexNet 663→118 ms, VGG 2590→428 ms, MobileNet V2 5037→300 ms, ResNet 1091→223 ms.
2. **arXiv 2601.10582 (GIL Bottlenecks)** — Multiprocessing limitado por ~8× memoria. Adaptive thread pool da ~4× throughput. Edge devices típicamente 1-4 cores.
3. **ankitbko.github.io (Patterns Vision on Edge)** — Threading da 3-7 FPS reales con objetivo 15 FPS por GIL en código Python puro. Multiprocessing alcanza 15 FPS. Importante: este experimento NO usa TFLite — TFLite libera GIL en su `invoke()` nativo, lo que rompe la conclusión "multiprocessing siempre" del blog.
4. **tildalice.io** — MobileNetV2 INT8 TFLite Nano: 12 ms (P95 12,89 ms). ONNX+TensorRT FP16: 14 ms. Throttling sin cooler en 90 s, GPU 921→640 MHz.
5. **NobuoTsukamoto/benchmarks** — Tabla de latencias TensorRT FP16 en Nano: SSD MobileNet v2 320×320 = 24,3 ms (~41 FPS); EfficientDet-Lite0 320×320 = 40,5 ms (~25 FPS); EfficientDet-Lite1 384×384 = 74,4 ms; SSD MobileNet V2 FPNLite 320×320 = 38 ms.
6. **Nature s41598-024-74798-3 Tabla 4** — YOLOv5 640×640 TRT: 17 FPS; YOLOv7-tiny TRT: 22 FPS; YOLOv8n TRT 416×416: 30 FPS.
7. **jkjung-avt JetPack 4.6 setup** — Procedimiento exacto: `sudo jetson_clocks`, deps `numpy==1.19.4 h5py==3.1.0 protobuf`, install TF `tensorflow>=2` desde wheel NVIDIA. TF 2.6.2 verificado con CUDA OK.
8. **Steinheilig/Banknote_Counter_Jetson** — Caso real. Jetson Nano + PCA9685 + 4 motores + servo. Pipeline secuencial single-thread (un objeto a la vez): motor → servo → cámara → inferencia → motor salida. TF 2.x Keras, transfer learning VGG16, input 60×60×3 (forzado por memoria), `sudo init 3` para training.
9. **DiamondSheep/Servo_driver** — Driver C++ PCA9685 Nano. Bus default 1 (SDA pin 3, SCL pin 5), dirección 0x40. `apt install libi2c-dev i2c-tools`, `i2cdetect -y -r 1`.
10. **google-coral/edgetpu#246** — "Python multiprocessing is hard with TF". `interpreter.invoke()` cuelga en `multiprocessing.Pool`. Recomendación oficial: `set_num_threads()` para tweakeo CPU.
11. **tensorflow/tensorflow#60563** — Bug TFLite (TF ≤ 2.4): output congelado tras alta carga CPU en otro thread. Resuelto en TF 2.12. NVIDIA wheel 2.5+nv21.8 incluye la fix.
12. **e96031413/TensorFlow-Lite-Object-Detection-and-Image-Classification-on-Jetson-Nano** — Repo Python directo: PiCamera + OpenCV + TFLite + Firebase. Estructura útil pero código de 2020.
13. **forums.developer.nvidia.com/t/.../224557** — Instalación PCA9685 en Nano: `pip3 install adafruit-circuitpython-servokit`. Issues: Blinka >= 7.0.0 puede fallar en Python 3.6 (instalar Blinka separado primero); docs Adafruit obsoletas (referencias a usuario `pi`).

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | Swaminathan et al. — Benchmarking DL Models on NVIDIA Jetson Nano (2024) | https://arxiv.org/abs/2406.17749 | Paper | 1 |
| 2 | Mitigating GIL Bottlenecks in Edge AI Systems | https://arxiv.org/html/2601.10582v4 | Paper | 1 |
| 3 | Chakraborty et al. — Profiling Concurrent Vision Inference on Jetson (2025) | https://arxiv.org/abs/2508.08430 | Paper | 1 |
| 4 | Lew et al. — Reproducible Benchmark for Trash Classification (2025) | IJRAS | Journal | 1 |
| 5 | Aral et al. — TrashNet Classification (2018) | https://ieeexplore.ieee.org/document/8621586 | Conference | 1 |
| 6 | Mardiati et al. — FSM + Fuzzy Logic Robot Gripper (2024) | https://doi.org/10.18196/jrc.v5i3 | Journal | 1 |
| 7 | Patterns for Vision on the Edge - Concurrent Processing | https://ankitbko.github.io/blog/2022/06/vision-on-edge-part-1/ | Blog técnico | 1 |
| 8 | Raspberry Pi 5 vs Jetson Nano benchmark | https://tildalice.io/raspberry-pi-5-vs-jetson-nano-ml-inference-benchmark/ | Blog técnico | 1 |
| 9 | jkjung-avt — JetPack 4.6 setup | https://jkjung-avt.github.io/jetpack-4.6/ | Blog técnico | 1 |
| 10 | NobuoTsukamoto — EfficientDet TensorRT Jetson Nano benchmarks | https://github.com/NobuoTsukamoto/benchmarks/blob/main/tensorrt/jetson/detection/README.md | Repo | 1 |
| 11 | Nature s41598-024-74798-3 — Comparison Jetson Nano YOLO speeds | https://www.nature.com/articles/s41598-024-74798-3/tables/4 | Journal | 1 |
| 12 | Steinheilig — Banknote Counter Jetson + PCA9685 | https://github.com/Steinheilig/Banknote_Counter_Jetson | Repo | 1 |
| 13 | DiamondSheep — Servo_driver Jetson Nano PCA9685 | https://github.com/DiamondSheep/Servo_driver | Repo | 1 |
| 14 | spehj — yolov7-counter-jetson-nano | https://github.com/spehj/yolov7-counter-jetson-nano | Repo | 1 |
| 15 | msubzero2000 — project-ellee-public (Jetson Nano + ServoKit) | https://github.com/msubzero2000/project-ellee-public | Repo | 1 |
| 16 | JetsonHacksNano — ServoKit | https://github.com/JetsonHacksNano/ServoKit | Repo | 1 |
| 17 | saeth40 — Garbage-Sorting-Robot-Using-Object-Detection | https://github.com/saeth40/Garbage-Sorting-Robot-Using-Object-Detection | Repo | 1 |
| 18 | NVIDIA-AI-IOT — jetson-multicamera-pipelines | https://github.com/NVIDIA-AI-IOT/jetson-multicamera-pipelines | Repo | 1 |
| 19 | e96031413 — TFLite Object Detection on Jetson Nano | https://github.com/e96031413/TensorFlow-Lite-Object-Detection-and-Image-Classification-on-Jetson-Nano | Repo | 1 |
| 20 | google-coral/edgetpu#246 — Does tflite_runtime support multiprocessing? | https://github.com/google-coral/edgetpu/issues/246 | Issue | 1 |
| 21 | tensorflow/tensorflow#60563 — TFLite stops after high CPU on different thread | https://github.com/tensorflow/tensorflow/issues/60563 | Issue | 1 |
| 22 | NVIDIA forum — Adafruit PCA9685 installation on Jetson Nano | https://forums.developer.nvidia.com/t/adafruit-board-pca9685-installation-of-drivers-on-jetson-nano/224557 | Foro oficial | 1 |
| 23 | NVIDIA — JetPack SDK 4.6.5 | https://developer.nvidia.com/jetpack-sdk-465 | Doc oficial | 1 |
| 24 | NVIDIA — JetPack SDK 4.6.6 | https://developer.nvidia.com/jetpack-sdk-466 | Doc oficial | 1 |
| 25 | NVIDIA — JetPack Archive | https://developer.nvidia.com/embedded/jetpack-archive | Doc oficial | 1 |
| 26 | NVIDIA — TF for Jetson Platform Release Notes | https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html | Doc oficial | 1 |
| 27 | NVIDIA — Installing TensorFlow for Jetson Platform | https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html | Doc oficial | 1 |
| 28 | NVIDIA/jetson-gpio README | https://github.com/NVIDIA/jetson-gpio | Repo oficial | 1 |
| 29 | JetsonHacks — Jetson Nano J41 Header Pinout | https://jetsonhacks.com/nvidia-jetson-nano-j41-header-pinout/ | Blog técnico | 1 |
| 30 | Seeed Studio — PWM output Jetson Nano | https://www.seeedstudio.com/blog/2020/05/27/configure-pwm-output-on-jetson-nano-m/ | Blog técnico | 1 |
| 31 | Google AI Edge — tf.lite.Interpreter API doc (GIL release) | https://ai.google.dev/edge/api/tflite/python/tf/lite/Interpreter | Doc oficial | 1 |
| 32 | NVIDIA forum — OpenCV VideoCapture doesn't work | https://forums.developer.nvidia.com/t/opencv-videocapture-doesnt-work/160486 | Foro oficial | 1 |
| 33 | NVIDIA forum — Adafruit PCA9685 I2C error | https://forums.developer.nvidia.com/t/adafruit-servo-driver-pca9685-not-working-due-to-i2c-error/191064 | Foro oficial | 1 |
| 34 | Adafruit Blinka Issue #344 — Errno 121 | https://github.com/adafruit/Adafruit_Blinka/issues/344 | Issue tracker | 1 |
| 35 | Stack Overflow 79551320 — Adafruit-Blinka Jetson Nano Ubuntu 18.04 | https://stackoverflow.com/questions/79551320 | Foro comunidad | 1 |
| 36 | NVIDIA forum — h5py install error Jetson Nano | https://forums.developer.nvidia.com/t/cannt-install-tensorflow-due-to-h5py-jetson-nano/290156 | Foro oficial | 1 |

---

## Próximas direcciones (rondas siguientes posibles)

1. **Modelo y dataset:** búsqueda específica de datasets de cartón con imagen impresa, técnicas de augmentación para fondos heterogéneos, TF Lite Model Maker workflow para fine-tuning rápido.
2. **Calibración mecánica de servos:** investigar mecanismos del "servo central" (V invertida, retracción, levantamiento, doble paleta) — ya identificado como problema mecánico, pendiente de investigación.
3. **Validación empírica de los gaps de Track A:** medir TFLite INT8 + MobileNetV2 INT8 en la Nano real, latencia VideoCapture, throughput sostenido sin throttling.
4. **Logging y observabilidad:** patrones de logging asíncrono en Python con `QueueHandler`/`QueueListener`, rotación, formato para post-mortem.
