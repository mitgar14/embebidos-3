# Investigación — Detector de proximidad / orden de caída sobre banda transportadora

**Dominio:** `embebidos-3` — determinar qué objeto detectado está más próximo a caer por el hueco de la banda y encolar los lejanos con un *time-to-fall* estimado, para pre-posicionar los 3 servos por clase y reducir la latencia de actuación.
**Proyecto:** clasificador 3 clases (`glass`/`paper`/`plastic`), demo **2026-05-26 (T-3 días)**.
**Ronda 1 inaugural** — dominio nuevo. Cruza con: `2026-05-15-mejoras-modelo-deteccion-plasticos.md` (recall del detector), `HANDOFF-track-b-2026-05-13.md` (runtime Nano), y el pipeline de streaming (engine FP16 @416, ~43 FPS).

---

## Contexto operacional del problema

- **Montaje:** cámara Logitech C920 OG (FOV 78°) **fija en ángulo diagonal** mirando hacia abajo a una **banda transportadora plana**. Los objetos se arrastran hacia un **hueco** (al final de la banda) por donde caen. Debajo del hueco, **3 servos SG90** (vía PCA9685, I²C) redirigen cada objeto a su clase (1 de 3).
- **Detector existente:** YOLOv8n FP16 → TensorRT @416, ~43 FPS / 23 ms, ya entrega bboxes multiclase. El modelo fue fine-tuneado para este entorno (iluminación/fondos variados).
- **Problema a resolver:** la decisión de actuación toma tiempo. Para reducir latencia, el sistema debe (1) saber qué objeto está **próximo a caer** (para pre-posicionar los servos) y (2) tratar los objetos **lejanos** como una **cola** con un instante de actuación previsto. Lo que faltaba: **cómo saber qué objeto está más lejos/cerca del hueco con la cámara.**

**Supuestos MVP (paramétricos, fijados 2026-05-23 a falta de medición física):** banda a velocidad ~constante (estimable por tracking), hueco al final, ~1 objeto cayendo a la vez, latencia de barrido SG90 entre clases ~300-500 ms (define el *lookahead* mínimo de la cola).

---

## Resumen ejecutivo — el problema NO es *depth estimation*

> **Reframe central (validado por el grounding del Nano y por Track B):** con cámara **fija** sobre una banda **plana**, "qué objeto está más cerca del hueco" **no requiere estimación de profundidad monocular**. Es un problema de **homografía / Inverse Perspective Mapping (IPM)**: existe una transformación proyectiva 3×3 entre el plano de la imagen y el plano de la banda. Mapeando el punto de contacto de cada bbox a coordenadas físicas sobre la banda, la proximidad al hueco es **directa**. La profundidad monocular (MiDaS/Depth Anything) sería cara, sin `torch` en el runtime, y **redundante** dada la geometría plana fija.

**Arquitectura recomendada (a validar/afinar con Track A):**

1. **Calibración de homografía (1 sola vez):** 4 puntos de la banda con coordenadas físicas conocidas (p. ej. esquinas de un rectángulo de tamaño conocido) → `H = cv2.getPerspectiveTransform(src_img_pts, dst_belt_pts)` (3×3). Se recalibra solo si la cámara se mueve.
2. **Por frame (sobre los bboxes que YOLOv8n YA produce):** punto de contacto del objeto = `(x_centro, y2)` (centro inferior del bbox) → `cv2.perspectiveTransform` con `H` → coordenada `(u, v)` en cm sobre la banda → **distancia al hueco** = `dist(v, v_hueco)` a lo largo del eje de avance.
3. **Tracker multi-objeto NumPy puro** (SORT-lite: asociación IoU/centroide con `scipy.optimize.linear_sum_assignment` + modelo de velocidad constante) → mantiene identidad entre frames y estima **velocidad en cm/s** sobre la banda.
4. **`time_to_fall = distancia_al_hueco / velocidad`** por objeto → **ordenar** ⇒ *next-to-fall* + **cola** con instante de actuación.
5. **Scheduler de servos** (`smbus` → PCA9685) consume la cola y pre-posiciona la rampa de la clase correspondiente con el *lookahead* adecuado.

**Costo de cómputo ≈ 0:** un producto matriz-vector 3×3 por detección + asociación de pocas cajas. **No añade ninguna red neuronal** ni compite por la GPU Maxwell con el detector. Todo cabe en `cv2 4.1.1` + `numpy 1.13.3` + `scipy 0.19.1` ya presentes en el Nano.

**Quick wins ordenados por costo × impacto:**
1. **IPM + ordenar por distancia al hueco** (sin tracking) — da el *next-to-fall* inmediato. Pocas horas.
2. **Tracker SORT-lite NumPy** — añade identidad + velocidad ⇒ habilita la cola con *time-to-fall*.
3. **Scheduler de servos smbus→PCA9685** — capa de actuación (greenfield).
4. **Visualización en el dashboard** (`drawDetections`): resaltar *next-to-fall*, dibujar la línea del hueco y la cola con su ETA.

**Negative constraints (restricciones duras confirmadas):**
- **NO usar `supervision`/`ultralytics`/`filterpy`/`lap` como librería** — ausentes y no instalables en Py3.6.9 del Nano. Se usan solo como *referencia de algoritmo*; el tracker se reimplementa en NumPy+scipy.
- **NO meter una red de profundidad monocular** — innecesaria (geometría plana fija), cara en Maxwell sm_53 (RT-MonoDepth-S ≈ **54 ms / 18 FPS** en Nano vs **<1 ms** de la homografía) y sin `torch` en runtime; además daría profundidad **relativa, no métrica**.
- **NO depender de `apparent size`** como cue de distancia: las 3 clases tienen tamaños heterogéneos ⇒ poco fiable frente a la posición-sobre-plano.
- La **homografía se calibra a la resolución de captura** del cliente (los frames llegan del browser a 640×480); los bboxes vienen en coords del frame original.

---

## Track 0 — Grounding del Nano (SSH read-only, 2026-05-23)

Estado real verificado en el dispositivo antes de proponer el approach (no asumido):

| Qué | Hallazgo | Consecuencia de diseño |
|---|---|---|
| **Runtime Python** | System **3.6.9** (`/usr/bin/python3.6`, sin `/opt/venv`) | Código Py3.6-safe; tracker en stdlib+numpy+scipy |
| **Paquetes presentes** | `numpy 1.13.3`, `cv2 4.1.1`, `scipy 0.19.1`, `pycuda`, `Jetson.GPIO`, `smbus`, `tensorrt 8.2.1.8`, `fastapi 0.65.3`, `pydantic 1.9.2` | Todo lo necesario YA está |
| **Capacidades cv2/scipy** | `getPerspectiveTransform` ✓, `perspectiveTransform` ✓, `scipy.optimize.linear_sum_assignment` ✓ | Homografía + asociación Húngara disponibles nativamente |
| **Ausentes (no usar como lib)** | `supervision`, `ultralytics`, `filterpy`, `lap`, `lapx`, `smbus2`, `pyserial` → MISSING | Tracker = NumPy puro; servos = `smbus` (no Adafruit) |
| **Origen de frames** | Browser (`dashboard/app.js`, `getUserMedia` ideal **640×480**, JPEG q0.7) → WS `/ws` → Nano. **No hay `/dev/video*`** en el Nano | Cámara en la laptop; homografía calibrada a 640×480 |
| **Salida del detector** | `{x1,y1,x2,y2,conf,cls,cls_name}` en coords del **frame original**, `CLASSES=["glass","paper","plastic"]` | Input perfecto para IPM; engancha **post-`_postprocess`** |
| **Worker** | `TRTWorker` (1 hilo, secuencial, `seq` incremental por frame; `queue maxsize=2` con drop) | Estado del tracker se mantiene server-side entre frames |
| **Actuación** | 9 buses `/dev/i2c-*` + `smbus` y `Jetson.GPIO` presentes; **sin código de servos** (los `.ino` de `firmware/` son bench-tests Arduino; **no hay Arduino por USB** — sin `/dev/ttyACM*`) | Servos desde el Nano por `smbus`→PCA9685 (0x40); capa **greenfield** |
| **Modelo activo** | HF `mitgar14/embebidos-3-models-v1c`, rev `2e80e24`, engine FP16 13.5 MB, build 2026-05-23 | — |

**Puntos de integración concretos:**
- **Lógica de proximidad/cola/servos:** en el Nano, en `scripts/server/nano_server.py`, justo **después de `TRTWorker._postprocess`** (que ya devuelve bboxes en coords del frame original). Enriquecer el JSON del WS con `track_id`, `belt_xy`, `dist_to_hole`, `time_to_fall`, `order`, `next_to_fall`.
- **Visualización:** en `scripts/dashboard/app.js`, función `drawDetections` (overlay canvas) — resaltar el *next-to-fall*, dibujar la línea del hueco y la cola.
- **Actuación:** nuevo módulo Nano-side `smbus`→PCA9685 (no existe aún).

---

## Track B — Descubrimiento web + extracción (discover.py + Exa)

### B1 — IPM / homografía (docs + práctica)

El recipe es estándar y está plenamente soportado por `cv2 4.1.1`:
- **`cv2.getPerspectiveTransform(src, dst)`** con 4 correspondencias → `H` 3×3; **`cv2.perspectiveTransform(pts, H)`** mapea puntos imagen→plano banda. (Para >4 puntos y robustez: `cv2.findHomography(..., cv2.RANSAC)`.)

| Fuente | URL | Valor |
|---|---|---|
| OpenCV — Basic concepts of homography (oficial) | https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html | Teoría + código C++/Python/Java de referencia |
| OpenCV — Geometric Image Transformations | https://docs.opencv.org/ref/master/da/d54/group__imgproc__transform.html | API `getPerspectiveTransform`/`warpPerspective`/`perspectiveTransform` |
| PyImageSearch — 4 Point getPerspectiveTransform | https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ | Recipe canónico de 4 puntos (módulo `transform.py`) |
| StackOverflow — *Camera pixels to planar world points given 4 known points* | https://stackoverflow.com/questions/25769707/ | **Caso idéntico**: cámara mirando al piso en ángulo → coords 3D sobre el plano dados 4 puntos conocidos |
| StackOverflow — *Image perspective correction and measurement with a homography* | https://stackoverflow.com/questions/70038778/ | Medición de distancias reales con homografía en pose fija |
| arXiv 1905.02231 — *A Geometric Approach to Obtain a Bird's Eye View From an Image* | https://ar5iv.labs.arxiv.org/html/1905.02231 | Rectificación a vista cenital computando H |
| TheLinuxCode — Perspective Warp (homography, 4-point, real-time) | https://thelinuxcode.com/perspective-warp-in-python-with-opencv-homography-four-point-mapping-and-real-time-camera-views/ | Tutorial práctico end-to-end |

### B2 — Trackers multi-objeto en NumPy puro (repos)

| Repo | URL | Por qué importa |
|---|---|---|
| **adipandas/multi-object-tracker** (`motrackers`) | https://github.com/adipandas/multi-object-tracker | **Hallazgo clave**: `sort_tracker.py` usa `scipy.optimize.linear_sum_assignment` + IoU NumPy (sin filterpy); incluye `CentroidTracker`. Exactamente el patrón Py3.6-safe que buscamos |
| abewley/sort (canónico) | https://github.com/abewley/sort/blob/master/sort.py | Referencia del algoritmo SORT (usa `filterpy` → reescribir el Kalman en NumPy) |
| yakhyo/sort-tracker | https://github.com/yakhyo/sort-tracker | Implementación SORT mínima |
| PacktPublishing — OpenCV-4-with-Python-Blueprints (cap.10) | https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition/blob/master/chapter10/sort.py | `KalmanBoxTracker` didáctico |
| mohamedamine99/Object-tracking-and-counting-using-YOLOV8 | https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8 | Tracking + **conteo por cruce de línea** (referencia para el hueco) |

### B3 — PCA9685 vía `smbus` puro (sin Adafruit, Py3.6-OK)

Múltiples drivers `import smbus` a nivel de registros (MODE1, PRESCALE, LED0_ON_L/H), dirección I²C típica **0x40**, SG90 a 50 Hz con pulso 1-2 ms:

| Driver | URL |
|---|---|
| ControlEverythingCommunity/PCA9685 (Python) | https://github.com/ControlEverythingCommunity/PCA9685/blob/master/Python/PCA9685.py |
| Waveshare Pan-Tilt-HAT — PCA9685.py | https://github.com/waveshare/Pan-Tilt-HAT/blob/master/RaspberryPi/Servo_Driver/python/PCA9685.py |
| Freenove Robot Dog — PCA9685.py | https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi/blob/master/Code/Server/PCA9685.py |
| divadnoslo/Servo_Control — PCA9685.py + ServoDriver.py | https://github.com/divadnoslo/Servo_Control/blob/main/PCA9685.py |
| Grippy98 gist — PCA9685 Driver | https://gist.github.com/Grippy98/7ef6a75b2dc7a9470bd8c4dfc6b53f0a |

> Nota: el `i2c_scanner.ino` de `firmware/` sugiere que la dirección del PCA9685 ya se verificó por I²C. Confirmar el bus correcto en el Nano (`i2cdetect -y -r <bus>`) entre `/dev/i2c-0..8`.

---

## Track A — Agentes de research

> 5 agentes background lanzados 2026-05-23 (profundidad Media + video ligero). **Consolidado completo: 5/5** (research-code, research-academic, 3× research-video).

### A1 — Tracker multi-objeto NumPy puro (research-code)

**Ganador: `bochinski/iou-tracker`** — 68 LOC funcionales, **solo `numpy` + stdlib** (sin filterpy/lap/torch), **licencia MIT**, repo activo (702★). Asociación *greedy* por IoU (sin asignación global): para una banda **unidireccional a velocidad ~constante** los objetos no se cruzan en el eje perpendicular y, a 43 FPS, el IoU entre frames consecutivos es alto ⇒ el greedy basta. Adaptación estimada **~80 LOC**: añadir `track_id`, `cls_name`, historial del bottom-center proyectado y **velocidad por EMA**. Py3.6 + numpy 1.13.3 sin cambios (`np.expand_dims` OK desde 1.7).
→ https://github.com/bochinski/iou-tracker/blob/master/iou_tracker.py

| Tracker | LOC a escribir | Deps | Adecuación banda | Riesgo |
|---|---|---|---|---|
| **iou-tracker (bochinski)** | ~80 (adaptación) | numpy + stdlib | **Alta** (movimiento rectilíneo) | Mínimo |
| SORT + Kalman NumPy | ~160 (reimpl. Kalman) | numpy + scipy | Alta | Medio |
| Centroid tracker | ~60 | numpy + stdlib | Media (sin predicción) | Mínimo |

**Si se quisiera Kalman** (no necesario para el MVP): SORT usa `filterpy` (prohibido), pero `iou_batch` + `associate_detections_to_trackers` son 100% NumPy + `scipy.linear_sum_assignment` (presente en 0.19.1; `lap` no hace falta). El Kalman de velocidad constante 7-D (matrices F, H, P, Q, R hardcodeadas) se reimplementa en ~40 LOC tomando como referencia `nwojke/deep_sort/deep_sort/kalman_filter.py` (NumPy puro, usa `scipy.linalg`).
**Argumento para saltarse el Kalman:** velocidad ~constante + 43 FPS ⇒ una **EMA del `delta_d`** sobre los últimos 3-5 frames (en el espacio de la banda, post-homografía) da la velocidad para el `time_to_fall`. El Kalman solo suaviza; no es crítico.
> **Licencia:** `bochinski/iou-tracker` es MIT (sin restricciones). `abewley/sort` y el `sort.py` de ByteTrack son **GPL-3.0** y además dependen de `filterpy` → descartados como copia directa.

### A2 — Driver de servos PCA9685 vía `smbus` (research-code)

**Fuente de referencia: `waveshare/Pan-Tilt-HAT` → `PCA9685.py`** (solo `smbus`/`time`/`math`, copiable tal cual a Py3.6). Dirección I²C **0x40** (pines de dirección a 0; verificar con `i2cdetect -y -r 1`), **bus I²C-1** (pines 3/5 del header de 40 pines del Nano). Mapa de registros: `MODE1=0x00`, `PRESCALE=0xFE`, `LED0_ON_L=0x06`…`LED0_OFF_H=0x09` (stride 4 por canal). Para 50 Hz: `prescale = round(25e6/(4096·50) − 1) = 121`.

```python
import smbus, time, math
bus = smbus.SMBus(1); ADDR = 0x40
MODE1=0x00; PRESCALE=0xFE; LED0_ON_L=0x06

def set_pwm_freq(hz):
    pre = int(math.floor(25000000.0/(4096.0*hz) - 1 + 0.5))   # 50 Hz -> 121
    old = bus.read_byte_data(ADDR, MODE1)
    bus.write_byte_data(ADDR, MODE1, (old & 0x7F) | 0x10)     # SLEEP
    bus.write_byte_data(ADDR, PRESCALE, pre)
    bus.write_byte_data(ADDR, MODE1, old); time.sleep(0.005)
    bus.write_byte_data(ADDR, MODE1, old | 0x80)              # RESTART

def set_pwm(ch, on, off):
    b = LED0_ON_L + 4*ch
    bus.write_byte_data(ADDR, b,   on  & 0xFF); bus.write_byte_data(ADDR, b+1, on  >> 8)
    bus.write_byte_data(ADDR, b+2, off & 0xFF); bus.write_byte_data(ADDR, b+3, off >> 8)

def set_servo_angle(ch, deg):                 # SG90: 500-2400 us, periodo 20000 us @50Hz
    off = int((500 + (deg/180.0)*(2400-500)) * 4096 / 20000)
    set_pwm(ch, 0, off)
# ch0->glass, ch1->paper, ch2->plastic
```
→ https://github.com/waveshare/Pan-Tilt-HAT/blob/master/RaspberryPi/Servo_Driver/python/PCA9685.py
→ alternativa funcional (sin clase): https://github.com/AlexandreFrolov/ri-controller-i2c (`pca9685_test.py`)

### A3 — Conveyor CV + descarte de depth (research-video)

**Patrón establecido detect→track→count→trigger** (Roboflow, *Track & Count con YOLOv8 + ByteTrack + Supervision*, 189k views): una **línea virtual** (`LineZone`) dispara el evento cuando el centroide la cruza — exactamente el *trigger* de actuación que necesitamos, reimplementable como **comparación de signo de una coordenada umbral en el espacio de la banda** (post-homografía), sin la lib `supervision`.
- "counting objects moving on a conveyor … exactly the same code in both cases" — https://youtu.be/OS5qI9YBkfk?t=52
- Chapter *Counting objects crossing the line* — https://youtu.be/OS5qI9YBkfk?t=1060 · *…candies on the conveyor* — https://youtu.be/OS5qI9YBkfk?t=1370
- Pysource (*objects on a conveyor belt*, RPi+OpenCV): disparar la acción cuando el objeto llega a una posición X del frame — https://youtu.be/A29IqeahI84?t=83

**Descarte de profundidad monocular (cuantitativo):** FastDepth (MIT, el más liviano para embedded) corre a **178 FPS en Jetson TX2** (Pascal sm_62, 256 cores); extrapolado al Nano (Maxwell sm_53, 128 cores) → ~40-60 FPS **pero produce profundidad RELATIVA, no métrica**. MiDaS/Depth Anything V2 con calidad métrica caen a **<1-5 FPS** en sm_53. Frente a esto, la homografía es **algorítmica, determinista y <1 ms**. ⇒ **depth NN descartada** (innecesaria + cara + no métrica sin calibración).
- *FastDepth* (MIT.nano): "178 fps on the Jetson TX2, 8.8 W" — https://youtu.be/7EdFpgcD8vk

### A4 — IPM/homografía práctica (research-video)

- **Flujo de 4 puntos** idéntico en todos los tutoriales: `pts_src`/`pts_dst` `np.float32` → `M = cv2.getPerspectiveTransform` (1 sola vez al arranque) → `cv2.perspectiveTransform` sobre el **centroide/bottom-center** del objeto (NO `warpPerspective` del frame completo — más eficiente en el Nano).
- **Calibración interactiva con mouse** (`cv2.setMouseCallback`) — patrón recomendado para recalibrar en campo sin tocar código; aplicable a video en vivo (OMES, español): https://youtu.be/lkwtwhWKQjo?t=493
- **Gotcha de orden de puntos**: src y dst en el mismo orden (TL, TR, BR, BL); un orden cruzado tuerce la homografía (Murtaza's Workshop).
- **API invariante** cv2 4.1↔4.2: el código es portable directo al Nano (OpenCV 4.1.1).
- Validación visual: superponer líneas virtuales con la `M` calibrada (Roboflow, *Homography Fundamentals* https://youtu.be/aBVGKoNZQUw?t=3106 · *Top-Down Projection* https://youtu.be/aBVGKoNZQUw?t=4403).

### A5 — IPM teórico + prior art de *conveyor sorting timing* + descarte de depth (research-academic)

**IPM sobre plano conocido — fundamento y error de calibración:**
- Bertozzi, Broggi & Fascioli (1998), *Stereo Inverse Perspective Mapping* (DOI 10.1016/S0262-8856(97)00093-0) — referencia fundacional: la homografía `H` mapea imagen→coordenadas métricas del plano con costo O(1)/punto una vez calibrada. Idéntico a una banda plana vista en ángulo fijo.
- **Szulc & Iwanowski (2026)**, arXiv:2604.10805 — análisis directo para nuestro caso: el error de distancia en homografías inicializadas a mano crece **cuadráticamente con el rango** (ΔY ≈ (ε/K)·Y²); con **solo 3 puntos** se corrige 76-90% del error. ⇒ una calibración simple de 4 esquinas de la banda es robusta **siempre que el objeto esté sobre el plano** (siempre, en la banda).

**Prior art de timing de actuación en sorters (lo más citable para el informe IEEE):**
- **ZenRobotics** — Kujala, Lukka & Holopainen (2015), arXiv:1511.07608, *Picking a Conveyor Clean by an Autonomously Learning Robot*: pipeline industrial con **conveyor tracking**, una **"working area"** donde el actuador intercepta, y **retry policy**. Análogo directo: homografía + SORT-lite + time-to-fall = los módulos base; lo propio del proyecto = la prioridad de clase (glass/paper/plastic) para pre-posicionar el servo.
- Zhang et al. (2016), *Dynamic Conveyor Tracking Control of a Delta Robot*: sincroniza la velocidad del robot con la banda usando un **modelo de velocidad constante** para predecir la posición del ítem en el instante de intercepción. Base del **induction timing**: `t_ahead = d_to_gap / v_belt` (el adelanto compensa la latencia del actuador).
- **Zocco et al. (2024)**, IEEE TIM, arXiv:2405.06821, *Synchronized Object Detection for Autonomous Sorting*: framework de **sincronización detección↔actuación** a 12-22 FPS con gestión de **cola de ítems** ("synchromaterial"). Código: `fedezocco/2MMUsMed`. Lo más cercano arquitecturalmente a embebidos-3.

**Modelo de time-to-fall consolidado:** `y_belt(t) = y0 − v_belt·t`, con `y0` de la homografía y `v_belt` de SORT proyectado al plano ⇒ `time_to_fall = (y0 − y_gap) / v_belt`. Pre-posicionar el servo cuando `time_to_fall ≤ lookahead` (300-500 ms del SG90). La cola de objetos lejanos se mantiene ordenada por `time_to_fall`, refrescada cada frame.

**Descarte cuantitativo de profundidad monocular:**

| Método | Hardware | Latencia/frame | Ref |
|---|---|---|---|
| **Homografía** (`perspectiveTransform`) | Jetson Nano sm_53 | **<1 ms** | op. matricial |
| RT-MonoDepth-S | **Jetson Nano sm_53** | **54 ms (18.4 FPS)** | arXiv:2308.10569 |
| FastDepth (pruned) | Jetson TX2 Pascal (>Nano) | 5.6 ms (178 FPS) | arXiv:1903.03273 |
| Depth Anything V2-Small (25M) | no medido en Nano | ≫54 ms estimado | arXiv:2406.09414 |

La depth monocular es además **relativa (sin escala métrica)** y **redundante** sobre plano fijo: el *footprint* del objeto en el plano (lo único que importa para el time-to-fall) ya queda determinado por la homografía, sin ambigüedad. ⇒ **IPM supera a depth NN en costo, interpretabilidad y precisión**.

**Dominio *waste sorting* validado (citable):** SpectralWaste (Casao et al., IROS 2024, arXiv:2403.18033) — dataset de plásticos/cartón sobre banda con cámara fija; WasteGAN (Bacchin et al., IROS 2024, arXiv:2409.16999); benchmarks DL en Jetson Nano (Swaminathan et al., arXiv:2406.17749).

### A6 — Tracking multi-objeto *from scratch* (research-video)

Confirma que el tracker se reimplementa en NumPy+SciPy sin `filterpy`:
- **Kalman velocidad constante** (estado `[u,v,s,r, du,dv,ds]`; `F`=identidad + dt en velocidades; `H` observa `[u,v,s,r]`): predict `x=Fx`, `P=FPFᵀ+Q`; update `K=PHᵀ(HPHᵀ+R)⁻¹`, `x=x+K(z−Hx)`, `P=(I−KH)P`. — *Visually Explained: Kalman Filters* (354k views): predict https://youtu.be/IFeCIbljreY?t=337 · update https://youtu.be/IFeCIbljreY?t=434
- **Asociación Húngara + IoU**: matriz de costo `1−IoU` (N×M) → `scipy.optimize.linear_sum_assignment` → filtrar matches con IoU<0.3; dets sin match → tracks nuevos; tracks sin match N frames → eliminar. — *DataMListic*: cost matrix https://youtu.be/oo-H_ZY2TGA?t=121 · hungarian https://youtu.be/oo-H_ZY2TGA?t=140
- **ByteTrack** (2 rondas: alta conf → baja conf) recupera objetos parcialmente ocluidos; útil solo si dos objetos se solapan en la banda. — *Kevin Wood* https://youtu.be/6LGpf-a1K1Q?t=117
- **Centroid tracking** mínimo (distancia euclidiana + `linear_sum_assignment`, sin Kalman) como fallback. — *Pysource* (60 min) https://youtu.be/GgGro5IV-cs?t=783

**Esquema de reimplementación** (coincide con A1): `iou_batch(np)` · `associate_detections_to_trackers(iou_thr=0.3)` · clase `Sort` · `KalmanBoxTracker` (opcional). **Conclusión convergente con el agente de código:** para banda unidireccional a 43 FPS el **greedy IoU (iou-tracker) basta**; el Kalman es opcional (suaviza, no decide).

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-23 | Media + video ligero | Detector de proximidad / orden de caída: homografía/IPM + tracker NumPy puro + scheduler servos smbus; descarte de depth monocular |

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | OpenCV — Basic concepts of homography | https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html | Docs | 1 |
| 2 | OpenCV — Geometric Image Transformations | https://docs.opencv.org/ref/master/da/d54/group__imgproc__transform.html | Docs | 1 |
| 3 | PyImageSearch — 4 Point getPerspectiveTransform | https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/ | Tutorial | 1 |
| 4 | SO — Camera pixels to planar world points (4 known pts) | https://stackoverflow.com/questions/25769707/ | Q&A | 1 |
| 5 | SO — Perspective correction & measurement with homography | https://stackoverflow.com/questions/70038778/ | Q&A | 1 |
| 6 | arXiv 1905.02231 — Bird's Eye View via homography | https://ar5iv.labs.arxiv.org/html/1905.02231 | Paper | 1 |
| 7 | adipandas/multi-object-tracker (motrackers) | https://github.com/adipandas/multi-object-tracker | Repo | 1 |
| 8 | abewley/sort | https://github.com/abewley/sort | Repo | 1 |
| 9 | mohamedamine99 — Object tracking & counting YOLOv8 | https://github.com/mohamedamine99/Object-tracking-and-counting-using-YOLOV8 | Repo | 1 |
| 10 | ControlEverythingCommunity/PCA9685 (Python smbus) | https://github.com/ControlEverythingCommunity/PCA9685/blob/master/Python/PCA9685.py | Repo | 1 |
| 11 | Waveshare Pan-Tilt-HAT — PCA9685.py | https://github.com/waveshare/Pan-Tilt-HAT | Repo | 1 |
| 12 | Freenove Robot Dog — PCA9685.py | https://github.com/Freenove/Freenove_Robot_Dog_Kit_for_Raspberry_Pi | Repo | 1 |
| 13 | bochinski/iou-tracker (MIT, 68 LOC) | https://github.com/bochinski/iou-tracker | Repo | 1 |
| 14 | nwojke/deep_sort — kalman_filter.py (Kalman NumPy) | https://github.com/nwojke/deep_sort | Repo | 1 |
| 15 | Bewley et al. — SORT (ICIP 2016) | https://arxiv.org/abs/1602.00763 | Paper | 1 |
| 16 | Bertozzi, Broggi, Fascioli — Stereo IPM (1998) | https://doi.org/10.1016/S0262-8856(97)00093-0 | Paper | 1 |
| 17 | Szulc, Iwanowski — Homography ground-plane distance error | https://arxiv.org/abs/2604.10805 | Paper | 1 |
| 18 | Kujala et al. (ZenRobotics) — Picking a Conveyor Clean | https://arxiv.org/abs/1511.07608 | Paper | 1 |
| 19 | Zocco et al. — Synchronized Object Detection (IEEE TIM 2024) | https://arxiv.org/abs/2405.06821 | Paper | 1 |
| 20 | Casao et al. — SpectralWaste (IROS 2024) | https://arxiv.org/abs/2403.18033 | Paper | 1 |
| 21 | Bacchin et al. — WasteGAN (IROS 2024) | https://arxiv.org/abs/2409.16999 | Paper | 1 |
| 22 | Wofk et al. — FastDepth (ICRA 2019) | https://arxiv.org/abs/1903.03273 | Paper | 1 |
| 23 | Feng et al. — RT-MonoDepth | https://arxiv.org/abs/2308.10569 | Paper | 1 |
| 24 | Yang et al. — Depth Anything V2 (NeurIPS 2024) | https://arxiv.org/abs/2406.09414 | Paper | 1 |
| 25 | Swaminathan et al. — Benchmarking DL on Jetson Nano | https://arxiv.org/abs/2406.17749 | Paper | 1 |
| 26 | fedezocco/2MMUsMed (código Zocco) | https://github.com/fedezocco/2MMUsMed | Repo | 1 |
| 27 | Roboflow — Track & Count YOLOv8 + ByteTrack | https://youtu.be/OS5qI9YBkfk | Video | 1 |
| 28 | Pysource — Objects on a conveyor belt (OpenCV) | https://youtu.be/A29IqeahI84 | Video | 1 |
| 29 | Visually Explained — Kalman Filters | https://youtu.be/IFeCIbljreY | Video | 1 |
| 30 | DataMListic — Hungarian Matching Algorithm | https://youtu.be/oo-H_ZY2TGA | Video | 1 |
| 31 | Kevin Wood — ByteTrack explained | https://youtu.be/6LGpf-a1K1Q | Video | 1 |
| 32 | OMES — Transformación de perspectiva (OpenCV, ES) | https://youtu.be/lkwtwhWKQjo | Video | 1 |
| 33 | MIT.nano — FastDepth (charla) | https://youtu.be/7EdFpgcD8vk | Video | 1 |
