# Cámara local en el server de inferencia (Jetson Nano)

Investigación de fundamentos para añadir un modo "cámara local" al server: el
Nano captura de su propia C920 en `/dev/video0`, infiere con el engine TRT y
empuja el frame anotado + detecciones al dashboard. Hoy la única fuente es la
cámara "remota" (el navegador captura con `getUserMedia` y manda JPEG por WS).

## Contexto

**Objetivo.** El dashboard ofrece un selector remota/local. En local, la cámara
física junto a la banda y los servos es la fuente, que es el caso de uso real
del clasificador de residuos (glass/paper/plastic).

**Entorno verificado por SSH (2026-05-25).**

| Componente | Valor |
|---|---|
| Plataforma | Jetson Nano B01 4 GB, JetPack 4.6.1, L4T R32.7.1, Maxwell sm_53 |
| Python | 3.6.9 |
| OpenCV | 4.1.1 con GStreamer 1.14.5, **sin backend V4L2** (captura solo por GStreamer) |
| GStreamer | 1.14.5; plugins presentes: `v4l2src`, `nvv4l2decoder`, `nvjpegdec`, `nvvidconv`, `jpegdec`, `videoconvert`, `nvarguscamerasrc` |
| PyGObject (`gi`) | 3.26.1 (Gst 1.14.5) |
| TensorRT | 8.2.1.8; pycuda 2019.1.2; numpy 1.13.3 |
| Server web | FastAPI 0.65.3, Starlette 0.14.2, uvicorn 0.13.4, websockets 8.1 |
| Cámara | Logitech C920 OG (UVC USB `046d:082d`) en `/dev/video0`; MJPG/H264/YUYV hasta 1080p |

**Permisos resueltos:** `jetson` pertenece al grupo `video(44)` y el server
systemd corre como `User=jetson`, así que el proceso puede abrir `/dev/video0`
sin sudo. La cámara está libre (hoy el server no la toca).

**Arquitectura actual del server (`scripts/server/nano_server.py`).** Un único
`TRTWorker(threading.Thread)` dueño exclusivo del contexto CUDA (push/pop),
`queue.Queue(maxsize=2)`, resultados por `asyncio` future vía
`loop.call_soon_threadsafe`. Flujo **pull**: el WS recibe JPEG binario →
`worker.submit((jpeg, ts, seq, loop, future))` → `cv2.imdecode` → letterbox a
416×416 → `execute_async_v2` → `cv2.dnn.NMSBoxes` (ruta V0 sm_53) → JSON bboxes.
`IMGSZ=416`, clases glass/paper/plastic, conf 0.25, NMS 0.45.

**Reconocimiento del Nano:** el proyecto en el Nano **no es repo git** (deploy
por copia, no `git pull`); **no existe `scripts/dashboard/`** (el dashboard vive
en `web/`); **no hay ningún código de cámara** en `scripts/` (módulo desde cero).

---

## Ronda 1 — 2026-05-25 (medio)

### Track A — Agentes de research

#### A1. Pipeline GStreamer C920 MJPG → BGR (pregunta 1)

La C920 es UVC, va **obligatoriamente por `v4l2src`** (nunca `nvarguscamerasrc`,
que es exclusivo de CSI/MIPI). Tres variantes viables:

**Variante A — decode HW `nvv4l2decoder mjpeg=1` (recomendada).** Usa el motor
VIC/NVDEC, físicamente distinto de los CUDA cores de TRT, así que coexisten en
paralelo real. Resize a 416 dentro de `nvvidconv` (en el VIC, sin costo CPU):

```
v4l2src device=/dev/video0 io-mode=2 !
  image/jpeg, width=640, height=480, framerate=15/1 !
  nvv4l2decoder mjpeg=1 !
  nvvidconv ! video/x-raw, width=416, height=416, format=BGRx !
  videoconvert ! video/x-raw, format=BGR !
  appsink drop=1 max-buffers=1
```

**Variante B — decode CPU `jpegdec` (fallback robusto, sin pitfalls).** No
requiere `io-mode=2`. ~15 ms a 720p; a 640×480 cuesta ~10 % de un core A57.

```
v4l2src device=/dev/video0 !
  image/jpeg, width=640, height=480, framerate=15/1 !
  jpegdec ! videoconvert ! video/x-raw, format=BGR !
  appsink drop=1 max-buffers=1
```

**Variante C — YUYV nativo (sin decode).** Solo viable a ≤480p@30fps por ancho
de banda USB; útil como último recurso.

| Variante | Decoder | Latencia decode | CPU | NVMM | Coexistencia TRT | Confiabilidad JP 4.6.1 |
|---|---|---|---|---|---|---|
| A `nvv4l2decoder mjpeg=1` | VIC/NVDEC HW | ~4 ms | Bajo | Sí | Alta (unidades distintas) | Buena con `io-mode=2` |
| B `jpegdec` | libjpeg-turbo SW | ~15 ms @720p | Medio | No | Perfecta | Excelente, sin pitfalls |
| C YUYV nativo | sin decode | 0 ms | Bajo | No | Perfecta | Solo ≤480p@30fps |

**Decisión:** Variante A a 640×480@15 con resize VIC a 416; fallback automático
a B si `VideoCapture` no abre. La diferencia de latencia es marginal frente a los
60-100 ms de inferencia TRT.

#### A2. Integración captura + worker GPU async (pregunta 2)

El blueprint `hasantavision/jetson-security-cam` (JetPack R32.7.1, Py3.6,
FastAPI + GStreamer por `gi` + worker YOLO en thread) es CSI, pero su
**arquitectura de integración es directamente reutilizable**:

- **Captura en thread dedicado, no callback.** Con `emit-signals=True` el
  callback GLib compite por el GIL y, bajo carga uvicorn + WS, solo llega cada
  ~25 s. Como nosotros usamos `cv2.VideoCapture` (no `gi` crudo), un thread con
  `cap.read()` bloqueante en su propio hilo evita el problema de raíz.
- **Cola maxsize=1 con `put_nowait` + drop.** Desacopla la cámara de la latencia
  de inferencia sin acumular backlog; el worker procesa siempre el frame más
  reciente.
- **El worker bifurca por tipo del item:** `bytes` → modo remoto actual
  (`imdecode`); `np.ndarray` → modo local (frame ya decodificado, salta
  `imdecode`). Integración mínima, preserva intacto el modo remoto.
- **Cierre: `cap.release()` + pausa ~1 s** para que `v4l2src` libere el device;
  sin esa pausa, un restart inmediato falla con "device busy".

Esquema del thread de captura (sin entrega de resultado; eso lo maneja el
worker hacia los clientes WS):

```python
def _cam_reader_thread(inference_queue):
    cap = cv2.VideoCapture(_pipeline_hw(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(_pipeline_cpu(), cv2.CAP_GSTREAMER)  # fallback B
    while _cam_running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01); continue
        try:
            inference_queue.put_nowait(frame)   # np.ndarray BGR
        except queue.Full:
            pass
    cap.release()
```

#### A3. Transporte Nano → browser (pregunta 3)

| Criterio | MJPEG `multipart/x-mixed-replace` | WebSocket binario (`send_bytes`) |
|---|---|---|
| Server | `StreamingResponse` + generador async | `ws.send_bytes(jpeg)` |
| Browser | `<img src>` nativo | JS explícito |
| Sincronización overlay | Difícil (img a su propio ritmo) | Control total (dibuja al llegar el frame) |
| Backpressure | Ninguno | Full-duplex |
| Sobre Headscale | TCP/HTTP, sin config | TCP, igual |

**Decisión: WebSocket binario.** Encaja con FastAPI async y permite mandar el
frame JPEG + sus bboxes en mensajes contiguos, sincronizando overlay y video sin
esfuerzo. Ancho de banda a 14 fps con JPEG 416 q80: **~350-700 KB/s**, trivial
sobre Headscale. **Fallback de demo:** endpoint MJPEG con `<img src>`, por si
algo falla minutos antes de la sustentación.

Consumo en el browser sin bloquear el hilo principal: `createImageBitmap` en un
**Web Worker** + canvas con contexto `bitmaprenderer` (`transferFromImageBitmap`,
zero-copy). Para Chrome (cliente único de la demo) basta `createImageBitmap(blob)`
+ `drawImage` + `bitmap.close()`.

Para mantener el overlay del dashboard y minimizar CPU en el Nano, el Nano manda
el **frame limpio + bboxes JSON** y el browser dibuja con `drawDetections` ya
existente (consistente con el modo remoto), en vez de quemar las cajas en el JPEG.

#### A4. Footprint RAM/GPU en 4 GB (pregunta 4)

Presupuesto holgado para un solo stream: engine TRT YOLOv8n FP16 ~336-385 MB +
buffers del decoder VIC ~10-30 MB (hasta ~150-200 MB a 1080p) + contexto CUDA
~50-100 MB. Riesgo de OOM bajo. Mitigaciones:

- `appsink drop=1 max-buffers=1` (evita que el buffer crezca si la inferencia es
  más lenta que la cámara; causa #1 de crecimiento de memoria).
- `queue ... max-size-buffers=1 leaky=2` (descarta downstream ante jitter USB).
- Capturar en baja resolución (640×480), resize en VIC.
- `jetson_clocks` al entrar en modo local (el VIC en ahorro de energía añade
  latencia).
- `tegrastats --interval 500` para vigilar `RAM`, `EMC`, `GR3D%`, `VIC`.
- El zero-copy NVMM→CUDA no está expuesto limpio en pycuda 2019; la copia
  CPU→GPU (~0,5-1 ms a 416) es el camino pragmático, igual que hoy.

### Track B — Búsqueda ampliada

`discover.py` (medio) devolvió 18 fuentes sobre `developer.nvidia.com`,
`forums.developer.nvidia.com`, `github.com`, `docs.opencv.org`,
`stackoverflow.com`. Lectura activa (Exa) de las tres más densas, que
confirmaron el pipeline canónico y los gotchas de caps:

- **SO #75469492** (respuesta aceptada de SeB): `nvvidconv` no emite BGR, solo
  BGRx; falta la coma en caps; `videoconvert` solo trabaja en memoria de sistema.
  Pipeline correcto idéntico al de la Variante A.
- **SO #65638140** (Marcel Kopera, OpenCV 4.5.1 en Nano): `io-mode=2` resuelve el
  clásico "outputting YUYV not MJPG"; comentario de Hung Le: si `nvjpegdec` da
  problemas, reemplazar por `jpegdec`.
- **Foro NVIDIA #145421** (con OpenCV 4.1.1, nuestra versión exacta): hilo donde
  DaneLLL (moderador) valida el pipeline con `io-mode=2` y nota la limitación del
  VIC (BGR cuesta CPU vía `videoconvert`; I420 directo llega a 30fps pero BGR baja
  a ~20fps). Verificado con Logitech C615/C930 (familia de la C920).

---

## Pitfalls catalogados (checklist de implementación)

1. **`io-mode=2` obligatorio** para forzar MJPG en `v4l2src` (sin él, negocia
   YUYV y se hunde el FPS). Si falla en el kernel, probar `io-mode=4` (USERPTR).
2. **`nvvidconv` no emite BGR, solo BGRx** → cadena
   `nvvidconv ! ...BGRx ! videoconvert ! ...BGR`. Poner `format=BGR` en
   `nvvidconv` rompe el pipeline ("could not parse caps").
3. **`nvjpegdec` sin `nvvidconv` intermedio → frames negros/verdes**; además
   tiene más casos límite con webcams USB. Preferir `nvv4l2decoder mjpeg=1`.
4. **Altura alineada a múltiplo de 16:** `nvv4l2decoder` convierte 1080→1088
   (720 queda igual). `cap.get(HEIGHT)` devuelve 1088; recortar `frame[:1080]` si
   se captura a 1080p. Capturar a 640×480 evita el tema.
5. **`cap.release()` puede colgar sin `drop=1`** en el appsink.
6. **Callback `emit-signals` llega cada ~25 s bajo asyncio** (contención GIL);
   usar `cap.read()` en thread dedicado, no callbacks GLib.
7. **Cerrar con `release()` + pausa ~1 s** para liberar `/dev/video0` antes de
   reabrir (device busy).
8. **Probar siempre con `gst-launch-1.0` antes que con cv2** (cv2 oculta el error
   real de GStreamer); si "Internal data stream error", `fuser -k /dev/video0`.
9. Añadir `format=MJPG` al caps `image/jpeg` si la negociación falla sin él.

---

## Decisiones derivadas (insumo para el plan)

1. **Captura:** Variante A (`nvv4l2decoder mjpeg=1`, 640×480@15, resize VIC a
   416), con fallback automático a Variante B (`jpegdec`).
2. **Integración server:** thread `cam-reader` daemon → `worker.in_q.put_nowait`
   con frame `np.ndarray`; el worker bifurca `bytes` vs `ndarray`. El modo remoto
   queda intacto.
3. **Transporte:** WebSocket binario que manda frame JPEG limpio + bboxes JSON;
   el dashboard reutiliza `drawDetections`. Endpoint MJPEG `<img>` como fallback
   de demo.
4. **Dashboard:** toggle remota/local. En local no usa `getUserMedia`; abre el WS
   de streaming, pinta el frame del Nano y dibuja overlay; reutiliza el panel de
   métricas.
5. **Memoria/robustez:** `drop=1 max-buffers=1`, `queue leaky=2`, `jetson_clocks`
   al entrar en local, `tegrastats` para validar; liberar la cámara al salir.
6. **Concurrencia de modos:** definir en el plan qué pasa si un cliente está en
   remoto y otro pide local (la cámara local es un recurso único); para el MVP,
   un solo modo activo a la vez.

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|---|---|---|---|
| 1 | 2026-05-25 | medio | Pipeline GStreamer C920→BGR, integración captura+worker async, transporte Nano→browser, footprint 4 GB |

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|---|---|---|---|
| 1 | Accelerated GStreamer User Guide L4T R34.1 | https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/Multimedia/AcceleratedGstreamer.html | Doc NVIDIA | 1 |
| 2 | Foro: Gstreamer use MJPEG codec (OpenCV 4.1.1, io-mode=2) | https://forums.developer.nvidia.com/t/gstreamer-use-mjpeg-codec/145421 | Foro NVIDIA | 1 |
| 3 | Foro: GStreamer pipeline for accelerated streaming of USB camera | https://forums.developer.nvidia.com/t/gstreamer-pipeline-for-accelerated-streaming-of-usb-camera/233677 | Foro NVIDIA | 1 |
| 4 | Foro: nvvidconv no soporta BGR como salida (limitación VIC) | https://forums.developer.nvidia.com/t/why-jetson-gstreamer-plugin-nvvidconv-not-support-bgr-format-output/195994 | Foro NVIDIA | 1 |
| 5 | Foro: GPU vs CPU deep learning memory usage (TRT ~385 MB) | https://forums.developer.nvidia.com/t/gpu-vs-cpu-deep-learning-memory-usage/284630 | Foro NVIDIA | 1 |
| 6 | Foro: nvjpegdec does not work for USB camera | https://forums.developer.nvidia.com/t/nvjpegdec-does-not-work-for-usb-camera/280767 | Foro NVIDIA | 1 |
| 7 | SO: nvidia component to accelerate USB camera (pipeline correcto) | https://stackoverflow.com/questions/75469492 | StackOverflow | 1 |
| 8 | SO: Create Pipeline GStreamer USB Camera MJPG (io-mode=2) | https://stackoverflow.com/questions/65638140 | StackOverflow | 1 |
| 9 | SO: streaming VideoCapture frames con GStreamer + FastAPI | https://stackoverflow.com/questions/71816725 | StackOverflow | 1 |
| 10 | hasantavision/jetson-security-cam (blueprint integración) | https://github.com/hasantavision/jetson-security-cam | Repo | 1 |
| 11 | JetsonHacksNano/USB-Camera (C920 en JetPack 4.6.1) | https://github.com/JetsonHacksNano/USB-Camera | Repo | 1 |
| 12 | jetsonhacks/USB-Camera | https://github.com/jetsonhacks/USB-Camera | Repo | 1 |
| 13 | thehapyone/NanoCamera (pipelines USB/MJPEG/CSI) | https://github.com/thehapyone/NanoCamera | Repo | 1 |
| 14 | jkjung-avt/tensorrt_demos utils/camera.py | https://github.com/jkjung-avt/tensorrt_demos/blob/master/utils/camera.py | Repo | 1 |
| 15 | mad4ms/python-opencv-gstreamer-examples | https://github.com/mad4ms/python-opencv-gstreamer-examples | Repo | 1 |
| 16 | bdtinc/maskcam (Nano + GStreamer + streaming web) | https://github.com/bdtinc/maskcam | Repo | 1 |
| 17 | dusty-nv/jetson-inference issue #532 (C920 en Nano) | https://github.com/dusty-nv/jetson-inference/issues/532 | Issue | 1 |
| 18 | web.dev: OffscreenCanvas | https://web.dev/articles/offscreen-canvas | Doc | 1 |
| 19 | Web Performance Calendar: non-blocking canvas rendering | https://calendar.perfplanet.com/2025/non-blocking-cross-browser-image-rendering-on-the-canvas | Blog | 1 |
| 20 | Starlette docs: Responses (StreamingResponse) | https://www.starlette.io/responses/ | Doc | 1 |
| 21 | arxiv: Profiling Concurrent Vision Inference on Jetson | https://arxiv.org/html/2508.08430v1 | Paper | 1 |
