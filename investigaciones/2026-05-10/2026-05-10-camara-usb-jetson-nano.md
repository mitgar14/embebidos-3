# Selección de cámara USB — embebidos-3

> **Proyecto:** MVP clasificador de residuos (`embebidos-3`)
> **Decisión arquitectónica anclada:** Track B confirmado — YOLOv8n 416×416 FP16 TensorRT sobre GPU Maxwell del Jetson Nano Developer Kit 4 GB B01, JetPack 4.6.x.
> **Fecha investigación:** 2026-05-10 (ronda 1, profundidad media).
> **Pregunta:** ¿qué cámara USB UVC montar en diagonal sobre la banda transportadora (30–50 cm) para alimentar el pipeline GStreamer → nvv4l2decoder/nvjpegdec → CUDA → TRT?

---

## Recomendación priorizada (TL;DR)

| # | Cámara | Por qué | Precio Colombia ref. | Caveats |
|---|--------|---------|----------------------|---------|
| **1** | **Logitech C920 / C920s Pro** | FOV diagonal 78° (cubre banda con holgura desde 30–50 cm), MJPG 720p@30 y 1080p@30 vía UVC, autofocus deshabilitable por v4l2-ctl, **cámara más documentada para Jetson Nano en proyectos similares**. Caso de uso idéntico (banda + TensorRT + servos) reportado en Hackster.io 2024. | $322.500 (ML usado) – $359.900 (Logitech Store) – $380.000 COP | El sensor es muy bueno → autofocus puede "cazar" si no se fija. Fijarlo con `v4l2-ctl -c focus_automatic_continuous=0 -c focus_absolute=125` **antes** de lanzar GStreamer. |
| **2** | **Logitech C270** | Foco fijo nativo (cero autofocus hunt), MJPG 720p@30 validado en JetPack 4.6.1, **opción más barata y de disponibilidad inmediata en Colombia**. Mencionada en el caso Hackster como alternativa equivalente. | **$95.000 – $130.000 COP** (Logitech Store oficial $99.900 con descuento) | FOV diagonal 55° → más cerrado. A 40 cm de altura cubre ~35 cm de banda; si la banda es más ancha hay que subir la cámara a ~55–60 cm e introduce distorsión por perspectiva. |
| **3** | **ELP USB OV5640 foco fijo 75°** (industrial low-cost) | UVC compliant, MJPG 720p@30, foco fijo permanente (ideal para banda), FOV seleccionable al pedido. Mejor relación calidad/precio si se puede esperar importación. | ~$75.000–$130.000 COP en AliExpress + 19% IVA + arancel (total final puede igualar a C270 local). | 2–4 semanas de espera, sin garantía local, verificar el modelo exacto: el `ELP-USBFHD01M-SFV` específico tuvo issues con GStreamer en JetPack en 2020 (Developer Forums). Comprar con el sensor OV5640 y pedir explícitamente al vendedor "MJPEG 720p@30 UVC Linux compatible". |

**Decisión sugerida para el MVP académico:** si el presupuesto lo permite, **Logitech C920** (top 1) por la combinación de FOV adecuado + masa crítica de documentación + sensor de vidrio con mejor tolerancia a iluminación variable (que es exactamente uno de los augmentations declarados en el README, Brightness ±25% / Exposure ±15%). Si el presupuesto es restrictivo, la **C270 (top 2)** sigue siendo viable y está literalmente listada en el caso Hackster equivalente.

**Descartar:** Logitech Brio 100 (FOV 58° peor que C270 a mayor precio), Logitech StreamCam (USB-C + sin evidencia verificada en Nano), Logitech C922 (mismo hardware que C920 a precio mayor sin ventaja para 30 fps), Arducam B044601 wide 101° HDR (fixed focus 300 cm–∞, **no enfoca a 30–50 cm**, descalificada para banda corta).

---

## 1. Restricciones del proyecto que condicionan la elección

1. **JetPack 4.6.x + L4T R32.7.x** — la stack está congelada. El driver `uvcvideo` del kernel L4T y la cadena GStreamer disponible (gst-v4l2, nvv4l2decoder, nvjpegdec, nvvidconv) son los que hay. No hay JetPack 5.x ni `nvbufsurface` zero-copy moderno (eso llegó con JetPack 5; ver issue [`dusty-nv/jetson-utils#204`](https://github.com/dusty-nv/jetson-utils/issues/204)).
2. **GPU Maxwell + USB compartido** — los 4 puertos USB 3.0 del Nano B01 cuelgan de un **único hub interno** (referencia: JetsonHacks "In Practice: USB Cameras on Jetson", 2022). El ancho de banda efectivo se reparte entre puertos.
3. **Montaje diagonal sobre banda corta a 30–50 cm** — descalifica cámaras con foco fijo lejano (≥ 1 m) y favorece autofocus *fijable* o foco fijo cercano.
4. **Iluminación variable interior** — exposure y gain controlables vía `v4l2-ctl` son requisito real.
5. **Target captura 720p@30 fps** — el modelo se ejecuta a 416×416 (Track B confirmado), así que la cámara solo alimenta el preprocessing. **No tiene sentido pagar por 4K**.
6. **Servos SG90 + PCA9685 vía I2C concurrente** — la captura no puede saturar el bus USB ni la CPU porque el hilo de servos lo necesita.

---

## 2. Hallazgo crítico: MJPG es **eliminatorio**, no opcional

| Formato | 720p@30 bitrate sin comprimir | 1080p@30 bitrate sin comprimir | USB 2.0 efectivo |
|---------|-------------------------------|--------------------------------|-------------------|
| YUYV (4:2:2) | **442 Mbps** (55 MB/s) | 995 Mbps (124 MB/s) | ~480 Mbps teórico, ~40 MB/s real |
| MJPG (~10:1) | **~44 Mbps** (5.5 MB/s) | ~100 Mbps (12.4 MB/s) | igual |

**Conclusión:** YUYV a 720p@30 **no cabe en USB 2.0** y el driver `uvcvideo` del kernel L4T reserva ancho de banda según el peor caso del intervalo isócrono → bloquea el bus aunque el throughput real sea menor. Hay reportes específicos en NVIDIA Developer Forums ("Two 1080p USB cameras on Nano", `VIDIOC_STREAMON: No space left on device`).

→ **Cualquier cámara que no exponga MJPG a 720p@30 en su descriptor UVC queda descalificada para este proyecto.** Verificación obligatoria antes de comprar: pedir al vendedor la salida de `v4l2-ctl --list-formats-ext` o el datasheet con la columna explícita "MJPG 1280×720 30fps".

Las 5 candidatas top de la sección 1 **sí cumplen** (verificado vía datasheets oficiales o issues GitHub).

---

## 3. Tabla comparativa de candidatas

| Cámara | Precio CO (COP) | FOV diag. | Resol./FPS MJPG | Foco | USB | Consumo | Disponibilidad CO | Evidencia Jetson Nano |
|--------|------------------|-----------|-----------------|------|-----|---------|--------------------|------------------------|
| **Logitech C920 / C920s** | $322k–$380k | 78° (H 70°, V 43°) | 1080p@30, 720p@30 | Auto (fijable v4l2) | 2.0 | ~500 mA | Alta (Falabella, ML, Logitech Store) | `dusty-nv/jetson-inference#532` cerrado OK, [`jetsonhacks/USB-Camera`](https://github.com/jetsonhacks/USB-Camera) lo testea explícitamente en JetPack 4.6.1 / L4T 32.6.1 |
| **Logitech C270** | **$95k–$130k** | 55–60° | 720p@30 | **Fijo nativo** | 2.0 | ~200 mA | Alta (Logitech Store oficial $99.900, Speed Logic Bogotá 20 disp.) | [`dusty-nv/jetson-inference#1639`](https://github.com/dusty-nv/jetson-inference/issues/1639) (JetPack 4.6.1), SpyJetson blog (JetPack 4.3), oficialmente listada en docs jetson-inference |
| **Logitech C310** | $120k–$160k | 60° | 720p@30 | Fijo nativo | 2.0 | ~200 mA | Media-alta (ML) | gstreamer-devel mailing list, blog EnterBox (Jetson Nano self-driving car) |
| **Logitech C922 Pro** | $410k–$450k | 78° | 1080p@30, **720p@60** | Auto (fijable v4l2) | 2.0 | ~500 mA | Media | Mismo hardware que C920 → misma evidencia |
| **Logitech C615 / C930** | difícil de conseguir nuevo | 78° | 1080p@30 | Auto | 2.0 | ~500 mA | Baja | Pipeline `v4l2src io-mode=2 ! image/jpeg ! nvjpegdec` **verificado por NVIDIA moderador DaneLLL** ([NVIDIA Forums #145421](https://forums.developer.nvidia.com/t/gstreamer-use-mjpeg-codec/145421)) |
| **ELP OV5640 75° fijo** | $75k–$130k + import. | seleccionable 60°/75°/100° | 720p@30, 1080p@30 | **Fijo permanente** | 2.0 | ~300 mA | Baja (AliExpress 2–4 sem) | NVIDIA Forums "Two 1080p USB cameras", `avdec_mjpeg` documentado, Amazon "Jetson Nano compatible" |
| **Arducam IMX219 USB B0196** | ~$130k–$170k + import. | 75°(D)/60°(H) | 720p@30, 1080p@30 | Fijo 40 mm – ∞ | 2.0 | 200 mA max | Muy baja (Amazon/Arducam) | Datasheet oficial Arducam ([blog.arducam.com](https://blog.arducam.com/jetson-nano-xavier-nx-camera-solutions)), pero NO testeada explícitamente con jetson-utils en issues conocidos |
| **Arducam IMX219 USB B0292** (autofocus, metal case) | ~$200k–$250k + import. | 72°(D) | 720p@30, 1080p@30 | Auto 10 cm – ∞ | 2.0 | 200 mA | Muy baja | igual |
| **Arducam B044601 HDR 101°** | n/a | 101° | 1080p@30 | Fijo **300 cm – ∞** | 2.0 | 1.1 W | – | ❌ **DESCARTADA**: foco fijo a 3 m no enfoca banda corta |
| **Logitech Brio 100** | $209.900 (Logitech Store) | 58° | 1080p@30 | Fijo | 2.0/3.0 | ~500 mA | Alta | ❌ **DESCARTADA**: FOV peor que C270 a mayor precio |
| **Logitech StreamCam** | n/a | 78° | 1080p@60 | Auto | USB-C | – | Media | ❌ **DESCARTADA**: USB-C + sin evidencia documentada en Jetson Nano JetPack 4.x |

**Precios verificados el 2026-05-10** en Logitech Store Colombia, MercadoLibre Colombia, Capital Colombia (Bogotá), Speed Logic, Tecno Shopping, Falabella.

---

## 4. Caso de uso real que valida la elección

**`kinetika` en Hackster.io (febrero 2024)** publicó *"Counting for Inspection and Quality Control with TensorRT"*: Jetson Nano + **Logitech C270 o C920** + banda transportadora + TensorRT (Edge Impulse / MobileNet V2 320×320) + LEDs y servos para aceptar/rechazar. **5 ms por frame de inferencia.** El stack es prácticamente isomorfo al de `embebidos-3` excepto por el modelo (ellos usan MobileNet; nosotros YOLOv8n).

URL: <https://www.hackster.io/kinetika/counting-for-inspection-and-quality-control-with-tensorrt-550b91>

Otro proyecto convergente: **`imnuman/jetson-object-detection`** (GitHub, ene 2026) reporta **YOLOv8n TensorRT a 28 FPS @ 640×640 en Nano 4 GB** con cámara **Logitech C920 vía V4L2**. URL: <https://github.com/imnuman/jetson-object-detection>

→ Estos dos proyectos publicados son la mejor validación posible: **la elección entre C270 y C920 está empíricamente probada para tu caso de uso**.

---

## 5. Pipeline GStreamer canónico para la cámara seleccionada

### 5.1 Verificación previa a comprar / al recibir la cámara

```bash
# En el Jetson Nano, una vez conectada la cámara:
v4l2-ctl --list-devices
v4l2-ctl --list-formats-ext -d /dev/video0
```

Buscar línea explícita `Pixel Format: 'MJPG' (compressed)` con `Size: Discrete 1280x720` e `Interval: Discrete 0.033s (30.000 fps)`. Si esto no aparece, la cámara queda fuera.

### 5.2 Boost del Nano antes de lanzar el pipeline

```bash
sudo nvpmodel -m 0          # max power mode
sudo jetson_clocks          # clocks al máximo
# Desactivar USB autosuspend (evita que la cámara "se duerma"):
sudo sh -c "echo -1 > /sys/module/usbcore/parameters/autosuspend"
```

### 5.3 Pipeline recomendado para integrar con YOLOv8n FP16 TensorRT

```bash
v4l2src device=/dev/video0 io-mode=2
  ! image/jpeg, width=1280, height=720, framerate=30/1
  ! nvv4l2decoder mjpeg=1
  ! nvvidconv
  ! 'video/x-raw(memory:NVMM), format=NV12, width=416, height=416'
  ! appsink max-buffers=1 drop=true sync=false
```

**Notas:**
- `io-mode=2` (mmap) es **obligatorio** en `v4l2src` para que el pipeline no falle con *Internal data stream error*. Confirmado por NVIDIA moderador DaneLLL.
- `nvv4l2decoder mjpeg=1` usa el motor VIC (hardware) — máximo rendimiento.
- `nvvidconv` hace resize 720p → 416×416 y conversión a NV12 **en hardware (motor VIC)**; salida en `memory:NVMM` accesible desde CUDA con `cudaGraphicsEGLRegisterImage`.
- `max-buffers=1 drop=true sync=false` en `appsink` → política drop-oldest, alineada con la decisión arquitectónica del proyecto (ver `2026-05-05-arquitectura-software-jetson-nano.md` sec. producer-consumer + queue drop-oldest).

### 5.4 Fallback si `nvv4l2decoder mjpeg=1` falla

**Hallazgo crítico documentado:** `dusty-nv/jetson-utils` desactiva por default el HW NVDEC MJPEG porque *"was bugging out on some cameras"* (issue [`#66`](https://github.com/dusty-nv/jetson-utils/issues/66), comentario de @dusty-nv). Si tu cámara da problemas con `nvv4l2decoder mjpeg=1`, **cambiar a `nvjpegdec`**:

```bash
v4l2src device=/dev/video0 io-mode=2
  ! image/jpeg, width=1280, height=720, framerate=30/1
  ! nvjpegdec          # SOFT-GPU JPEG, sin VIC, validado en cámaras C615/C930
  ! video/x-raw
  ! nvvidconv
  ! 'video/x-raw(memory:NVMM), format=NV12, width=416, height=416'
  ! appsink max-buffers=1 drop=true sync=false
```

**Último recurso (NO recomendado, solo para debugging):** `jpegdec` (CPU puro) — consume 90–98% de CPU, queda < 20 FPS reales y deja los servos sin medio núcleo. Usar solo para validar conectividad básica.

### 5.5 Configuración v4l2 obligatoria si la cámara tiene autofocus (C920 / C922)

```bash
# Antes de lanzar GStreamer:
v4l2-ctl -d /dev/video0 -c focus_automatic_continuous=0   # firmware moderno C920
v4l2-ctl -d /dev/video0 -c focus_absolute=125              # calibrar empíricamente
v4l2-ctl -d /dev/video0 -c exposure_auto=1                 # 1=manual en UVC
v4l2-ctl -d /dev/video0 -c exposure_absolute=300           # calibrar
```

El valor antiguo `focus_auto=0` ya no funciona en firmwares C920 recientes ([Mainsail issue #96](https://github.com/mainsail-crew/crowsnest/issues/96)). Si `v4l2-ctl -L` no muestra `focus_automatic_continuous`, fallback al nombre viejo.

---

## 6. Issues conocidos y cómo evitarlos

| Issue | Síntoma | Mitigación |
|-------|---------|------------|
| **USB autosuspend** | Cámara funciona X minutos y se cuelga | `echo -1 > /sys/module/usbcore/parameters/autosuspend` |
| **`nvv4l2decoder mjpeg=1` bug** | Errores intermitentes de decode, colores raros, líneas verdes | Fallback a `nvjpegdec` (sec. 5.4) |
| **2 cámaras YUYV simultáneas** | `VIDIOC_STREAMON: No space left on device` | Solo cargar 1 cámara, usar MJPG, `modprobe uvcvideo quirks=128` si imprescindible |
| **OpenCV `cv2.imshow` lento** | FPS reportado correcto pero la ventana lagea | NO usar imshow en producción; solo medir con `fpsdisplaysink video-sink=fakesink` |
| **Autofocus hunt** (C920/C922) | Foco oscila cuando el objeto se mueve | Fijar foco con `focus_automatic_continuous=0 focus_absolute=<N>` (sec. 5.5) |
| **`tflite.Interpreter` no thread-safe** (Track A descartado) | n/a | Ya no aplica — Track B confirmado |
| **OpenCV no acepta `(memory:NVMM) format=BGR`** | Pipeline falla negociando caps | Cadena `nvvidconv → BGRx → videoconvert → BGR` |

---

## 7. Acciones siguientes (validación empírica)

1. **Comprar Logitech C920 (top 1) o C270 (top 2)** según presupuesto. Si se elige C270, considerar montar la cámara a 55–60 cm de altura para cubrir el ancho de banda.
2. **Antes de integrar**: ejecutar `v4l2-ctl --list-formats-ext -d /dev/video0` y confirmar línea MJPG 1280×720 @ 30 fps.
3. **Bench standalone** (sin modelo): correr el pipeline canónico de la sec. 5.3 con `fpsdisplaysink` y confirmar 30 fps sostenidos durante 60 s.
4. **Bench integrado**: integrar al harness `scripts/bench_jetson.py` y medir end-to-end (captura → preproc → TRT inferencia → postproc) contra el threshold MVP ≥ 10 FPS sostenidos con servos+I2C concurrentes.
5. **Calibrar exposición y foco** para la iluminación específica del laboratorio (sec. 5.5). Repetir captura representativa de 400 muestras del val split para sanity check del modelo (ver `2026-05-05-preprocessing-roboflow.md` sec. repr dataset).

---

## 8. Cálculo de cobertura de banda según FOV (para decidir altura de montaje)

Asumiendo cámara perpendicular sobre el centro de la banda (la diagonal real reduce ligeramente la cobertura):

| Cámara | FOV H | A 30 cm altura → ancho cubierto | A 40 cm | A 50 cm |
|--------|-------|----------------------------------|---------|---------|
| C920 / C922 | 70° | 42 cm | **56 cm** | 70 cm |
| C270 | ~48° | 27 cm | 36 cm | 45 cm |
| ELP OV5640 75° | ~63° | 37 cm | 49 cm | 61 cm |
| Arducam B0196 IMX219 | 60° | 35 cm | 46 cm | 58 cm |

Fórmula: `ancho = 2 × altura × tan(FOV_horizontal / 2)`.

**Implicación práctica:** si la banda del MVP tiene ancho ≈ 30–40 cm, la **C270 a 40 cm** ya cubre. Si la banda es de 50+ cm, **conviene la C920 o subir la C270 a 55 cm** (con la perspectiva diagonal asumida en el README, esto sigue funcionando).

---

## Track A — Agentes de research

### A1: `research-code` — Pipeline GStreamer / V4L2 / CUDA

Cubierto en sec. 2 (MJPG vs YUYV), sec. 5 (pipelines canónicos), sec. 6 (issues). Hallazgos clave:
- Stack de captura: UVC kernel module → V4L2 → GStreamer (sin paso por ISP Tegra como las CSI).
- 3 decodificadores disponibles: `nvjpegdec` (CUDA, software JPEG), `nvv4l2decoder mjpeg=1` (HW VIC, deshabilitado por default en jetson-utils por bug), `jpegdec/avdec_mjpeg` (CPU, no usar).
- Zero-copy real (DMA-BUF NvBuffer) **no disponible** para MJPEG en JetPack 4.x — requiere JetPack 5.x. Para 4.x, mitigación con `nvvidconv → memory:NVMM`.
- Repositorios canónicos: `dusty-nv/jetson-inference` (~9k ⭐), `dusty-nv/jetson-utils` (~881 ⭐), `jetsonhacks/USB-Camera` (~90 ⭐, JetPack 4.6.1 testeado).
- Cámaras explícitamente testeadas con jetson-utils: **Logitech C920** (issue #532 cerrado OK), Logitech C270 (issue #1639), C615/C930 (verificadas por NVIDIA moderator DaneLLL), Stereolabs ZED, Intel RealSense D435 (estas últimas no UVC, requieren SDK).

### A2: `research-web` — Reviews, casos de uso y precios Colombia

Cubierto en sec. 1 (recomendación), sec. 3 (tabla con precios), sec. 4 (caso Hackster). Hallazgos clave:
- **Caso de uso isomorfo:** `kinetika` Hackster 2024 (banda + Nano + C270/C920 + TensorRT + servos, 5 ms/frame).
- Logitech C270 mencionada oficialmente en docs `jetson-inference` (DeepWiki) como cámara testeada.
- Precios Colombia verificados 2026-05-10: C270 $99.900 (Logitech Store oficial), C920 $322.500 (ML), C920s $331.900 (Falabella), Brio 100 $209.900.
- Disponibilidad inmediata local: C270, C310, C920, C920s, Brio. Importación 2–4 semanas: ELP, Arducam UVC.
- Paper académico relevante: ICOSIET 2024 "Performance of YOLOv5 for Waste Classification" — confirma YOLO + edge device para clasificación de residuos como tema publicable IEEE.

---

## Track B — Búsqueda ampliada

### B.1 discover.py (4 queries Exa)

| Query | Resultados | URLs más valiosas extraídas |
|-------|-----------|-------------------------------|
| `USB camera GStreamer Jetson Nano MJPG nvv4l2decoder pipeline` | 21 | JetsonHacks "In Practice", `jetsonhacks/USB-Camera`, NVIDIA Forums "MJPEG codec", Stack Overflow "MJPG nvv4l2decoder" |
| `Logitech C920 C270 USB webcam Jetson Nano TensorRT YOLO benchmark FPS` | 18 | `dusty-nv/jetson-inference#532` (C920), thread C270 NVIDIA Forums, `Qengineering/YoloV8-TensorRT-Jetson_Nano`, `imnuman/jetson-object-detection`, jkjung-avt YOLOv4 |
| `Arducam ELP USB UVC camera Jetson Nano industrial vision computer` | 19 | Arducam B029201 autofocus, Arducam B0196 fixed focus, ELP modelos AliExpress, Amazon Jetson-tagged |
| `Logitech C920 C270 webcam precio Colombia Mercadolibre` | 11 | Logitech Store CO, Capital Colombia, Falabella, Speed Logic, Tecno Shopping, ML CO |

### B.2 MCP youtube (2 queries)

| Query | top_k | Video clave |
|-------|-------|-------------|
| `USB webcam Jetson Nano YOLO real time object detection` | 12 | **`rs4mQcJAjMM`** — JetsonHacks "USB Cameras - NVIDIA Jetson" (1:434, transcript completo extraído, chapters explícitos sobre `v4l2-ctl`, USB bandwidth, USB autosuspend, V4L2 vs GStreamer OpenCV interface) |
| `Logitech C920 C270 Jetson Nano camera setup tutorial` | 10 | Resultados poco relevantes (Eran Feit canal generalista) → query secundaria descartada |

Chapters relevantes de [`rs4mQcJAjMM`](https://youtu.be/rs4mQcJAjMM):
- [12:19 Note about USB bandwidth](https://youtu.be/rs4mQcJAjMM?t=739)
- [13:43 Note about USB auto suspend](https://youtu.be/rs4mQcJAjMM?t=823)
- [15:09 V4L2 OpenCV Interface](https://youtu.be/rs4mQcJAjMM?t=909)
- [18:04 GStreamer OpenCV Interface](https://youtu.be/rs4mQcJAjMM?t=1084)

### B.3 Lectura activa (crawling_exa + WebFetch)

11 URLs leídas en detalle (ver tabla de fuentes consultadas).

---

## Ronda 2 — 2026-05-11 (medio) — Revisiones internas del Logitech C920 OG y riesgos en unidades de logo antiguo + identificación C930e (imagen 3)

> **Contexto:** El usuario ya decidió comprar un Logitech C920 (top 1 de la ronda 1) y entregó tres imágenes para validar la unidad disponible. Dos imágenes corresponden a un **C920 OG con logotipo antiguo de Logitech y branding "Carl Zeiss"** sobre el lente; la tercera, analizada al cierre de esta ronda, **resultó ser un Logitech C930e (NO un C920)** — distinta unidad. Esta ronda investiga las revisiones internas silenciosas del C920 a lo largo de su vida comercial (2012–presente) y los riesgos específicos al configurar unidades antiguas en JetPack 4.6.x / L4T R32.7.x (kernel Linux 4.9.337).

### 9. Revisiones documentadas del Logitech C920 (2012–presente)

A diferencia del SKU minorista inmutable (`C920` / part number `960-000764`), Logitech ha **modificado silenciosamente el hardware interno** del C920 al menos tres veces sin cambiar el numeral comercial. La evidencia es pública en Hacker News, BATC Wiki, Stream Tech Reviews y commits del kernel Linux.

| Generación | Años aprox. | Identificadores visibles | Cambios internos | Linux PID (`lsusb`) |
|------------|-------------|---------------------------|-------------------|----------------------|
| **Rev 1 ("OG")** | 2012–2015 | Logo "ojo verde" antiguo de Logitech + marca **"Carl Zeiss"** impresa sobre el lente. Sticker inferior con **M/N: V-U0028**. | **Encoder H.264 hardware activo** (UVC 1.5 con stream mux H.264 en contenedor MJPG). ISP Pixart PAC7332. Autofocus reportadamente más confiable. | `046d:082d` (default) |
| **Rev 2 (transición "OrbiCam")** | 2016–2018 | Logo nuevo "Logi" wordmark **o** logo viejo según lote. **"Carl Zeiss" eliminado del lente.** M/N **V-U0060**. | ISP mantenido pero **encoder H.264 hardware retirado del firmware**. Reportes de respuesta cromática distinta. | `046d:0892` |
| **Rev 3 (moderna)** | 2019–presente | Logo "Logi" wordmark moderno. M/N **V-U0068** / **V-U0070** según lote. SKUs renombrados como **C920e** (corporate) o **C920s** (con shutter de privacidad). | Sin H.264 hardware. Control v4l2 renombrado: el legacy `focus_auto` ya no existe → solo responde a `focus_automatic_continuous`. | `046d:08e5` (algunas unidades) o `046d:0892` (heredado) |

**Fuentes principales para esta cronología:**

- Hacker News comentario `31276282` (2022, validación colectiva): *"My older Logitech C920 has an on-board H.264 encoder. Newer revisions of the same model does not"* — <https://news.ycombinator.com/item?id=31276282>
- Stream Tech Reviews "Logitech C920 vs C920x vs C920S vs C920e" — <https://www.streamtechreviews.com/blog/c920-variations>
- iFixit teardown C920 (M/N V-U0062 versión intermedia) — <https://www.ifixit.com/Device/Logitech_C920>
- AllAboutCircuits "Teardown Tuesday: Logitech HD Pro Webcam (C920)" — <https://www.allaboutcircuits.com/news/teardown-tuesday-logitech-hd-pro-webcam-c920/>
- Linux kernel commit `5d0fd3c806b9e932010931ae67dbb482020e0882` (johnstultz-work/linux-dev): *"uvcvideo: Disable hardware timestamps by default — reported as not working correctly on at least the Logitech C920"* — <https://github.com/johnstultz-work/linux-dev/commit/5d0fd3c806b9e932010931ae67dbb482020e0882>
- Linux kernel commit `aa50ff54f13381ab45bf611f80dee4a5696b0264` (xanmod/linux): *"uvcvideo: Quirk for autosuspend in Logitech B910 and C910"* — <https://github.com/xanmod/linux/commit/aa50ff54f13381ab45bf611f80dee4a5696b0264>
- linux-uvc-devel mailman threads (sourceforge) sobre H.264 frame-based out-of-the-box (2012), dropped/duplicated frames + decreasing timestamp en H.264 (2014), bug 66847ef capture broken por UVC timestamp support.

### 9.1 Cómo identificar la revisión que tengo en la mano

```bash
# 1. PID + bcdDevice + iProduct + iSerial:
lsusb -v -d 046d:082d 2>/dev/null | grep -E "bcdDevice|iProduct|iSerial|bcdUSB"
lsusb -v -d 046d:0892 2>/dev/null | grep -E "bcdDevice|iProduct|iSerial|bcdUSB"
lsusb -v -d 046d:08e5 2>/dev/null | grep -E "bcdDevice|iProduct|iSerial|bcdUSB"

# 2. ¿Tiene encoder H.264 hardware? Buscar 'H264' en formats:
v4l2-ctl --list-formats-ext -d /dev/video0 | grep -i h264

# 3. Modelo según el sticker inferior:
#    M/N: V-U0028 → Rev 1 OG (original, con Carl Zeiss, H.264 hardware)
#    M/N: V-U0060 → Rev 2 OrbiCam (sin Carl Zeiss, sin H.264 hardware)
#    M/N: V-U0068 / V-U0070 → Rev 3 / C920e / C920s
```

**Si aparece `Pixel Format: 'H264' (compressed)`** además de MJPG y YUYV, la unidad es Rev 1 OG (rara y valiosa). Para este proyecto **no aprovecharemos el encoder H.264** (TensorRT consume frames raw, no comprimidos) — pero la presencia de hardware H.264 indica sensor de la primera tirada con **calidad cromática y autofocus históricamente reportados como superiores**.

### 9.2 Riesgos específicos al configurar una unidad antigua (Rev 1 OG) en JetPack 4.6.x / L4T R32.7.x

| Riesgo | Causa | Mitigación |
|--------|-------|------------|
| **El control v4l2 `focus_automatic_continuous` no existe** | Kernels Linux ≥ 5.4 renombraron `focus_auto` → `focus_automatic_continuous`. **El kernel de L4T R32.7.x es Linux 4.9.337** → usa el nombre legacy. La sec. 5.5 del documento se escribió asumiendo firmware Rev 3. | Para Rev 1 OG en JetPack 4.6.x usar: `v4l2-ctl -d /dev/video0 -c focus_auto=0`. Verificar con `v4l2-ctl -L -d /dev/video0` qué nombre lista. |
| **UVC timestamp brokenness** (kernel commit `5d0fd3c806b9`) | El C920 OG reporta hardware timestamps corruptos en captura H.264 mux. | El kernel L4T R32.7.x ya incluye el parche que deshabilita `hwtimestamps` por default. Verificar `modinfo uvcvideo \| grep hwtimestamps`. Si reaparecen desfases, forzar `uvcvideo.hwtimestamps=0` en `/boot/extlinux/extlinux.conf`. |
| **USB autosuspend (heredado C910/B910)** | El quirk `UVC_QUIRK_RESTRICT_FRAME_RATE` para autosuspend roto en C910/B910 (kernel commit `aa50ff54`) **no aplica formalmente al C920**, pero el firmware Rev 1 OG comparte código. Reportes anecdóticos de cuelgue tras 5–15 min de idle. | Mantener la mitigación ya documentada en sec. 5.2: `echo -1 > /sys/module/usbcore/parameters/autosuspend`. |
| **Decreasing timestamp en H.264** (linux-uvc-devel msg `33564420`) | Modo H.264 del C920 OG reporta timestamps decrecientes → desincroniza GStreamer. | NO usar el encoder H.264 hardware. Pipeline MJPG (sec. 5.3) sigue siendo el camino correcto. |
| **`UVC_QUIRK_INVALID_DEVICE_SOF`** (kernel ≥ 6.10) | Nuevo quirk añadido upstream en 2024 para C920 con Start-of-Frame inválido. **NO disponible en L4T R32.7.x (kernel 4.9)** → no se puede backportear trivialmente. | Si `dmesg` muestra `Invalid SOF` repetidos, probar `modprobe uvcvideo quirks=128` (UVC_QUIRK_PROBE_MINMAX, lo más cercano disponible). |

### 9.3 Variantes del SKU vs revisiones internas — aclaración

Es importante distinguir entre **revisiones internas silenciosas del C920** (sec. 9) y **SKUs alternos del lineup** (todos comparten plataforma con el C920 Rev 3 moderno):

| SKU | Año | Diferencia clave vs C920 OG |
|-----|-----|------------------------------|
| **C920s** | 2019 | + Privacy shutter físico. Resto idéntico a Rev 3. |
| **C920e** | 2020 | Variante corporate/educación (gris, sin software Logi G Hub). Hardware = Rev 3. |
| **C920x** | 2021 | Variante para retailers (Best Buy / Costco). Mismo hardware Rev 3, bundle distinto. |
| **C922 Pro** | 2016 | + 720p@60fps real, background removal por software. FOV 78° igual. |
| **C930e** | 2014 | **NO es C920.** Business-class, FOV 90°, H.264 hardware **mantenido**, RightLight 2, 4× digital zoom. Ver sec. 10. |

---

### 10. Identificación de la imagen 3 — Logitech C930e (NO C920)

La tercera imagen entregada por el usuario **no corresponde a un Logitech C920** sino a un **Logitech C930e** (sucesor business-class del Logitech B910, lanzado en 2014). Identificadores visuales que la distinguen:

| Indicio en la imagen | C920 OG (imágenes 1+2) | C930e (imagen 3) |
|----------------------|--------------------------|--------------------|
| Anillo decorativo alrededor del lente | Plástico negro mate, sin destacar | **Anillo plateado/cromado distintivo** ← presente en imagen 3 |
| Branding sobre lente | "Carl Zeiss" (Rev 1 OG) o ninguno (Rev 2/3) | **"Logitech" + número de serie impreso** |
| Forma del clip de montaje | Curva simple, plástico mate uniforme | **Más ergonómico, ángulo y proporción distintos** |
| FCC ID / M/N en sticker | V-U0028 / V-U0060 / V-U0068 | **V-U0029** |
| USB Vendor:Product ID | 046d:082d / 0892 / 08e5 | **046d:0843** |

### 10.1 Tabla comparativa C920 OG vs C930e — relevante para el MVP

| Característica | C920 OG (imágenes 1+2) | C930e (imagen 3) |
|----------------|--------------------------|--------------------|
| **FOV diagonal** | 78° | **90°** |
| **FOV horizontal** | 70° | **81°** |
| Resoluciones MJPG | 1080p@30, 720p@30 | 1080p@30, 720p@30 |
| **Encoder H.264 hardware** | Solo en Rev 1 OG (probable en estas imágenes por Carl Zeiss + logo antiguo) | **Mantenido en todas las unidades** (UVC 1.5 H.264 mux) |
| RightLight 2 (HDR) | No | **Sí** |
| Digital zoom 4× | No | **Sí** (controlable vía v4l2) |
| Precio Colombia (orden de magnitud) | $300k–$380k (retail local) | $500k–$700k (importación, business channel) |
| Disponibilidad CO | Alta (Falabella, ML, Logitech Store) | Baja (canal corporativo/importación) |
| Documentación pública en Jetson Nano | **Abundante** (`dusty-nv/jetson-inference#532`, etc.) | Escasa (business-class menos común en hobby/académico) |

### 10.2 Cobertura de banda — comparativa C920 OG vs C930e

| Altura cámara | C920 OG (FOV H 70°) | C930e (FOV H 81°) |
|---------------|----------------------|---------------------|
| 30 cm | 42 cm | 51 cm |
| **40 cm** | **56 cm** | **70 cm** |
| 50 cm | 70 cm | 86 cm |

Fórmula: `ancho cubierto = 2 × altura × tan(FOV_H / 2)`.

→ **Para banda corta (altura ≤ 40 cm) y banda angosta (≈ 30–40 cm de ancho):** el C920 OG cubre con margen suficiente sin sobre-extenderse. El C930e cubre más, pero el FOV horizontal extra introduce **distorsión geométrica en los extremos** y reduce la resolución efectiva por objeto → peor entrada para YOLOv8n 416×416.

### 10.3 Recomendación final de la ronda 2

**Preferir las cámaras de las imágenes 1 y 2 (C920 OG con Carl Zeiss + logo antiguo)** sobre la imagen 3 (C930e) para este proyecto, por las siguientes razones acumuladas:

1. **FOV 78° óptimo** para la geometría declarada de la banda (30–50 cm de altura, 30–40 cm de ancho). El 90° del C930e sobrecubre y distorsiona.
2. **Precio y disponibilidad locales**: C920 OG retail Colombia ~$300k–$380k vs C930e ~$500k–$700k vía importación corporativa.
3. **Encoder H.264 hardware** del C920 OG Rev 1 (probable, por los marcadores "Carl Zeiss" + logo antiguo) es un bonus para usos futuros (streaming RTSP, telemetría remota), aunque **no es necesario** para el pipeline GStreamer + TensorRT.
4. **Mayor masa crítica de documentación pública** del C920 en Jetson Nano (issues GitHub, blogs JetsonHacks, casos Hackster) versus el C930e business-class que rara vez aparece en proyectos académicos.
5. **Sensor con Carl Zeiss** (imágenes 1+2) tiene mejor reputación cromática reportada en reviews 2012–2015, alineado con la robustez deseada frente a iluminación variable (augmentation Brightness ±25% del README).

**Si la única unidad disponible fuese el C930e (imagen 3)**, también es viable — bajar la altura de montaje a 30–35 cm para no sobrecubrir, verificar que `focus_automatic_continuous=0` responda (firmware moderno suele aceptarlo), y mantener el pipeline canónico de sec. 5.3 (MJPG → `nvv4l2decoder` / `nvjpegdec`).

---

### Track A — Ronda 2 (agentes)

**A1 (`research-web`):** Cobertura de cronología de revisiones internas del C920 OG y aclaración de la confusión SKU lineup (C920s/e/x) vs revisiones internas. Hallazgo clave: la distinción entre las dos categorías es **crítica** y no está bien documentada de forma centralizada — se reconstruye a partir de reviews, comentarios HN y commits del kernel.

**A2 (`research-code`):** Cobertura de kernel/UVC quirks específicos del C920 y comportamiento en kernels antiguos. Hallazgos clave:
- Kernel commit `5d0fd3c806b9` deshabilita `hwtimestamps` por default por bug en C920.
- Kernel commit `aa50ff54f1338...` añade quirk autosuspend para C910/B910 (predecesores del C920 OG, código de firmware compartido).
- `UVC_QUIRK_INVALID_DEVICE_SOF` introducido en kernel 6.10 — no disponible en L4T R32.7.x (kernel 4.9.337).
- `v4l2-ctl` con kernel 4.9 expone `focus_auto`, no `focus_automatic_continuous` (firmware moderno) — corregir la sec. 5.5 si la unidad resulta ser Rev 1 OG.

### Track B — Ronda 2 (búsqueda ampliada)

**discover.py — queries Exa:**

| Query | Resultados | URLs más valiosas |
|-------|-----------|---------------------|
| `Logitech C920 old new revision differences Carl Zeiss firmware history hardware change` | 29 | Stream Tech Reviews "C920 vs C920x vs C920S vs C920e", PCWorld C922 review (retirement del C920), Logitech Support C920 specs |
| `Logitech C920 960-000764 firmware version identify PCB revision teardown bcdDevice lsusb` | 19 | iFixit teardown, AllAboutCircuits Teardown Tuesday, DeviWiki entry, linux-hardware.org PIDs |
| `Logitech C920 H.264 hardware encoding disabled removed firmware update Linux UVC streaming` | 24 | HN comment 31276282 (validación H.264 retirado), Logitech UVC H.264 mux payload PR FreeRDP, OBS Forums "Disable h264 encoding", kernel commit `5d0fd3c806b9`, linux-uvc-devel threads SourceForge |

**MCP youtube:** sin sub-track de video en esta ronda (la pregunta era específica de identificación de hardware, no de tutoriales en video).

**Lectura activa:** 12 URLs nuevas extraídas con Exa `crawling_exa` + WebFetch, incluyendo iFixit teardown, AllAboutCircuits teardown, kernel commits con su descripción completa, Stream Tech Reviews comparativas, HN comment thread y linux-uvc-devel mailman archives.

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-10 | medio | Selección de cámara USB UVC para pipeline Jetson Nano + YOLOv8n FP16 TensorRT |
| 2 | 2026-05-11 | medio | Revisiones internas del C920 OG (3 generaciones documentadas) + identificación de imagen 3 como Logitech C930e + riesgos al configurar unidades antiguas en JetPack 4.6.x / L4T R32.7.x (kernel 4.9) |

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | In Practice: USB Cameras on Jetson | <https://jetsonhacks.com/2022/02/02/in-practice-usb-cameras-on-jetson/> | Blog técnico canónico | 1 |
| 2 | USB Cameras - NVIDIA Jetson (video) | <https://youtu.be/rs4mQcJAjMM> | Video JetsonHacks | 1 |
| 3 | jetsonhacks/USB-Camera | <https://github.com/jetsonhacks/USB-Camera> | GitHub repo, JetPack 4.6.1 testeado | 1 |
| 4 | dusty-nv/jetson-inference | <https://github.com/dusty-nv/jetson-inference> | GitHub repo (9k ⭐) | 1 |
| 5 | dusty-nv/jetson-utils | <https://github.com/dusty-nv/jetson-utils> | GitHub repo (881 ⭐) | 1 |
| 6 | jetson-inference #532 USB Camera C920 | <https://github.com/dusty-nv/jetson-inference/issues/532> | Issue cerrado OK | 1 |
| 7 | jetson-inference #1639 C270 JetPack 4.6.1 | <https://github.com/dusty-nv/jetson-inference/issues/1639> | Issue resuelto | 1 |
| 8 | jetson-utils #66 MJPG HW decode disabled | <https://github.com/dusty-nv/jetson-utils/issues/66> | Issue con comentario de @dusty-nv | 1 |
| 9 | jetson-inference #267 Xavier + C920 | <https://github.com/dusty-nv/jetson-inference/issues/267> | Issue + fix YUY2 format | 1 |
| 10 | jetson-utils #204 zero-copy NVMM | <https://github.com/dusty-nv/jetson-utils/issues/204> | PR JetPack 5 | 1 |
| 11 | Gstreamer use MJPEG codec (DaneLLL) | <https://forums.developer.nvidia.com/t/gstreamer-use-mjpeg-codec/145421> | NVIDIA Forums, verificado por NVIDIA mod | 1 |
| 12 | Best option(s) to decode camera mjpg frame | <https://forums.developer.nvidia.com/t/best-option-s-to-decode-camera-mjpg-frame-in-python/144903> | NVIDIA Forums | 1 |
| 13 | GStreamer pipeline for accelerated USB camera | <https://forums.developer.nvidia.com/t/gstreamer-pipeline-for-accelerated-streaming-of-usb-camera/233677> | NVIDIA Forums | 1 |
| 14 | OpenCV camera lag (Honey_Patouceul pipeline canon) | <https://forums.developer.nvidia.com/t/opencv-camera-lag/161682> | NVIDIA Forums Top Contributor | 1 |
| 15 | Two 1080p USB cameras on Nano | <https://forums.developer.nvidia.com/t/two-1080p-usb-cameras-on-nano/120276> | NVIDIA Forums (bandwidth issue) | 1 |
| 16 | Logitech c270 in python with cv2 v4l2 | <https://forums.developer.nvidia.com/t/logitech-c270-camera-in-python-with-cv2-and-v4l2/129007> | NVIDIA Forums | 1 |
| 17 | Cheese crashes C270 JetPack 4.5.1 | <https://forums.developer.nvidia.com/t/cheese-crashes-logitech-c270-usb-camera-jetpack-4-5-1-nano-2gb/174778> | NVIDIA Forums (issue conocido) | 1 |
| 18 | Getting no video output from USB webcam | <https://forums.developer.nvidia.com/t/getting-no-video-output-from-usb-webcam/204856> | NVIDIA Forums (A4tech) | 1 |
| 19 | Stack Overflow USB Camera MJPG pipeline | <https://stackoverflow.com/questions/65638140/create-pipeline-for-gstreamer-for-usb-camera-mjpg-format> | Stack Overflow | 1 |
| 20 | Counting Inspection Quality Control TensorRT (kinetika) | <https://www.hackster.io/kinetika/counting-for-inspection-and-quality-control-with-tensorrt-550b91> | Hackster.io 2024, **caso isomorfo** | 1 |
| 21 | imnuman/jetson-object-detection | <https://github.com/imnuman/jetson-object-detection> | YOLOv8n TRT 28 FPS confirmado con C920 | 1 |
| 22 | Qengineering/YoloV8-TensorRT-Jetson_Nano | <https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano> | C++ TRT YOLOv8 | 1 |
| 23 | C920 Technical Specifications (Logitech oficial) | <https://support.logi.com/hc/en-001/articles/360023307294-C920-Technical-Specifications> | Datasheet oficial | 1 |
| 24 | Logitech C270 vs C920 (techvert) | <https://www.techvert.com/logitech-c270-vs-c920/> | Comparativa técnica | 1 |
| 25 | NVIDIA Jetson and Raspberry Pi: webcam resolutions (SpyJetson) | <https://spyjetson.blogspot.com/2020/02/webcam-search-for-supported-resolutions.html> | Blog FPS benchmark C270 | 1 |
| 26 | Logitech Store Colombia oficial | <https://www.logitechstore.com.co/> | Tienda oficial CO (C270 $99.900) | 1 |
| 27 | Capital Colombia C920 | <https://www.capitalcolombia.com/productos/colombia/bogota/hardware/impresoras_camaras_escaners_televisores_video_proyectores_memorias_cables_accesorios/logitech/camara_web_logitech_c920_pro_fhd_1080x720_usb_webcam_cc1864c.php> | Retailer Bogotá ($296.600) | 1 |
| 28 | MercadoLibre Colombia C920 | <https://articulo.mercadolibre.com.co/MCO-452156023-camara-web-logitech-hd-profesional-webcam-c920-_JM> | Marketplace CO | 1 |
| 29 | Speed Logic Bogotá C270 | <https://speedlogic.com.co/tienda/camaras-web/camara-web-logitech-c270-hd-720/> | Retailer CO (20 disp.) | 1 |
| 30 | Falabella Colombia C920 | <https://www.falabella.com.co/falabella-co/product/117720216/Camara-Web-Logitech-HD-Webcam-C270-Video-HD-720p-1280x720/117720217> | Retail CO | 1 |
| 31 | Tecno Shopping Colombia C270 | <https://tecnoshopping.com.co/producto/camara-web-logitech-c270/> | Retailer CO ($130.000) | 1 |
| 32 | Arducam B0196 IMX219 USB | <https://www.arducam.com/product/b0196arducam-8mp-1080p-usb-camera-module-1-4-cmos-imx219-mini-uvc-usb2-0-webcam-board-with-1-64ft-0-5m-usb-cable-for-windows-linux-android-and-mac-os/> | Datasheet Arducam | 1 |
| 33 | Arducam B029201 IMX219 autofocus | <https://www.arducam.com/blog/product/arducam-autofoucs-imx219-usb-camera-b029201/> | Datasheet Arducam | 1 |
| 34 | Arducam B044601 HDR 101° (descartada) | <https://us.amazon.com/Arducam-Raspberry-Support-Windows-Android/dp/B0GK5CMVLB> | Amazon | 1 |
| 35 | Arducam blog Jetson camera solutions | <https://blog.arducam.com/jetson-nano-xavier-nx-camera-solutions> | Blog técnico Arducam | 1 |
| 36 | Mainsail crowsnest #96 (focus_automatic_continuous) | <https://github.com/mainsail-crew/crowsnest/issues/96> | Issue C920 firmware moderno | 1 |
| 37 | OctoPrint v4l2-ctl exposure (jaimyn.dev) | <https://blog.jaimyn.dev/how-to-get-the-best-webcam-quality-with-octoprint/> | Blog técnico C920 | 1 |
| 38 | Accelerated GStreamer NVIDIA Linux DevGuide | <https://docs.nvidia.com/jetson/archives/r35.6.2/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html> | Docs oficiales NVIDIA | 1 |
| 39 | HN comment 31276282 — H.264 encoder retirado en revisiones nuevas | <https://news.ycombinator.com/item?id=31276282> | Hacker News (validación crowdsourced del cambio silencioso) | 2 |
| 40 | Stream Tech Reviews — C920 vs C920x vs C920S vs C920e | <https://www.streamtechreviews.com/blog/c920-variations> | Review comparativo SKUs del lineup | 2 |
| 41 | iFixit — Logitech C920 Device page | <https://www.ifixit.com/Device/Logitech_C920> | Teardown + M/N V-U0062 | 2 |
| 42 | iFixit — Logitech C920 Webcam Disassembly Guide | <https://www.ifixit.com/Guide/Logitech+C920+Webcam+Disassembly/115077> | Guía de desensamble | 2 |
| 43 | AllAboutCircuits — Teardown Tuesday: Logitech HD Pro Webcam (C920) | <https://www.allaboutcircuits.com/news/teardown-tuesday-logitech-hd-pro-webcam-c920/> | Teardown con imágenes PCB (ISP Pixart PAC7332) | 2 |
| 44 | Kernel commit 5d0fd3c806b9 — uvcvideo: Disable hardware timestamps by default | <https://github.com/johnstultz-work/linux-dev/commit/5d0fd3c806b9e932010931ae67dbb482020e0882> | Linux kernel (mitigación bug C920 timestamps) | 2 |
| 45 | Kernel commit aa50ff54f1338... — uvcvideo: Quirk for autosuspend in Logitech B910 and C910 | <https://github.com/xanmod/linux/commit/aa50ff54f13381ab45bf611f80dee4a5696b0264> | Linux kernel (autosuspend quirk predecesor C920) | 2 |
| 46 | linux-uvc-devel msg 33564420 — Logitech C920 dropped/duplicated frames + decreasing timestamp en H.264 | <https://sourceforge.net/p/linux-uvc/mailman/message/33564420/> | Mailing list (bug histórico H.264 mux) | 2 |
| 47 | linux-uvc-devel thread 54930D93 — C920 capture broken by UVC timestamp support (66847ef) | <https://sourceforge.net/p/linux-uvc/mailman/linux-uvc-devel/thread/54930D93.8000008@rabbit.us/> | Mailing list (regresión 2014) | 2 |
| 48 | linux-uvc-devel thread 4F6064D6 — C920 H.264 Frame Based out-of-the-box (UVC 1.5) | <https://sourceforge.net/p/linux-uvc/mailman/linux-uvc-devel/thread/4F6064D6.5060200@fisher-privat.net/> | Mailing list (descubrimiento original UVC 1.5 mux) | 2 |
| 49 | FreeRDP PR #11132 — support Logitech UVC H.264 stream mux payload | <https://github.com/FreeRDP/FreeRDP/pull/11132> | PR código (demux H.264 desde MJPG container, referencia técnica) | 2 |
| 50 | OBS Forums — Disable h264 encoding (confirma encoder interno mantenido en C920) | <https://obsproject.com/forum/threads/disable-h264-encoding.149582/> | OBS Forums (debate sobre uso de H.264 interno) | 2 |
| 51 | linux-hardware.org — USB ID 046d:0892 (Logitech C920 HD Pro Webcam Rev 2) | <https://linux-hardware.org/index.php?id=usb%3A046d-0892> | Base de datos hardware Linux (PID Rev 2) | 2 |
| 52 | DeviWiki — Logitech HD Pro Webcam C920 | <https://deviwiki.com/wiki/Logitech_HD_Pro_Webcam_C920> | Wiki hardware (M/N 960-000764 + cross-refs) | 2 |
| 53 | Stream Tech Reviews — Logitech C920 Webcam Review | <https://www.streamtechreviews.com/blog/c920> | Review C920 con discusión 60 fps fake | 2 |

---

## Notas de la ronda 1

1. **Track A (SSD INT8 CPU) descartado durante esta investigación** — pipeline GPU FP16 TensorRT confirmado. La cámara elegida se evaluó solo contra el pipeline GStreamer GPU.
2. **No se activó AssemblyAI fallback** en `youtube_get_transcript` — los videos top tenían auto-subs disponibles.
3. **Sin gap de fuentes con autenticación**: todo lo consultado fue público.
4. **Brecha menor identificada**: no se midió empíricamente la latencia capture→tensor para cada candidata; solo se infirió de pipelines validados. Si en ronda 2 se quiere precisión, hacer bench standalone con `fpsdisplaysink` en la cámara comprada.

---

## Notas de la ronda 2

1. **Aclaración semántica inicial:** la pregunta del usuario "todos los posibles modelos del C920" se interpretó primero como SKUs del lineup (C920s/C920e/C920x). El usuario corrigió: se refería a las **revisiones internas silenciosas del mismo C920** detectables por logotipo antiguo y branding Carl Zeiss. El plan se reformuló y la sec. 9 documenta exactamente esa cronología.
2. **Imagen 3 no era C920:** la unidad de la tercera imagen es un **Logitech C930e** (business-class, FOV 90°, sucesor del B910). Identificado por el anillo cromado del lente y el branding sobre el cristal. Documentado en sec. 10 con comparativa contra C920 OG y recomendación final.
3. **Corrección al pipeline canónico:** si la unidad comprada resulta ser **C920 Rev 1 OG** (logo antiguo + Carl Zeiss + M/N V-U0028), el control v4l2 a usar es `focus_auto=0` (legacy), **NO** `focus_automatic_continuous=0` documentado en sec. 5.5 (esa nomenclatura aplica a firmware Rev 3 con kernel ≥ 5.4). El kernel 4.9.337 de L4T R32.7.x expone el nombre legacy.
4. **Sub-track YouTube no activado en esta ronda:** la pregunta era de identificación de hardware/firmware, no de tutoriales. Sin transcripts ni AssemblyAI fallback en R2.
5. **Brecha no resuelta:** no se pudo confirmar con 100% de certeza el USB PID del C930e (`046d:0843` reportado en múltiples fuentes; `046d:0841` en linux-hardware.org corresponde a "Webcam C920-C", variante OEM separada). Validación empírica al recibir la cámara: `lsusb` y comparar con la tabla de sec. 9.
6. **Decisión final pendiente del usuario:** elegir entre las imágenes 1+2 (C920 OG con Carl Zeiss — recomendado) o imagen 3 (C930e — viable pero sobre-cobertura para banda corta). Ambas funcionan con el pipeline canónico de sec. 5.3 sin modificaciones.
