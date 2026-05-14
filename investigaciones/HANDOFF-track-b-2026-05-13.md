# HANDOFF Track B — YOLOv8n sobre Jetson Nano B01

> **Fecha de corte:** 2026-05-13
> **Autor sesión origen:** Martín García + Claude Code (sesión interrumpida por context compaction)
> **Tipo de documento:** registro de decisiones, no manual operacional. Aquí NO hay comandos copy-pasteables; aquí está el **por qué** de cada decisión que se tomó y el estado en que quedó el proyecto.
> **Ámbito:** solamente **Track B** (YOLOv8n → ONNX → TensorRT FP16 sobre Maxwell). Track A (SSD MobileNet v2 → TFLite INT8) queda **fuera de alcance** por decisión del usuario el 2026-05-13 — la racional está en §1.3.
> **Lo descartado explícitamente:** AP-mode con `hostapd-rtw`, driver out-of-tree TP-Link, todo lo relacionado con el chipset Realtek RTL8188EUS. Si en alguna parte del documento aparece una mención a esos temas, es solo como contexto histórico para no reabrir caminos descartados.

---

## 0. Cómo usar este documento

Tres modos de lectura:

1. **Lectura completa de orientación (recomendada al iniciar la siguiente sesión):** §1 → §3 → §6 → §9. Da el cuadro general, lo ya validado, los aprendizajes que se perderían sin documentarlos, y los siguientes pasos sugeridos.
2. **Consulta puntual de una decisión:** §2 (ledger filtrado a Track B). Cada decisión tiene ID, estado, razón, dependencias y referencias.
3. **Cuando aparece un problema concreto:** §8 (gotchas críticos de Track B) → §11 (referencias).

Este documento sustituye sin pérdida de contenido los archivos previos en `c:\Users\mitgar14\Documentos\embebidos-3\investigaciones\`:

- `2026-05-05/2026-05-05-arquitectura-software-jetson-nano.md`
- `2026-05-05/2026-05-05-datasets-deteccion-residuos.md`
- `2026-05-05/2026-05-05-dual-track-yolov8-vs-ssd.md`
- `2026-05-05/2026-05-05-preprocessing-roboflow.md`
- `2026-05-10/2026-05-10-camara-usb-jetson-nano.md`
- `2026-05-12/CONSOLIDADO-embebidos-3.md`

Todos serán eliminados después de validar este handoff. Si en algún momento se necesita revisar el material original, hay que recuperarlo del histórico de git (`git log --all -- investigaciones/`).

---

## 1. Contexto del proyecto

### 1.1 Objetivo MVP

Construir un clasificador de residuos para banda transportadora industrial, embebido en una Jetson Nano B01 (4 GB), capaz de:

- Detectar y clasificar en tiempo real objetos de tres clases: **plastic**, **glass**, **paper**.
- Mantener al menos **10 FPS sostenidos** en condiciones de operación reales (con servos PCA9685 vía I²C consumiendo ciclos CPU en paralelo).
- Funcionar **sin conexión a internet** durante la demo de evaluación final el **2026-05-26**.

El proyecto se desarrolla en el marco de un curso de Sistemas Embebidos en una universidad colombiana, con entrega esperada como informe IEEE-style + demo física.

### 1.2 Hardware actual (verificado en SSH 2026-05-13; re-verificado 2026-05-14)

| Componente | Especificación verificada |
|---|---|
| SoC | NVIDIA Tegra X1 (Maxwell `sm_53`, 128 CUDA cores, NO Tensor Cores, NO instrucción `dp4a`) |
| CPU | ARM Cortex-A57 quad-core @ 1.43 GHz, con NEON SIMD |
| RAM | 4 GB LPDDR4 unificada CPU/GPU |
| Almacenamiento | microSD (capacidad por confirmar empíricamente) |
| OS | Ubuntu 18.04 bionic, kernel `4.9.337-tegra`, aarch64 |
| Userspace ML | JetPack **4.6.1** (L4T R32.7.1) — CUDA 10.2.300, cuDNN 8.2.1, TensorRT **8.2.1.8**, Python **3.6.9**, OpenCV 4.1.1 |
| Cámara | Logitech **C920 OG** Rev 1 (PID `046d:082d`) — todavía **no conectada físicamente** a la Nano |
| Servos / actuadores | PCA9685 vía I²C (bus por verificar empíricamente: probable `/dev/i2c-1`) |
| Cooling | `pwm-fan` accesible, actualmente con `target_pwm=0` (apagado) |
| Wi-Fi | dongle USB TL-WN722N v4 (RTL8188EUS) ya configurado por Nicolas Cuaran como cliente, sin AP mode |
| Acceso remoto | x11vnc + GDM autologin + Tailscale `--ssh` operativos vía herramienta `looker-remote-desktop` |
| Tailscale IP | `100.100.166.120` (alias SSH `nano` configurado en `C:\Users\mitgar14\.ssh\config`) |
| `uv` | instalado en `/home/jetson/.local/bin/uv` (musl standalone, instalado por Nicolas) |

### 1.3 Por qué Track B (y no Track A)

El proyecto **arrancó con un diseño dual-track** que se proponía:

- **Track A:** SSD MobileNet v2 plain 320×320 → TFLite INT8 (CPU + XNNPACK + NEON SIMD).
- **Track B:** YOLOv8n 416×416 → ONNX → TensorRT FP16 (GPU Maxwell).

El propósito era comparar las dos rutas en mAP y FPS reales sobre el hardware embebido, y dejar la decisión final como aporte al informe IEEE.

**El usuario decidió el 2026-05-13 enfocar exclusivamente Track B.** Razones que se desprenden de las investigaciones consolidadas:

1. **Track B explota el hardware de la Nano más distintivamente.** La Tegra X1 tiene una GPU Maxwell con 128 CUDA cores que solo brilla con FP16; Track A correría en CPU, exactamente igual que si lo desplegáramos sobre un Raspberry Pi 4. Si la diferenciación competitiva del proyecto es "estamos usando una GPU embedded", Track B es el único que lo demuestra.
2. **El stack Track A tiene más puntos frágiles para el calendario disponible.** Track A depende de QAT obligatorio (sin QAT, MobileNet v2 INT8 colapsa de 71% a 3% top-1 según Yun & Wong CVPR 2021), de `TFLite_Detection_PostProcess` embebido en el grafo, de calibración con representative dataset de 300-500 muestras bien distribuidas, y de cuatro gates de validación antes del deploy. Track B también tiene gates pero son más predecibles.
3. **Benchmarks reproducibles públicos respaldan Track B en este hardware.** Nature Scientific Reports 2024 (DOI 10.1038/s41598-024-74798-3 Tabla 4) reporta YOLOv8n 416×416 TensorRT FP16 en Jetson Nano: **30 FPS**. Qengineering verifica **19 FPS a 640×640**. Špeh 2023 verifica YOLOv7-tiny a **17 FPS @ 416**. Hay redundancia de evidencia.
4. **El usuario ya invirtió tiempo y créditos en preparar Track B.** Existe la versión 1-B del dataset en Roboflow (`embebidos3/waste-3class-lwld8` v1-B, exportada en formato `yolov8`, 3 clases, resize Fit-black 416×416), existe el repo HuggingFace Hub `mitgar14/embebidos-3-models` privado para persistir checkpoints, existe el bootstrap script para Vast.ai con dual venv y CommitScheduler.

**Lo descartado por enfoque Track B exclusivo:**

- D1 (Stack Track A: TF 2.15 + TFOD API SHA `9cafa3d150` + SSD MobileNetV2 320).
- D10 (TFLite_Detection_PostProcess embebido).
- D12 (Gate 1 + Gate 2 TFLite, op_version + wheel Coral CP38 x86).
- Toda la versión 1-A del dataset Roboflow (exportada en `tfrecord`).
- Toda la rama del repo bootstrap relativa a Track A (TF 2.15 install, TFOD API checkout, Pillow<10, protobuf==3.20).

Estas decisiones quedan en el ledger del CONSOLIDADO original como referencia histórica, pero **no se ejecutan en este proyecto**.

### 1.4 Lo que está fuera de alcance (recordatorio explícito del usuario)

El usuario fue enfático en su mensaje del 2026-05-13:

> "Todo lo que tenga que ver con el driver del TP Link, descartalo. Reitera."
> "Descartar AP-mode por el momento: no es necesario realmente."
> "Es muchísimo más necesario ir viendo compatibilidad de drivers primero, para verificar compatibilidad con todo el stack a usar en el training de vast.ai que luego se desplegará a este dispositivo."

Por lo tanto **NO se trabajará** en:

- Compilación o reinstalación del driver `lwfinger/rtl8188eu`. El dongle USB ya funciona en modo cliente; eso basta.
- AP mode con `hostapd-rtw`, `dnsmasq`, IP estática `10.42.0.1/24` en `wlan1`.
- Persistencia del módulo del kernel `8188eu` vía systemd.
- Configuración de `country_code=CO`, `channel=6`, `driver=rtl871xdrv`.
- Conflictos `aircrack-ng/rtl8188eus` vs `lwfinger/rtl8188eu`.

Si en una sesión futura aparece un problema con el dongle USB, ese problema lo resuelve Nicolas Cuaran o se documenta como bloqueo, **no se aborda desde Claude Code**.

### 1.5 Topología de acceso a la Nano

```
                 Internet
                    │
   ┌────────────────┼─────────────────┐
   │                │                 │
Jetson Nano       Win11               (Nicolas Cuaran tiene
(Tailscale       (Tailscale            la cuenta Tailscale,
 100.100.166.120  100.100.166.121      el operador es
 SSH alias        SSH alias            cliente del tailnet)
 'nano')          'mitgar14-win11')

Protocolo: Tailscale `--ssh` con ACL en modo `check` (auth periódica vía URL)
Cliente Win11: OpenSSH built-in
Auth: federada vía Tailscale (no se usan llaves SSH manuales)
```

Configuración SSH ya creada en `C:\Users\mitgar14\.ssh\config`:

```sshconfig
Host nano
    HostName 100.100.166.120
    User jetson
    ServerAliveInterval 30
    ServerAliveCountMax 6
Host nano-mdns
    HostName jetson-nano
    User jetson
    ServerAliveInterval 30
    ServerAliveCountMax 6
```

**Implicación operativa importante (descubierta empíricamente en SSH 2026-05-13):** los comandos SSH no-login (`ssh nano 'CMD'`) NO cargan `~/.profile`, por lo que el binario `uv` instalado en `/home/jetson/.local/bin/uv` no aparece en el `PATH` por defecto. Hay dos formas de manejarlo: (a) usar `ssh nano 'bash -lc "CMD"'` para forzar shell de login, o (b) prefijar manualmente `PATH=$HOME/.local/bin:$PATH CMD`. La opción (a) es la canónica y la que se aplicó en los smoke tests TRT.

---

## 2. Ledger de decisiones vinculantes (filtrado Track B)

Convenciones repetidas para conveniencia de lectura:

- **VINCULANTE:** decisión firme, no cuestionar sin nueva evidencia.
- **REFINADA:** sustituida por otra decisión más reciente.
- **OBSOLETA:** descartada por evidencia empírica.
- **Ronda origen:** indica en qué momento del proyecto se consolidó (R4-R8 cubren las rondas de investigación previas).

### D2 — Stack Track B: YOLOv8n 416×416 + Ultralytics ≥8.4.46 + ONNX opset 11

**Estado:** VINCULANTE | **Ronda:** R4 | **Sustituye:** ninguna | **Dependencias:** D3, D5, D6, D13, D14

**Decisión:** entrenar **YOLOv8n (nano)** a resolución **416×416**, exportar a ONNX con **opset 11**, `dynamic=False`, `simplify=True`, y `nms=False`. Versión de Ultralytics fija en el rango **≥8.4.46, <8.5** para evitar tanto el bug de calibración INT8 con `imgsz` no cuadrado (fixed en PR #24028, marzo 2026) como rupturas de API hacia 8.5+.

**Razón:**

1. **YOLOv8n es la única variante con menos de 3 M parámetros y latencia aceptable en Maxwell.** Las variantes `s` (9 M) y `m` (25 M) saturan la memoria unificada o no llegan a 10 FPS.
2. **416×416 es óptimo por evidencia cuantitativa redundante.** Nature 2024 Tabla 4: YOLOv8n FP16 TRT en Nano → 416 = 30 FPS, 512 = 29 FPS, 640 = 24 FPS. Espstack 2023: YOLOv7-tiny → 416 = 17 FPS, 640 = 9 FPS (1.9× de ganancia bajando de 640 a 416). Alqahtani 2024 confirma que reducciones agresivas (640 → 320) cuestan ~28 puntos de mAP — el sweet spot está exactamente en 416.
3. **ONNX opset 11 es el máximo soportado por TensorRT 8.2.** Opset 12+ falla con `UNSUPPORTED_NODE` para nodos `Slice` y `Resize` que YOLOv8 usa internamente. La documentación oficial NVIDIA confirma: TRT 8.2 soporta operadores hasta opset 13, pero las features reales se quedan en opset 11.
4. **`nms=False` es obligatorio en el `.onnx`.** El NMS se hace fuera del grafo en el Nano. Estrategia tri-path (corrección 2026-05-14 vía SSH): **V0 default** — `cv2.dnn.NMSBoxes` CPU NumPy post-proc, compatibilidad garantizada. **V1 smoke test** — `EfficientNMS_TRT` plugin: el binary de TRT 8.2.1.8 del JP 4.6.1 lo incluye con el fix del issue NVIDIA/TensorRT#1538 (commit `3235cc2`, julio 2021); queda como path a validar empíricamente, no descartado. **V2 fallback** — `BatchedNMSDynamic_TRT`, plugin estable del mismo binary.
5. **`dynamic=False` y `simplify=True` son obligatorios.** `dynamic=True` requiere optimization profile (overhead innecesario para una resolución fija). `simplify=False` deja nodos `Slice` con shape dinámica que TRT 8.2 rechaza.

**Implicaciones:**

- `numpy < 2.0` en el venv Track B (Ultralytics 8.4.x todavía usa aliases removed: `np.float`, `np.int`).
- `onnxslim >= 0.1.34` para evitar bug de simplificación silenciosa de nodos `Slice`.
- El export ONNX en x86 produce un archivo con 234 nodos, 14 tipos de ops, input shape `[1, 3, 416, 416]`, output `[1, 84, 3549]` — todos los ops están dentro del whitelist de TRT 8.2.

**Referencias clave:** Ultralytics docs (`docs.ultralytics.com/integrations/onnx/`), NVIDIA TRT operators matrix (`docs.nvidia.com/deeplearning/tensorrt/operators/`), Nature Scientific Reports 2024 (DOI 10.1038/s41598-024-74798-3).

---

### D3 — Cloud training en Vast.ai con container `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` + RTX 4090

**Estado:** VINCULANTE | **Ronda:** R4-R5 | **Sustituye:** D4 (Colab/Kaggle) | **Dependencias:** D5, D6, D8, D11, D17

**Decisión:** entrenar YOLOv8n en Vast.ai sobre una instancia con RTX 4090 a ~0.40 USD/hora, usando el container base oficial Vast.ai con CUDA 12.4.1, cuDNN devel, Ubuntu 22.04 y Python 3.10. Presupuesto total estimado para Track B: ~3 USD (6-8 horas de training + experimentos de ablation).

**Razón:**

1. **Colab Pro+ y Kaggle Notebooks no son viables.** Colab tiene timeout de 24 h sin garantía de GPU asignada, sin sudo, kernel reinicia al `apt install`. Kaggle limita a 12 h continuas y 30 h/semana, sin sudo, sin SSH, sin `apt install`.
2. **Vast.ai resuelve los tres bloqueos:** SSH funcional + sudo + sin timeouts arbitrarios. El container base oficial tiene driver NVIDIA preinstalado y CUDA 12.4 listo.
3. **RTX 4090 es el sweet spot de costo/throughput.** Tarda ~2 h en entrenar YOLOv8n 100 epochs sobre el dataset 1-B (10 439 imágenes). A 0.40 USD/h, ~0.80 USD por entrenamiento.
4. **CUDA 12.4 en cloud + CUDA 10.2 en Nano es intencional y aceptable.** El modelo entrenado se exporta a ONNX, formato neutral al CUDA del cloud. La compilación del engine TensorRT ocurre EN LA NANO, no en el cloud (ver D14).
5. **Ubuntu 22.04 garantiza `glibc 2.35`**, compatible con todos los wheels modernos de PyTorch y Ultralytics. Bionic 18.04 (Nano) tiene `glibc 2.27`, demasiado vieja, pero como solo necesitamos correr en Nano el `.engine` compilado localmente + Python wrappers livianos, no requiere wheels modernos en Nano.

**Implicaciones:**

- Costo presupuestado: ~3 USD total Track B (vs ~5 USD del proyecto dual-track original).
- Dual venv obligatorio (D6) — aunque solo se use Track B ahora, queda el patrón para extensibilidad futura.
- Auto-destroy obligatorio (D11) para evitar facturación por olvido.
- Persistencia obligatoria en HF Hub (D8) porque el filesystem del container se borra al destroy.

**Referencias clave:** Vast.ai base image registry (`hub.docker.com/r/vastai/base-image/tags`), Vast.ai CLI docs (`vast.ai/docs/cli/commands`).

---

### D5 — `uv` como gestor Python en Vast.ai (y en Jetson Nano por extensión)

**Estado:** VINCULANTE | **Ronda:** R5 | **Sustituye:** ninguna | **Dependencias:** D3

**Decisión:** usar `uv` (https://docs.astral.sh/uv/) como gestor de paquetes y entornos virtuales tanto en el container Vast.ai como en la Jetson Nano. NO usar `pip + venv`, NO usar conda, NO usar poetry, NO usar pipx.

**Razón:**

1. **Cumple regla global del usuario.** El archivo `~/.claude/CLAUDE.md` declara explícitamente: "Usar uv en lugar de pip + venv + pyenv + poetry + pipx + twine".
2. **uv resuelve dependencias 10-100× más rápido que pip.** En la práctica, instalar el stack Track B (Ultralytics + numpy + onnx + onnxslim + roboflow + jupyter) tarda <30 s vs ~3 min con pip + resolver clásico.
3. **uv soporta lockfile multi-platform.** Permite congelar versiones exactas y replicar el entorno entre Vast.ai (x86 Ubuntu 22.04 py310) y, si fuera necesario, en otro entorno.
4. **uv está ya instalado en la Jetson** (Nicolas Cuaran lo instaló en `/home/jetson/.local/bin/uv` con el installer musl standalone). Esto significa que el patrón "todo con uv" se mantiene end-to-end.

**Implicaciones:**

- `pyproject.toml` opcional con `[project.optional-dependencies] trackb` para el stack de training; `uv pip install --python /opt/venv/trackb/bin/python <pkgs>` es la sintaxis canónica.
- En la Jetson, las invocaciones deben usar `bash -lc 'uv ...'` o expandir `PATH` manualmente porque SSH no-login no carga `~/.profile`. Ver §1.5.
- Para deps con compilación nativa (`pycuda`, `tensorrt` python bindings), `uv` puede fallar si no hay headers de desarrollo — en ese caso, fallback a `pip` dentro del mismo venv es aceptable.

**Referencias clave:** uv docs (`docs.astral.sh/uv/`).

---

### D6 — Dual venv `/opt/venv/trackb` registrado como ipykernel

**Estado:** VINCULANTE (Track A queda colgante pero el patrón se mantiene) | **Ronda:** R5 | **Sustituye:** ninguna | **Dependencias:** D3, D5

**Decisión:** crear el venv `/opt/venv/trackb` con Python 3.10 dentro del container Vast.ai, registrarlo como ipykernel separado (`display-name "Track B (YOLOv8)"`), e instalar ahí Ultralytics 8.4.46 + numpy<2.0 + onnx + onnxslim + onnxruntime-gpu + huggingface_hub + wandb + roboflow + jupyter + ipykernel.

**Razón:**

Originalmente el proyecto tenía un dual venv `/opt/venv/tracka` + `/opt/venv/trackb` porque TF 2.15 (Track A) y Ultralytics 8.4.46 (Track B) tienen conflictos transitivos: TF requiere `numpy<1.26` + `protobuf==3.20`, Ultralytics requiere `numpy<2.0` (compatible con 1.26.x pero no con 1.23.x que TF prefiere). Forzarlos en un solo venv genera resolución imposible.

Con el enfoque exclusivo a Track B, el venv `/opt/venv/tracka` ya no es necesario. **Pero el patrón se mantiene** porque:

- Si en algún momento se necesita un segundo venv (por ejemplo para correr Polygraphy en x86 con versiones de TRT diferentes), el patrón está montado.
- El uso de ipykernel separado permite que cualquier notebook escoja explícitamente "Track B (YOLOv8)" al abrirse, eliminando ambigüedad sobre qué Python está activo.

**Implicaciones:**

- Notebooks Track B deben empezar con la cabecera `# Kernel: Track B (YOLOv8)` y ser ejecutables solamente desde ese kernel.
- El bootstrap script (D17) crea el venv vacío + lo registra como ipykernel + instala las deps en orden determinístico (Ultralytics primero, después el resto).

**Referencias clave:** Jupyter ipykernel docs (`ipython.readthedocs.io/en/stable/install/kernel_install.html`).

---

### D8 — Persistencia en HF Hub `mitgar14/embebidos-3-models` (privado) + `CommitScheduler(every=5)`

**Estado:** VINCULANTE | **Ronda:** R5 | **Sustituye:** ninguna | **Dependencias:** D3

**Decisión:** los artefactos (checkpoints `.pt`, ONNX exportado, métricas, logs, sample predictions) se persisten en el repo privado de Hugging Face Hub `mitgar14/embebidos-3-models`. El upload es incremental cada **5 minutos** vía `huggingface_hub.CommitScheduler`, que corre como thread separado y no interfiere con el training.

**Razón:**

1. **El filesystem del container Vast.ai se borra cuando se destruye la instancia.** Sin persistencia externa, todo se pierde — incluido el último checkpoint en caso de OOM o crash.
2. **CommitScheduler es superior a `upload_folder` manual** porque no requiere coordinar con el script de training (no hay "sube al final"; sube continuamente). Si el training falla en epoch 47, hay snapshot hasta el epoch 45 ya en el Hub.
3. **HF Hub privado es gratuito** para repos individuales hasta cierto tamaño. Para Track B los artefactos pesan <500 MB total (un YOLOv8n.pt pesa ~6 MB, un ONNX exportado pesa ~12 MB).
4. **El usuario ya tiene la cuenta y el repo creado.** El token va por env var `HF_TOKEN` (NUNCA hardcoded en código).

**Implicaciones:**

- Bootstrap script debe configurar `huggingface-cli login` con el token leído de env var.
- La última celda del notebook debe forzar un `scheduler.trigger()` final + `scheduler.stop()` para garantizar que el último estado se subió antes del auto-destroy.
- Si en alguna ronda futura el Hub se cae, plan B es Google Drive vía `rclone`. Pero hasta hoy nunca ha sido necesario.

**Referencias clave:** `huggingface_hub.CommitScheduler` API docs.

---

### D9 — `tmux` como wrapper de procesos de larga duración

**Estado:** VINCULANTE | **Ronda:** R5 | **Sustituye:** ninguna | **Dependencias:** ninguna

**Decisión:** lanzar el training, los exports y las validaciones dentro de sesiones `tmux` separadas. NO usar `nohup`, NO usar `screen`.

**Razón:**

1. **El SSH a Vast.ai puede caerse en cualquier momento** (universidad bloquea conexiones largas, hotspot móvil pierde señal, etc.). Sin tmux, la sesión SSH muere y el proceso muere con ella.
2. **tmux preserva la sesión completa** — al reconectar (`tmux attach -t training`), se ven los outputs históricos del comando, no solo lo nuevo.
3. **`screen` tiene UX inferior** (atajos legacy, menos windows-friendly).
4. **`nohup` pierde stdout interactivo** y no permite re-attach: solo deja un `nohup.out` post-mortem.

**Implicaciones:**

- Convención: sesión `training` para los notebooks principales, sesión `jupyter` para el JupyterLab server.
- Atajos canónicos: `Ctrl+B D` para detach, `tmux attach -t training` para reattach, `Ctrl+B [` para entrar en modo scroll.

**Referencias clave:** tmux man page (`man.openbsd.org/tmux.1`).

---

### D11 — Auto-destroy con cron watchdog + última celda + `$VAST_CONTAINERLABEL`

**Estado:** VINCULANTE | **Ronda:** R5 | **Sustituye:** ninguna | **Dependencias:** D3, D17

**Decisión:** combinar dos mecanismos de protección contra facturación por olvido:

1. **Cron watchdog en el container:** cada 5 minutos, un script verifica `nvidia-smi --query-gpu=utilization.gpu`. Si la GPU está por debajo del 5 % durante 6 muestreos consecutivos (30 minutos idle), ejecuta `vastai destroy instance ${VAST_CONTAINERLABEL#C.}`.
2. **Última celda del notebook:** al terminar el training y el export, la celda final hace `scheduler.stop()` + `vastai destroy instance ${VAST_CONTAINERLABEL#C.}` de forma síncrona.

**Razón:**

1. **RTX 4090 on-demand cuesta ~0.40 USD/hora.** Olvidar una instancia un fin de semana = factura inesperada de ~$15-20. Doble protección elimina ese riesgo.
2. **Cron watchdog cubre el caso "el notebook falló al final".** Si el training crashea antes de la última celda, el cron termina destruyendo después de 30 min idle.
3. **Última celda cubre el caso "el training terminó pero la GPU no quedó idle aún".** El cron tarda hasta 35 min en disparar; la última celda destruye inmediatamente.
4. **`$VAST_CONTAINERLABEL` es la variable de entorno que el host Vast.ai inyecta en el container con el ID de instancia.** El sufijo `#C.` strip el prefijo `C.` que Vast.ai pone delante (formato: `C.123456`).

**Implicaciones:**

- Si se está debugging interactivamente y la GPU queda idle por un break para café, **se destruye sola**. Hay que mantener `nvidia-smi -l 5` corriendo en otra ventana de tmux para tener algo de utilización continua, o ajustar el threshold del watchdog.
- El cron script vive en `/opt/scripts/check-gpu-idle.sh` con un counter en `/tmp/gpu_idle_count`.

**Referencias clave:** Vast.ai CLI destroy docs (`vast.ai/docs/cli/commands`).

---

### D13 — Gate 3: ONNX ops blacklist TRT 8.2 + Gate 4: Polygraphy en Docker NGC TRT 21.11

**Estado:** VINCULANTE | **Ronda:** R6 | **Sustituye:** ninguna | **Dependencias:** D2

**Decisión:** antes de copiar el ONNX a la Jetson Nano, validar dos cosas:

1. **Gate 3:** inspeccionar el ONNX en x86 con un script Python que verifica:
   - `opset_imports[0].version == 11` (no más, no menos).
   - `ir_version <= 10`.
   - Ninguna op presente en blacklist (`GridSample`, `DFT`, `IsInf`, `IsNaN`, `MelWeightMatrix`, `STFT`, `SequenceInsert`, `CumSum`, `NonZero`, `RoiAlign`, `QLinearConv`, `QLinearMatMul`, `Reciprocal`).
   - Cualquier `ConstantOfShape` tiene `data_type == 1` (FP32).
   - NO hay `NonMaxSuppression` embebido (debería estar removido por `nms=False`).
2. **Gate 4:** correr Polygraphy en el container Docker NGC `nvcr.io/nvidia/tensorrt:21.11-py3` (TRT 8.2.1, CUDA 11.5, Ubuntu 20.04, Py 3.8), que es el match exacto de versión de TRT a la Nano (8.2.1.8). Comando canónico:
   - `polygraphy run /workspace/best.onnx --onnxrt --trt --atol 1e-2 --rtol 1e-2 --input-shapes images:[1,3,416,416]`
   - Esto compila el engine in-container y compara los outputs de ONNX Runtime vs TensorRT para detectar divergencias numéricas.

**Razón:**

1. **Compilar engines en la Nano y descubrir errores tras 25 minutos es un flujo intolerable.** Polygraphy en x86 con la MISMA versión de TRT predice >95 % de los errores de compilación.
2. **El parser ONNX → TRT es el origen más común de errores.** Ops fuera de opset 11, `ConstantOfShape` con tipos no-FP32 (Int64 a veces aparece tras `onnxslim`), o `NonMaxSuppression` mal embebido son los responsables del 80 % de los rebotes.
3. **Polygraphy NO funciona nativo en la Nano** (Python 3.6.9 incompatible con `polygraphy ≥0.45` que requiere Py 3.8+). Por eso el flujo es: Polygraphy validation en x86 Docker → si pasa, copiar ONNX a Nano y compilar con `trtexec` directo.
4. **El Docker NGC 21.11 tiene TRT 8.2.1 + Ubuntu 20.04 + Py 3.8.** TRT 21.10 (8.0.x) y TRT 22.01 (8.2.3) NO son matches exactos; 21.11 sí.

**Implicaciones:**

- Requiere Docker daemon disponible en el host de validación. En Vast.ai, si Docker-in-Docker no está activo, hay que correr Gate 4 desde la Win11 del operador con Docker Desktop.
- Gate 3 corre rápido (segundos); Gate 4 tarda ~2 minutos por la pulldown de la imagen NGC la primera vez (~6 GB) y ~30 s una vez cached.
- **Gate 4 es NECESARIO PERO NO SUFICIENTE.** Las fusiones de kernels específicas de Maxwell `sm_53` no se reproducen en x86 con TRT 8.2 (Maxwell desktop). Después de Gate 4, hay que hacer un smoke test rápido en la Nano (compilar el engine con `trtexec --fp16 --workspace=1024 --saveEngine=best.engine` + medir latencia con `trtexec --loadEngine=best.engine --iterations=100`). Solo ese smoke test confirma que el modelo está realmente listo.

**Referencias clave:** Polygraphy CHANGELOG (`NVIDIA/TensorRT/tools/Polygraphy/CHANGELOG.md`), NGC TensorRT container catalog, foro NVIDIA #349598 (Polygraphy falla en JP 4.6.1).

---

### D14 — INT8 Maxwell `sm_53` queda como FP16-only por default

**Estado:** VINCULANTE | **Ronda:** R6 | **Sustituye:** ninguna | **Dependencias:** D2, D13

**Decisión:** compilar el engine TensorRT en la Nano con `--fp16` y NO `--int8`. Considerar un experimento INT8 opcional de 45-60 minutos en la propia Nano, **únicamente si hay margen de tiempo antes de la entrega** y con criterio binario de adopción.

**Razón:**

1. **Maxwell `sm_53` carece de la instrucción `dp4a`** (dot product de 4 × INT8 acumulando a INT32). Esta instrucción fue introducida en Pascal `sm_61` en 2016 y es la única forma de acelerar INT8 en hardware NVIDIA. Sin `dp4a`, TensorRT tiene tres opciones:
   - Usar kernels CUDA INT8 SIMD vía `dp4a` → no disponible.
   - Emular INT8 vía FP16/FP32 → elimina cualquier beneficio de velocidad.
   - Mixed precision fallback → TensorRT revierte la capa a FP16, generando grafo mixto.
   En ninguno de los tres casos hay speedup neto.
2. **Confirmación empírica de Qengineering (repo `YoloV8-TensorRT-Jetson_Nano`):** *"All models are quantized to `FP16`. The `int8` models don't give any increase in FPS, while, at the same time, their mAP is significantly worse."* El autor publicó la tabla FP16 (YOLOv8n = 19 FPS @ 640) pero no la INT8 porque concluyó que el upside es nulo.
3. **Confirmación empírica de espstack.com es contradictoria** (reporta YOLOv8n INT8 a 28-32 FPS), pero esa fuente NO especifica dataset de calibración, NO publica código, y reporta mAP50 0.887 que es coherente con COCO no con custom 3 clases. Confianza baja, descartada.
4. **El criterio binario de adopción** si se hace el experimento opcional:
   - **Adoptar INT8** si y solo si `FPS_INT8 ≥ FPS_FP16 × 1.10` **Y** `mAP_INT8 ≥ mAP_FP16 − 5 pp`.
   - **Abandonar INT8** si cualquiera de los dos criterios falla.
   - Zona gris (entre 0% y 10% de ganancia FPS) → abandonar por el criterio del 10%. No vale la pena la complejidad por <10% de gain.

**Implicaciones:**

- El stack default para Track B es FP16. Todo el flujo de validación (D13) está pensado para FP16.
- Si se hace el experimento INT8, requiere generar un calibration cache con `IInt8EntropyCalibrator2` Python custom (porque Polygraphy falla en JP 4.6.1) + `trtexec --int8 --calib=<cache>`.
- La calibración debe usar 300-500 muestras del val split, distribuidas proporcionalmente entre plastic/glass/paper. Imágenes limpias, sin augmentations adicionales post-hoc (Karimov 2025 refuta el beneficio de mixed calibration en INT8 PTQ).

**Referencias clave:** Issue NVIDIA/TensorRT#3762 (`--int8 means Enable int8 precision, in addition to fp32`), Qengineering repo, Karimov et al. 2025 (`arXiv:2508.19600`).

---

### D17 — Vast.ai como entorno primario de training (sustituye D4)

**Estado:** VINCULANTE | **Ronda:** R4 | **Sustituye:** D4 (Colab Pro+ / Kaggle) | **Dependencias:** D3, D5, D6, D8

**Decisión:** RTX 4090 on-demand en Vast.ai es el entorno de training canónico. Costo total presupuestado para Track B: ~3 USD (6-8 horas de cómputo distribuidas en training base + experimentos de ablation).

**Razón:** ver D3 para la racional completa. Resumen: Colab y Kaggle no proveen sudo + sin timeouts + GPU dedicada, los tres bloqueos críticos del proyecto.

---

### D18 — Notebooks `.ipynb` ejecutados vía `jupyter nbconvert --execute --inplace` dentro de tmux (sustituye D7)

**Estado:** VINCULANTE | **Ronda:** R5 | **Sustituye:** D7 (scripts `.py` con `python` directo) | **Dependencias:** D6, D8, D9

**Decisión:** el pipeline de training Track B corre como un notebook `.ipynb` (no como `.py`), ejecutado con `jupyter nbconvert --execute --inplace --to notebook tracka_trackb.ipynb` dentro de una sesión tmux.

**Razón:**

1. **Los notebooks preservan outputs incrustados** (loss curves, sample predictions, métricas pre/post export). Estos outputs se commitean al HF Hub y son evidencia auditable para el informe IEEE.
2. **Los `.py` requerirían un código separado para generar y guardar las gráficas a disco.** Duplicación de esfuerzo.
3. **Con `--inplace`, el archivo `.ipynb` se actualiza con los resultados de la ejecución más reciente.** Combinado con `--ExecutePreprocessor.allow_errors=False`, levanta excepciones visibles si una celda falla.

**Implicaciones:**

- El `.ipynb` debe ser **idempotente** (no asumir estado de variables de celdas anteriores; cada celda restablece imports y configuraciones necesarias).
- Tests offline en JupyterLab antes de lanzar la corrida final en tmux. JupyterLab corre como sesión `tmux jupyter` con `jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''`.
- El SSH del operador hace `ssh -L 8888:localhost:8888 root@<INSTANCE_IP>` para acceder a JupyterLab desde Win11.

**Referencias clave:** nbconvert docs (`nbconvert.readthedocs.io/`).

---

### D26 — Roboflow SDK `roboflow-python` >=1.1.27 con fix del bug `location` (sustituye D19)

**Estado:** VINCULANTE | **Ronda:** R5 | **Sustituye:** D19 (workaround manual del bug `location`) | **Dependencias:** D3

**Decisión:** el dataset 1-B de Roboflow (`embebidos3/waste-3class-lwld8` v1-B, formato `yolov8`) se descarga al container Vast.ai usando el SDK oficial `roboflow-python >= 1.1.27`. NO se usa la REST API cruda.

**Razón:**

1. **El SDK aplica el fix oficial del bug `location: null`** (issue GitHub Roboflow #473). Antes del fix, la respuesta JSON del endpoint de descarga venía con el campo `location` como `null` en lugar de la URL S3 firmada — workaround manual era hacer un segundo POST y parsear la respuesta otherwise.
2. **El SDK incluye lógica de retry y manejo de errores transitorios** que la REST API cruda no.
3. **Hay un workaround histórico para un segundo bug del Roboflow Python SDK** que aparecía con datasets de >3 clases originales filtrados con Modify Classes: el SDK reportaba clases "fantasma" (clases que se eliminaron). El fix es exportar con `--format yolov8` (no `yolov8s` ni `yolov8-obb`) y, después de la descarga, validar manualmente que `train/labels/*.txt` solo contienen IDs de clase 0, 1, 2 (plastic, glass, paper).

**Implicaciones:**

- API key Roboflow va por env var `ROBOFLOW_API_KEY`.
- El dataset descargado tiene estructura `train/`, `val/`, `test/` con subdirectorios `images/` y `labels/`. El archivo `data.yaml` describe las 3 clases.
- Es responsabilidad del bootstrap verificar el split: si Filter Null eliminó más imágenes de lo esperado (>3 000 imgs), hay que revisar la matriz de co-ocurrencia clase-imagen.

**Referencias clave:** Roboflow Python SDK (`github.com/roboflow/roboflow-python`).

**Verificación 2026-05-14 (ronda /investiga + lectura source):** el bug del argumento `location` en `version.download(...)` **persiste en releases ≥1.3.x** (verificado leyendo `version.py` y `dataset.py` de la rama main; release 1.3.9 del 2026-05-07 no incluye fix). Mitigación: el notebook implementa una cascada de 3 estrategias (`location` directo → `download()` sin location + `shutil.move` → 3 retries con backoff exponencial), y NO se setea la env var `DATASET_DIRECTORY` (entra en conflicto con `location` y deja ambas ubicaciones vacías). Esta cascada vive en el notebook, no en el SDK; el SDK sigue siendo la dependencia oficial pero su contrato no se asume confiable para `location`.

---

### D27 — Tailscale `--accept-dns=false --ssh` como workaround del bug DNS ARM64 #14902

**Estado:** VINCULANTE (vigente para acceso remoto a Nano) | **Ronda:** R7 | **Sustituye:** ninguna | **Dependencias:** ninguna

**Decisión:** la Nano se conecta a Tailscale con la combinación de flags `sudo tailscale up --accept-dns=false --ssh`. La flag `--accept-dns=false` previene que Tailscale toque `/etc/resolv.conf` (origen del bug #14902); la flag `--ssh` habilita auth SSH federada (Tailscale verifica identidad sin necesidad de gestionar llaves manuales).

**Razón:**

1. **El bug Tailscale #14902 es severo en ARM64 + kernel 4.9.x.** Síntoma: al boot, `tailscaled` arranca y los logs muestran `dns: [rc=unknown ret=direct]` + `logtail: dial "log.tailscale.io:443" failed: no DNS fallback candidates remain`. Sin DNS, Tailscale no puede contactar el control plane. Workaround manual `sudo systemctl restart tailscaled` funciona pero el bug puede reaparecer en cualquier link change.
2. **`--accept-dns=false` evita el bug por completo** porque Tailscale NO toca `/etc/resolv.conf`. Trade-off: pierdes MagicDNS (`nano.tailnet.ts.net` no resuelve), tienes que usar la IP `100.x.x.x` directa o un alias SSH local. Para nuestro caso es aceptable porque solo necesitamos el túnel (no DNS in-tailnet) y el alias SSH ya está en `~/.ssh/config` apuntando a `100.100.166.120`.
3. **`--ssh` activa Tailscale SSH** — el daemon de Tailscale acepta conexiones SSH entrantes directamente, sin depender del `sshd` del sistema. Esto da auth federada (cualquier device autenticado en el tailnet puede conectarse sin llaves manuales). Bonus: si la Win11 también tiene Tailscale, `ssh nano` funciona sin password ni clave.
4. **El kernel 4.9-tegra NO tiene módulo WireGuard.** Tailscale detecta esto y automáticamente usa `wireguard-go` (userspace), con ~15-25% más overhead CPU pero estable. No requiere intervención manual.

**Implicaciones:**

- ACL en modo `check`: cada device tiene que re-autenticar periódicamente vía URL del navegador. Nicolas Cuaran administra los devices del tailnet — el operador es cliente.
- Si en algún momento se requiere MagicDNS (improbable para este proyecto), habría que migrar a self-hosted Headscale en el VPS Contabo, ver D28 — pero **descartado por el usuario el 2026-05-13** ("descartar AP-mode" implica también descartar la complejidad de Headscale).

**Referencias clave:** Issue Tailscale #14902, foro NVIDIA #184764 (WireGuard kernel module broken en k4.9-tegra).

---

### D29 (nueva, no estaba en CONSOLIDADO original) — Track A descartado en sesión 2026-05-13

**Estado:** VINCULANTE | **Ronda:** sesión interactiva 2026-05-13 | **Sustituye:** D1, D10, D12 (los deja inactivos pero no obsoletos en sentido histórico)

**Decisión:** el proyecto se enfoca exclusivamente en Track B. Las decisiones D1, D10 y D12 (relacionadas con TF 2.15 + TFOD API + TFLite_Detection_PostProcess + gates 1-2 de TFLite) quedan **inactivas** — no se ejecutarán, pero se mantienen en el ledger histórico para el caso futuro de retomar dual-track.

**Razón:** ver §1.3. En resumen: Track B aprovecha mejor el hardware de la Nano (GPU Maxwell), Track A tiene más puntos frágiles para el calendario disponible (QAT, TFLite_Detection_PostProcess, gates de validación TFLite con runtime 2.5), y los benchmarks reproducibles públicos respaldan Track B.

**Implicaciones:**

- No se genera la versión 1-A del dataset Roboflow (320×320 `tfrecord`). Solo 1-B (416×416 `yolov8`).
- No se instalan TF 2.15, TFOD API, Pillow<10, protobuf==3.20 en el container Vast.ai.
- Los gates 1 y 2 del pipeline de validación (TFLite op_version + carga Coral wheel CP38) no se ejecutan. Solo se ejecutan Gate 3 (ONNX ops blacklist) y Gate 4 (Polygraphy NGC).
- El bootstrap script se simplifica: ~40 % menos pasos.

**Referencias clave:** instrucción explícita del usuario el 2026-05-13: "necesito un handoff para priorizar la exploración preliminar, planeación e investigación de la implementación del training en track B solamente (dado que este track aprovecha mejor las posibilidades de la jetson nano)".

---

### D30 (nueva, sesión 2026-05-13) — SSH alias `nano` apuntando a Tailscale `100.100.166.120` + dependencia `bash -lc` para PATH completo

**Estado:** VINCULANTE | **Ronda:** sesión interactiva 2026-05-13 | **Sustituye:** ninguna

**Decisión:** el alias SSH canónico para acceder a la Nano desde Win11 es `nano`, definido en `C:\Users\mitgar14\.ssh\config` apuntando a la IP Tailscale `100.100.166.120`. Para invocar comandos no-interactivos que requieren binarios instalados en `~/.local/bin/` (como `uv`), hay que usar `ssh nano 'bash -lc "CMD"'` — la flag `-l` fuerza shell de login que carga `~/.profile`.

**Razón:**

1. **Tailscale `--ssh` ya hace el trabajo de NAT traversal y auth federada.** No se necesita IP local de la red UAO ni reverse SSH ni nada más.
2. **El bug de SSH no-login + PATH** no es bug de la Nano, es comportamiento estándar POSIX. `~/.profile` solo se carga en login shells (`bash -l` o ssh interactivo con `-t`). En no-login shells (`ssh user@host 'CMD'`), `~/.profile` se ignora.
3. **uv vive en `/home/jetson/.local/bin/uv`** (musl standalone) y Nicolas Cuaran exportó `PATH=$HOME/.local/bin:$PATH` en `~/.profile`, no en `~/.bashrc`. Por eso solo se ve en login shells.

**Implicaciones:**

- Scripts que invoquen comandos remotos por SSH deben prefijar con `bash -lc` o expandir el PATH manualmente.
- Si en el futuro se mueve la exportación de PATH a `~/.bashrc`, este workaround deja de ser necesario — pero NO se toca por ahora porque el setup de Nicolas Cuaran funciona y no hay razón para perturbarlo.

**Referencias clave:** investigación empírica de la sesión 2026-05-13 (saved as negative-constraint memory `97dc5e7c-c2e0-40f4-8313-8ddcea0366c2` en mnemon).

---

## 3. Estado actual validado empíricamente (2026-05-13)

Esta sección documenta **qué se hizo en la sesión interactiva del 2026-05-13** y, por ende, qué está validado.

### 3.1 Acceso SSH operativo

- SSH a la Nano vía Tailscale: ✅ validado.
- Alias `nano` en `C:\Users\mitgar14\.ssh\config`: ✅ creado y validado.
- `uv --version` en login shell: ✅ `0.11.14` (binary musl standalone).
- `nvcc --version` (CUDA en Nano): ✅ `Cuda compilation tools, release 10.2, V10.2.300`.
- `python3.6 -c 'import tensorrt; print(tensorrt.__version__)'`: ✅ `8.2.1.8`.

### 3.2 Diagnóstico hardware confirmado

`tegrastats` corrido durante 3+ minutos de cómputo sostenido:

- RAM idle: 1.4 GB / 3.9 GB.
- RAM peak durante TRT engine build: 1.5 GB / 3.9 GB.
- SWAP: 267 MB / 1.9 GB.
- CPU @ 1224 MHz (frequency governor, no throttling).
- Temperatura PLL: idle 28°C, peak 34°C → **margen térmico amplio**, no hay throttling en condiciones de banco.
- GPU @ 31-32°C estable.
- Power mode: `nvpmodel -q` → `MAX-N` (10W mode, todos los cores activos).

### 3.3 Demo 1 — MNIST TRT FP16 (smoke test mínimo)

Modelo: MNIST CNN preentrenado, exportado a ONNX por NVIDIA samples (`/usr/src/tensorrt/data/mnist/mnist.onnx`).

Comando: `trtexec --onnx=mnist.onnx --fp16 --saveEngine=/tmp/mnist_fp16.engine`.

Resultados:
- Engine build time: **21.02 s**.
- Latency: min = 0.234 ms, max = 1.73 ms, mean = 0.321 ms, median = 0.318 ms, p99 = 0.36 ms.
- Engine file size: **46 KB**.

**Conclusión:** la pipeline básica trtexec → engine → inferencia está operativa en la Nano. Tiempos consistentes con baseline esperado para Maxwell sm_53.

### 3.4 Demo 2 — ResNet50 TRT FP16 (modelo intermedio, smoke test medio)

Modelo: ResNet50 estándar ONNX (descargado de `https://github.com/onnx/models`).

Comando: `trtexec --onnx=resnet50.onnx --fp16 --saveEngine=/tmp/resnet50_fp16.engine`.

Resultados:
- Engine build time: **173.68 s** (~2 min 54 s).
- Latency: min = 29.07 ms, max = 30.36 ms, mean = 29.33 ms, median = 29.31 ms, p99 = 30.35 ms.
- Engine file size: **64 MB**.

**Conclusión:** modelos del orden de 25 M parámetros se compilan en <3 min y corren a ~34 FPS. ResNet50 NO es target de Track B, pero validar latencia esperada para modelos de ese rango de peso da confianza para YOLOv8n.

### 3.5 Demo 3 — YOLOv8n TRT FP16 (validación del pipeline TARGET)

Modelo: YOLOv8n preentrenado en COCO, exportado a ONNX localmente en Win11 con `uvx ultralytics export model=yolov8n.pt format=onnx imgsz=416 opset=11 simplify=True dynamic=False nms=False`.

Validaciones previas al deploy:
- `onnx.checker.check_model(model)`: ✅ OK.
- IR version: 6, opset 11 (FP32 input).
- Input: `('images', [1, 3, 416, 416])`.
- Output: `('output0', [1, 84, 3549])` (3549 anchors × 84 channels = 4 box coords + 80 class scores COCO).
- 234 nodes, top ops: Conv (64), Sigmoid (58), Mul (58), Concat (17), Split (8), Add (8), Reshape (8), MaxPool (3), Resize (2), Transpose (2), Slice (2), Sub (2), Softmax (1), Div (1).
- **Ninguna op en blacklist TRT 8.2.** ✅ Gate 3 efectivamente pasado (sin script automatizado, pero verificable manualmente).

Transferencia a Nano vía `scp -P <PORT> yolov8n.onnx jetson@100.100.166.120:/tmp/`. SHA256: `d43d503acfa7818d0bfca399e1ece47e` (también verificable post-copy con `md5sum`).

Comando: `trtexec --onnx=/tmp/yolov8n_demoA.onnx --fp16 --saveEngine=/tmp/yolov8n_fp16.engine --workspace=1024`.

Resultados (Nano TRT 8.2.1, sm_53, FP16):
- Engine build time: **500.22 s** (~8 min 20 s).
- Throughput: **39.97 qps** (equivale a ~25 ms/frame).
- Latency: min = 24.87 ms, max = 25.97 ms, **mean = 25.01 ms**, median = 24.96 ms, p99 = 25.94 ms.
- End-to-end host latency: min = 24.89 ms, mean = 25.02 ms.
- Enqueue time: mean = 7.15 ms.
- H2D latency: mean = 0.20 ms.
- GPU compute time: mean = 24.68 ms.
- D2H latency: mean = 0.12 ms.
- Engine file size: **14.9 MB**.

**Conclusión crítica:** YOLOv8n FP16 416×416 en este hardware específico (Jetson Nano B01 con JetPack 4.6.1, L4T R32.7.1) corre a **~40 FPS** end-to-end, **superando la predicción de Nature 2024 Tabla 4 (30 FPS)**. El margen sobre el target del proyecto (≥10 FPS sostenido) es **4× mayor**, dejando holgura cómoda para concurrencia con el control de servos PCA9685.

### 3.6 Limpieza post-validación

Después de las 3 demos, los archivos en `/tmp/` fueron limpiados explícitamente con `rm` para liberar espacio en `tmpfs`:
- `/tmp/yolov8n_demoA.onnx` (12.7 MB) → eliminado.
- `/tmp/yolov8n_fp16.engine` (14.9 MB) → eliminado.
- `/tmp/mnist_fp16.engine` (46 KB) → eliminado.
- `/tmp/resnet50_fp16.engine` (64 MB) → eliminado.

### 3.7 Lo que NO se validó en sesión 2026-05-13

| Componente | Status validación | Razón |
|---|---|---|
| Cámara C920 conectada físicamente | ❌ no conectada | C920 OG todavía no físicamente acoplado a la Nano |
| Pipeline GStreamer con cámara real | ❌ no probado | depende del anterior |
| PCA9685 vía I²C | ❌ no probado | hardware no instalado aún |
| Bus I²C de la Nano (`/dev/i2c-1` vs otro) | ❌ no verificado | depende del anterior |
| Bootstrap Vast.ai completo end-to-end | ❌ no ejecutado | se documentó pero no se corrió en esta sesión |
| Training YOLOv8n sobre dataset 1-B | ❌ no realizado | siguiente paso |
| Polygraphy Gate 4 dry-run en x86 | ❌ no ejecutado | Gate 3 pasó manualmente, Gate 4 pendiente |
| Inferencia real (no `trtexec --iterations`) sobre frames de cámara | ❌ no probado | depende de la cámara conectada |
| NMS en CPU NumPy con outputs reales | ❌ no implementado | depende del anterior |
| Persistencia HF Hub durante un training real | ❌ no testeado | depende del bootstrap completo |
| Auto-destroy de instancia Vast.ai | ❌ no testeado | depende del bootstrap completo |
| Roboflow SDK descarga 1-B en Vast.ai | ❌ no testeado | depende del bootstrap completo |

---

## 4. Stack Track B (resumen consolidado, no ejecutivo)

| Capa | Decisión | Razón resumida | Decisión ID |
|---|---|---|---|
| Modelo | YOLOv8n (nano) | Solo nano cabe en VRAM Maxwell con latencia <30 ms | D2 |
| Resolución | 416×416 | Sweet spot mAP/FPS confirmado en 3 fuentes independientes | D2 |
| Framework training | Ultralytics ≥8.4.46, <8.5 | Estable, fix calibración INT8 incluido, API compatible | D2 |
| Dataset | Roboflow `embebidos3/waste-3class-lwld8` v1-B | 3 clases (plastic/glass/paper), Fit-black 416×416, augmentations Basic 3× | D26 |
| Cloud training | Vast.ai RTX 4090 + container CUDA 12.4/Ubuntu 22.04/Py 3.10 | Solo entorno con sudo + sin timeouts + GPU dedicada | D3, D17 |
| Gestor Python | uv | Regla global del usuario + 10-100× más rápido que pip | D5 |
| Venv en cloud | `/opt/venv/trackb` registrado como ipykernel | Aísla conflictos transitivos numpy/protobuf | D6 |
| Workflow training | `.ipynb` vía `jupyter nbconvert --execute --inplace` dentro de tmux | Preserva outputs incrustados para evidencia IEEE | D18 |
| Persistencia | HF Hub `mitgar14/embebidos-3-models` (privado) + `CommitScheduler(every=5)` | Filesystem Vast.ai se borra al destroy | D8 |
| Auto-destroy | cron watchdog + última celda + `$VAST_CONTAINERLABEL` | Doble protección anti-facturación por olvido | D11 |
| Export modelo | ONNX opset 11, `dynamic=False`, `simplify=True`, `nms=False` | Compatibilidad estricta con TRT 8.2 + NMS roto en Maxwell | D2 |
| Validación pre-deploy | Gate 3 (ONNX ops blacklist) + Gate 4 (Polygraphy NGC 21.11) | Detecta >95% de errores antes de los 8 min de build en Nano | D13 |
| Compilación engine | trtexec en la Nano con `--fp16 --workspace=1024` | Engines son GPU-arch + TRT-version específicos | D13, D14 |
| Quantización engine | FP16-only | Maxwell sin `dp4a`: INT8 NO acelera pero SÍ degrada mAP | D14 |
| Postproc Nano | NMS tri-path: V0 `cv2.dnn.NMSBoxes` CPU (default), V1 `EfficientNMS_TRT` (smoke test), V2 `BatchedNMSDynamic_TRT` (fallback) | V1 y V2 sí están en el binary JP 4.6.1 (fix #1538 commit `3235cc2`, jul-2021); Polygraphy no corre en JP 4.6.1 | D13, D14 |
| Inferencia runtime Nano | TensorRT Python bindings + cuda-python 11.0 | ORT + TRT EP requiere CUDA 11.4 (Nano tiene 10.2) | (implícito) |
| Acceso remoto Nano | Tailscale `--accept-dns=false --ssh` | Bug DNS ARM64 + WireGuard kernel module broken en k4.9-tegra | D27, D30 |
| Captura cámara | OpenCV `VideoCapture` con pipeline GStreamer `v4l2src` + MJPG decode | Cámara C920 OG, MJPG mandatorio para evitar saturación USB 2.0 | (heredada de investigación 2026-05-10) |
| Actuadores | PCA9685 vía I²C, bus por verificar | Servos para clasificación post-detección | (out of scope inmediato) |

---

## 5. Pipeline conceptual end-to-end (sin código)

Esta sección describe **el flujo de datos** desde el origen (cámara o dataset Roboflow) hasta el output final (clasificación on-device + acción servo). NO es manual operacional — para los comandos canónicos, ver el bootstrap script consolidado en el repo de código.

### 5.1 Flujo training (Vast.ai)

**Etapa 1 — Provisioning Vast.ai:**
- Operador en Win11 ejecuta `vastai search offers` con filtros (RTX 4090, <0.50 USD/hr, num_gpus=1, cuda_max_good >= 12, reliability > 0.95, inet_down > 500 Mbps), ordenado por precio.
- Lanza la instance con `vastai create instance <OFFER_ID> --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 --disk 50 --label "embebidos-3-trackb" --ssh`.
- Espera ~1 minuto a que el daemon SSH esté listo.
- Conecta con `ssh -p <PORT> -L 8888:localhost:8888 root@<INSTANCE_IP>`.

**Etapa 2 — Bootstrap:**
- Dentro del container, ejecuta el script `bootstrap.sh` que:
  1. Instala paquetes del sistema vía `apt`: `tmux`, `git`, `curl`, `wget`, `unzip`, `nano`, `ffmpeg`, `libgl1-mesa-glx`, `libglib2.0-0`, `python3-pip`, `python3-venv`, `protobuf-compiler`.
  2. Instala `uv` con `curl -LsSf https://astral.sh/uv/install.sh | sh`.
  3. Clona el repo del proyecto en `/workspace/embebidos-3`.
  4. Crea el venv Track B con `uv venv /opt/venv/trackb --python 3.10`.
  5. Instala el stack Track B: Ultralytics 8.4.46 + numpy<2 + onnx + onnxslim + onnxruntime-gpu + huggingface_hub + wandb + roboflow + jupyter + ipykernel.
  6. Registra ipykernel "Track B (YOLOv8)".
  7. Instala el cron watchdog auto-destroy.
  8. Lanza JupyterLab en sesión tmux `jupyter`, port 8888, sin token.

**Etapa 3 — Descarga dataset:**
- Notebook Track B: ejecuta `from roboflow import Roboflow; rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY']); dataset = rf.workspace("embebidos3").project("waste-3class-lwld8").version(1).download("yolov8")`.
- Verifica `data.yaml` (3 clases: plastic, glass, paper), número de imágenes en `train/`, `val/`, `test/`.

**Etapa 4 — Configuración HF Hub:**
- `scheduler = CommitScheduler(repo_id="mitgar14/embebidos-3-models", folder_path="/workspace/embebidos-3/runs", every=5, private=True, token=os.environ['HF_TOKEN'])`. Corre como thread separado.

**Etapa 5 — Training:**
- `from ultralytics import YOLO; model = YOLO("yolov8n.pt"); model.train(data="/workspace/embebidos-3/datasets/.../data.yaml", epochs=100, imgsz=416, batch=16, device=0, project="/workspace/embebidos-3/runs", name="trackb-baseline", patience=20, save_period=5, plots=True)`.
- Cada 5 epochs el `save_period` deja un `weights/epoch{N}.pt` en `runs/`, que el CommitScheduler sube al HF Hub.
- Hiperparámetros Ultralytics defaults: `mosaic=1.0`, `mixup=0.0` (considerar `mixup=0.15` post-eval base), `degrees=0` (Roboflow ya aplicó rotation), `translate=0.1`, `scale=0.5`, `fliplr=0.5`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`.

**Etapa 6 — Evaluación:**
- `metrics = model.val(data=..., imgsz=416, split="test")` produce mAP@0.5, mAP@0.5:0.95, P, R por clase.
- Sample predictions sobre 20 imágenes random del test split, render con bounding boxes, save a `runs/.../predictions/`.

**Etapa 7 — Export ONNX:**
- `model.export(format="onnx", imgsz=416, opset=11, simplify=True, dynamic=False, nms=False)`.
- Output: `runs/.../weights/best.onnx`.

**Etapa 8 — Gate 3 (en el container):**
- Script Python valida: opset=11, ir_version<=10, ninguna op en blacklist TRT 8.2, ConstantOfShape dtype FP32, no NonMaxSuppression.

**Etapa 9 — Gate 4 (Docker NGC):**
- `docker run --rm --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:21.11-py3 bash -c "pip install -q polygraphy onnx && polygraphy run /workspace/best.onnx --onnxrt --trt --atol 1e-2 --rtol 1e-2 --input-shapes images:[1,3,416,416]"`.
- Si exit code == 0 y la comparación numérica ONNX-Runtime vs TRT está dentro de tolerancia → ✅ listo para Nano.

**Etapa 10 — Upload artefactos a HF Hub:**
- `scheduler.trigger()` para forzar sync final. `scheduler.stop()` para shutdown limpio.
- Artefactos en `runs/trackb-baseline/`: `weights/best.pt`, `weights/best.onnx`, `train_batch{N}.jpg`, `val_batch{N}.jpg`, `results.csv`, `confusion_matrix.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png`.

**Etapa 11 — Auto-destroy:**
- Última celda del notebook: `subprocess.run(["vastai", "destroy", "instance", os.environ["VAST_CONTAINERLABEL"].replace("C.", "")])`.

### 5.2 Flujo deploy (Jetson Nano)

**Etapa 1 — Descarga del ONNX validado:**
- Operador en Win11 ejecuta `huggingface-cli download mitgar14/embebidos-3-models weights/best.onnx --local-dir ./artifacts/` (vía SDK Python o CLI).
- Transferencia a Nano: `scp ./artifacts/best.onnx jetson@100.100.166.120:/home/jetson/embebidos-3/models/best.onnx`. Verificar SHA256.

**Etapa 2 — Compilación engine en Nano:**
- SSH a Nano con `ssh nano`. Sesión tmux opcional pero recomendada (el build tarda ~8 min y SSH puede caerse).
- `cd /home/jetson/embebidos-3 && trtexec --onnx=models/best.onnx --fp16 --workspace=1024 --saveEngine=models/best.engine --verbose 2>&1 | tee logs/trt_build_$(date +%Y%m%d_%H%M%S).log`.
- Esperar ~500 s. Verificar que el engine no tenga warnings críticos en el log.

**Etapa 3 — Smoke test latencia:**
- `trtexec --loadEngine=models/best.engine --shapes=images:1x3x416x416 --iterations=100`.
- Esperar mean latency en rango 24-28 ms (consistente con la validación 2026-05-13).

**Etapa 4 — Inferencia con cámara real:**
- Pipeline:
  1. Cargar engine con `tensorrt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)`.
  2. Crear `execution_context = engine.create_execution_context()`.
  3. Allocate buffers GPU (`cuda.mem_alloc(...)`) e host (`np.empty(..., dtype=np.float32)`).
  4. Abrir cámara con `cv2.VideoCapture` + pipeline GStreamer `v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink`.
  5. Loop:
     - Capturar frame BGR.
     - Preprocess: resize a 416×416 (puede ser Fit-black con `cv2.copyMakeBorder` para consistencia con training; valor de padding 0 según D26).
     - Normalize: `/255.0`, HWC→CHW, batch dim.
     - `cuda.memcpy_htod_async(d_input, inp, stream)`.
     - `execution_context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)`.
     - `cuda.memcpy_dtoh_async(h_output, d_output, stream)`. `stream.synchronize()`.
     - Postprocess outputs (cx, cy, w, h, conf, c0, c1, c2):
       - Filter por `conf > 0.5`.
       - Convert cx,cy,w,h → x1,y1,x2,y2.
       - Multiply scores: `score_final = conf * max(class_scores)`.
       - `argmax(class_scores)` para class_id.
       - NMS CPU: `cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=0.45)`.
     - Para cada detección que sobrevive NMS:
       - Renderizar bbox + label en el frame (cv2.rectangle + cv2.putText).
       - Si la clase es la target del bin actual de la banda, enviar comando al PCA9685 vía I²C para activar el servo correspondiente.
     - Mostrar frame con `cv2.imshow` (durante demo, no en producción) o guardar a disco (auditoría post-demo).

**Etapa 5 — Integración con servos:**
- Out of scope para el handoff inmediato. Pendiente verificar bus I²C, instalar `python3-smbus` o `Adafruit-PCA9685`, mapeo angle → tipo de residuo.

### 5.3 Flujo demo final (2026-05-26)

- La Nano arranca con la cámara C920 acoplada físicamente y el PCA9685 conectado por I²C.
- El operador llega con su Win11 y se conecta a la red WiFi que tenga (UAO, hotspot móvil, o **directamente a la Nano si fuera necesario** — pero AP mode descartado en sesión 2026-05-13, así que asume WiFi disponible o se prepara hotspot móvil del celular del operador como backup).
- `ssh nano` vía Tailscale → start del binario de inferencia que arranca el loop completo cámara → engine → servos.
- Si Tailscale falla durante la demo (improbable pero posible), plan B: ejecutar el binario directamente en la Nano con el monitor HDMI conectado.

---

## 6. Aprendizajes de la sesión interactiva 2026-05-13 (lo que se perdería sin documentar)

### 6.1 SSH no-login no carga PATH de uv

**Síntoma observado:** ejecutar `ssh nano 'uv --version'` desde Win11 devolvió `bash: uv: command not found`. Inicialmente se interpretó como "uv no está instalado", lo cual fue contradicho por el usuario: "Tené en cuenta que en la Jetson se tiene uv instalado".

**Diagnóstico empírico:** el binario está en `/home/jetson/.local/bin/uv`, exportado en `PATH` vía `~/.profile`. SSH no-login (`ssh user@host 'CMD'`) NO carga `~/.profile` por defecto. Solo lo cargan login shells (`ssh -t` interactivo o `bash -l`).

**Solución canónica:** usar `ssh nano 'bash -lc "CMD"'` o, alternativamente, prefijar el comando con `PATH=$HOME/.local/bin:$PATH CMD`. Adoptamos la primera por consistencia.

**Lección general:** cualquier herramienta instalada en `~/.local/bin/` con un installer userspace (musl standalone, pipx, rustup) tiene este pitfall. La regla "uv como gestor por defecto" se vuelve operacionalmente más frágil cuando uno se conecta remoto vía SSH no-login.

### 6.2 Ultralytics imprime `requirements:` en rojo, NO es error

**Síntoma observado:** durante el export ONNX local (`uvx ultralytics export model=yolov8n.pt format=onnx imgsz=416 opset=11`), Ultralytics emitió mensajes en color rojo del estilo `requirements: Ultralytics requirement ['onnxruntime'] not found, attempting AutoUpdate...`. El usuario interpretó esto como un error: "Pero hubo un error con ONNX: mira completo los logs para confirmar".

**Diagnóstico empírico:** re-ejecutar el export sin `Select-String` filter mostró el log completo. Ultralytics imprime `requirements:` en rojo por convención visual, no porque sea error. El export terminó con `ONNX: export success` + `Export complete (1.2s)`. La validación posterior con `onnx.checker.check_model(model)` retornó OK.

**Lección general:** filtrar logs con grep/Select-String puede ocultar contexto crítico. Ante duda sobre el éxito de una operación, ver el log COMPLETO antes de juzgar.

### 6.3 Falsa alarma del sub-agente mnemon `replaced_id`

**Síntoma observado:** al guardar una memoria nueva del CONSOLIDADO en mnemon, el sub-agente reportó que la nueva memoria `ace95f8a` reemplazó a una previa `207d435d`. Una verificación posterior demostró que `207d435d` estaba INTACTA y `ace95f8a` nunca existió (mnemon había hecho `skipped DUPLICATE`, no `replaced`).

**Diagnóstico empírico:** el sub-agente interpretó incorrectamente el output de mnemon. Re-ejecución con flag `--no-diff` produjo un ID válido nuevo `9a29b10d-89a7-4bd7-9453-d28487d6ce0f`.

**Lección general:** los sub-agentes de mnemon pueden reportar IDs ficticios si hay duplicación detectada. Siempre verificar con `mnemon recall "<query>" --limit 5` después de un remember para confirmar que la memoria efectivamente se persistió.

### 6.4 Validación TRT en hardware real fue +33% mejor que la predicción del paper

**Predicción de Nature Sci Reports 2024 Tabla 4:** YOLOv8n FP16 TRT en Jetson Nano @ 416×416 = **30 FPS**.

**Medición empírica 2026-05-13 (este proyecto):** YOLOv8n FP16 TRT en Jetson Nano @ 416×416 = **39.97 FPS** (latencia mean 25.01 ms).

**Diferencia:** +33% sobre la predicción. Posibles causas:
1. La Nature 2024 puede estar reportando con un YOLOv8n fine-tuned (mas custom layers que el COCO baseline).
2. Nature 2024 usa JetPack 4.6.1 — coincide con nuestra Nano (verificado vía SSH 2026-05-14). No queda hipótesis pendiente sobre diferencia de versiones de JetPack como explicación del delta de FPS.
3. La Nano de este proyecto tiene `MAX-N` power mode activo (10W); la Nature 2024 no especifica modo.
4. La medición de Nature 2024 puede incluir overhead de preprocessing + render que `trtexec --iterations=100` no incluye.

**Lección general:** los benchmarks publicados son piso conservador, no techo. Para Track B este margen extra es bienvenido (10 FPS sustained target con margen 4× → margen efectivo casi 4× sigue).

### 6.5 La búsqueda de tunable `--workspace` es delicada

**Observación:** trtexec en YOLOv8n con `--workspace=1024` MB compila exitosamente sin OOM. Issue Ultralytics #14751 documenta casos donde `--workspace` muy grande (default puede ser 2-4 GB) provoca OOM en Nano por la RAM unificada.

**Recomendación operacional:** usar siempre `--workspace=1024` en la Nano. Si en una capa específica TRT requiere más workspace, va a aparecer en el verbose log con `Trying to allocate XXX MB`. En ese caso, bajar a 512 o evaluar quitar capas problemáticas (poco probable en YOLOv8n).

### 6.6 La memoria libre estable post-build es de ~2.5 GB

**Medición empírica 2026-05-13:** durante el TRT engine build de YOLOv8n (peak load):
- RAM total: 3.9 GB.
- RAM used: 1.5 GB.
- RAM free: 1.3 GB.
- buff/cache: 1.1 GB.
- RAM available: 2.7 GB.

**Implicación:** queda ~2.7 GB libres para el resto del pipeline (OpenCV captura, postproc NumPy, control I²C, render). Margen cómodo. Si en producción la inferencia + control I²C empieza a presionar, OpenCV puede liberar memoria configurando `cv2.VideoCapture` con bufrer size pequeño.

### 6.7 La temperatura nunca pasó de 34°C en compute sostenido

**Medición empírica 2026-05-13:** 3+ minutos de TRT engine build de YOLOv8n (CPU + GPU mix):
- PLL temp: 28°C → 34°C peak.
- CPU temp: 28.5°C → 32.5°C peak.
- GPU temp: 30°C → 31°C peak.
- Throttle status: never triggered.

**Implicación:** el cooling actual de la Nano (`pwm-fan` con `target_pwm=0`) ES SUFICIENTE para Track B en condiciones de banco. NO se necesita actualizar el cooler. Sin embargo, durante demo con 30+ minutos de inferencia continua + servos consumiendo CPU adicional, monitorear `tegrastats` para detectar throttling. Si aparece, activar el fan a `target_pwm=128` (50%) es trivial: `sudo sh -c 'echo 128 > /sys/devices/pwm-fan/target_pwm'`.

### 6.8 onnx checker pasa, pero hay que verificar manualmente los ops contra TRT 8.2 whitelist

El export ONNX local produjo:
- IR version: 6 (≤10 ✅).
- Opset: 11 (==11 ✅).
- Input: `('images', [1, 3, 416, 416])` (FP32 estático ✅).
- Output: `('output0', [1, 84, 3549])` (3549 anchors ✅).
- 234 nodos, 14 tipos de ops.

Top ops verificados:
- Conv (64): ✅ soportado FP32, FP16, INT8.
- Sigmoid (58): ✅ soportado FP32, FP16, INT8.
- Mul (58): ✅ soportado.
- Concat (17): ✅ soportado.
- Split (8): ✅ soportado.
- Add (8): ✅ soportado FP32, FP16, INT32.
- Reshape (8): ✅ soportado.
- MaxPool (3): ✅ soportado.
- Resize (2): ✅ soportado FP32, FP16. Hay que verificar `coordinate_transformation_mode` — no es `tf_half_pixel_for_nn` (no soportado en TRT 8.2), debería ser `half_pixel` o `pytorch_half_pixel`.
- Transpose (2): ✅ soportado.
- Slice (2): ✅ soportado.
- Sub (2): ✅ soportado.
- Softmax (1): ✅ soportado FP32, FP16.
- Div (1): ✅ soportado.

**Ninguna op de la blacklist** (GridSample, DFT, IsInf, IsNaN, MelWeightMatrix, STFT, SequenceInsert, CumSum, NonZero, RoiAlign, QLinearConv, QLinearMatMul, Reciprocal) está presente.

**Conclusión:** Gate 3 efectivamente pasado manualmente. Para reproducibilidad, hay que automatizar este check con un script Python como parte del notebook training.

### 6.9 El export ONNX agregó `onnxslim` simplifications automáticamente

Ultralytics 8.4.50 (versión que `uvx` resolvió, no la 8.4.46 fijada en D2 — pero está dentro del rango ≥8.4.46, <8.5) aplicó `onnxslim 0.1.93` automáticamente durante el export. Esta versión incluye el fix del bug del nodo `Slice` (`onnxslim>=0.1.34`), por lo que no se requiere intervención manual.

**Lección general:** Ultralytics maneja `onnxslim` internamente; no hay que invocarlo por separado. Pero verificar la versión post-export con `pip show onnxslim` para confirmar que es >= 0.1.34.

### 6.10 `uvx` es la forma canónica de invocar Ultralytics one-off en Win11

Para hacer el export ONNX local sin contaminar un venv permanente:
- `uvx ultralytics export model=yolov8n.pt format=onnx imgsz=416 opset=11 simplify=True dynamic=False nms=False`.

`uvx` descarga Ultralytics + deps en un venv efímero, ejecuta, y limpia al terminar. No persiste nada en la instalación global.

**Lección general:** para experimentos one-off de exports/conversiones de modelo, `uvx <tool> <args>` evita contaminar el sistema y es trivialmente reproducible.

---

## 7. Decisiones aún abiertas para Track B

Estas son cuestiones que la siguiente sesión debe abordar antes o durante el training:

### 7.1 Hiperparámetros de training

| Decisión | Opciones | Recomendación inicial | A validar |
|---|---|---|---|
| Epochs | 50 vs 100 vs 200 con early stopping | **100 con `patience=20`** | Si la curva val_loss se estabiliza antes de epoch 50, ahorrar tiempo de cómputo |
| Batch size | 16 vs 32 vs 64 | **16** (cabe en VRAM 4090 con headroom) | Beneficio marginal de 32+ para nano model; mantener 16 |
| Optimizer | SGD vs AdamW | **default Ultralytics (SGD)** | Ultralytics empíricamente prefiere SGD para detection |
| LR schedule | cosine vs step | **default cosine** | No tocar |
| `mixup` | 0.0 vs 0.15 | **0.0** (baseline), `0.15` (ablation 2) | Crasto 2024: +11.3 pp con `mosaic=1.0 + mixup=0.3` en COCO-ZIPF, pero gain decrece en datasets de 10k imgs |
| `mosaic` | 0.0 vs 1.0 | **1.0 (default)** | Confirmado +17% mAP en TACO (AliHamzaAzam) |
| Class weights | sí vs no | **no** | Crasto 2024: -1.9 pp con class weights tuned en single-stage; mosaic mitiga imbalance suficiente |
| Validation split | 20% vs 30% | **20% (default Roboflow 70/20/10)** | Suficiente con dataset de 10k imgs |

### 7.2 Dataset

**Estado actual:** version 1-B del dataset Roboflow está exportada y disponible.

**Lo que falta validar:**
1. **Conteo real post-Filter Null:** esperado ~10 439 imágenes. Verificar con `wc -l` sobre `train/labels/*.txt`.
2. **Matriz de co-ocurrencia clase-imagen:** plastic = 7 128 bbox, glass = 1 927, paper = 1 384 (5.15× imbalance). Si Filter Null eliminó >3 000 imágenes (más de lo esperado), revisar la matriz para entender por qué.
3. **Calidad anotaciones aleatorias:** revisar 30 imágenes random del val split visualmente, confirmar bboxes coherentes con el contenido. Si >10% tienen anotaciones incorrectas, considerar re-anotar.

**Si fallan estas validaciones:** considerar **dataset suplementario**. Opciones documentadas en la investigación 2026-05-05:
- Roboflow `material-identification/garbage-classification-3` (10 000 imgs CC BY 4.0).
- Domain adaptation con k=5 o k=15 muestras few-shot.

Pero: **el camino menor riesgo es no tocar nada y entrenar primero**. Si la baseline da mAP@0.5 > 0.7, no necesitamos más datos. Si da <0.5, considerar dataset suplementario.

### 7.3 Ablations propuestas (orden de prioridad)

**Ablation 1 (baseline obligatoria):**
- 100 epochs, defaults Ultralytics, imgsz=416, dataset 1-B intacto.
- Métricas: mAP@0.5, mAP@0.5:0.95, P/R por clase, F1 curve.
- Estimación: 2 h training + 0.5 h eval = 2.5 h × 0.40 USD/h = **1 USD**.

**Ablation 2 (mixup ON, opcional):**
- 100 epochs, `mixup=0.15`, todo lo demás igual.
- Comparar contra baseline. Si mejora mAP@0.5 por más de 1 pp y no degrada FPS post-export, adoptar.
- Estimación: 2 h training × 0.40 USD/h = **0.80 USD**.

**Ablation 3 (resolution 416 vs 640, opcional):**
- Generar Version 1-B-alt-640 (640×640 Fit-black) en Roboflow.
- Entrenar YOLOv8n con `imgsz=640` sobre esa versión.
- Comparar mAP + FPS en Nano (predicción Nature 2024: 416=30, 640=24).
- Estimación: 1 versión Roboflow extra + 3.5 h training × 0.40 = **1.40 USD + 1 versión Premium**.

**Ablation 4 (letterbox vs stretch — APORTE IEEE, opcional):**
- Generar Version 1-B-alt-stretch (Stretch 416 sin padding).
- Entrenar YOLOv8n con `imgsz=416` sobre esa versión.
- Comparar mAP contra Version 1-B (Fit-black 416).
- Esperado (Ultralytics #7454): letterbox > stretch por 4-5 pp.
- Estimación: 1 versión Roboflow extra + 2 h training × 0.40 = **0.80 USD + 1 versión Premium**.

**Ablation 5 (INT8 vs FP16 — opcional, solo si margen de tiempo):**
- Compilar engine INT8 con calibración (300-500 imgs val split).
- Comparar mAP + FPS contra FP16 baseline.
- Decisión adoptación binaria según D14 (FPS_INT8 ≥ FPS_FP16 × 1.10 Y mAP_INT8 ≥ mAP_FP16 − 5 pp).
- Estimación: 1 h training Nano (no Vast.ai) = **gratis pero 1 h de operador**.

**Presupuesto total estimado para Track B:**
- Solo Ablation 1: **1 USD**.
- Ablations 1+2: **1.80 USD**.
- Ablations 1+2+3: **3.20 USD + 1 versión Roboflow**.
- Ablations 1-4: **4 USD + 2 versiones Roboflow**.

Recomendación: empezar con Ablation 1 y reservar el resto para post-eval baseline.

### 7.4 Padding en inferencia: 0 vs 114

**Cuestión:** Roboflow Fit-black aplicó padding 0 (negro) en training. Ultralytics LetterBox interno aplica padding 114 (ImageNet mean) en `val`/`predict`. Si en producción Jetson usamos `Ultralytics.predict()` directo sobre el engine, hay mismatch.

**Opciones:**
1. **Aplicar `cv2.copyMakeBorder(value=(114,114,114))` en preprocess de Nano** para alinear con Ultralytics LetterBox interno. Riesgo: discrepancia con training (que usó padding=0).
2. **Aplicar `cv2.copyMakeBorder(value=(0,0,0))` en preprocess de Nano** para alinear con Roboflow Fit-black training. Riesgo: si en algún momento se llega a usar Ultralytics LetterBox, mismatch.

**Recomendación:** **opción 2** (padding=0) porque es consistente con el training. Documentar la decisión en el código de inferencia. Si en eval real aparece caída de mAP, considerar re-entrenar con augmentation que incluya padding aleatorio (0 + 114 + 255) para hacer el modelo robusto a ambos.

**Gap residual confirmado:** no hay paper publicado con ablación del efecto del padding value (0 vs 114) sobre INT8 calibration o FP16 inference accuracy. Esto es un GAP de literatura — si el ablation interno muestra delta significativo, es minor contribution publicable.

### 7.5 NMS thresholds

**Configuración default Ultralytics val:** `conf=0.001`, `iou=0.6`. Pero esos son para mAP@N computation, no para producción.

**Para deploy:** `conf=0.5` (filter weak detections), `iou_nms=0.45` (estándar industry).

**A validar empíricamente:** en banda real con C920, ¿qué confidence threshold da menor false positive rate? Probable que sea más alto (0.6-0.7) si la cámara tiene buena iluminación. Más bajo (0.3-0.4) si la iluminación es pobre y queremos prioritizar recall.

### 7.6 Frame rate target operacional

Hardware actual da ~40 FPS. Pero **no necesitamos 40 FPS**. El target del proyecto es 10 FPS sostenido. Razones para artificialmente limitar FPS:
- Reduce uso de CPU (postproc + render + servos compiten por CPU).
- Reduce temperatura GPU/CPU en demo larga.
- Reduce variabilidad latencia (más cómputo libre para responder a interrupts).

**Recomendación:** loop de inferencia a 15-20 FPS (intervalo de captura ~50-66 ms). Trivial de implementar con `time.sleep(...)` o `cv2.waitKey(...)`.

### 7.7 ¿Cuándo conectar la cámara C920 físicamente?

**Acción pendiente:** acoplar la C920 OG Rev 1 a la Nano vía USB.

**Una vez conectada, verificar empíricamente:**
1. `lsusb` muestra `046d:082d Logitech Webcam C920`.
2. `v4l2-ctl --list-devices` muestra `/dev/video0`.
3. `v4l2-ctl --device=/dev/video0 --list-formats-ext` confirma soporte MJPG 1280×720 @ 30 FPS.
4. Pipeline GStreamer `v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink` captura sin errores.
5. `cv2.VideoCapture(<pipeline>, cv2.CAP_GSTREAMER)` lee frames consistentemente.

**Si la C920 tiene Rev 1 (no Rev 3), el control de focus es `focus_auto` (V4L2 control, no `focus_automatic_continuous`).** Si rev 3, es lo contrario. Investigación 2026-05-10 cubre esto.

### 7.8 ¿Bus I²C para PCA9685?

**Acción pendiente:** conectar PCA9685 al bus I²C de la Nano (probable `/dev/i2c-1`).

**Una vez conectado, verificar:**
1. `i2cdetect -y 1` muestra el chip en alguna dirección (default `0x40`).
2. `python3 -c "import Adafruit_PCA9685; pwm = Adafruit_PCA9685.PCA9685(busnum=1)"` no falla.

Si la dirección es distinta de `0x40`, hay que pasarla explícitamente al constructor.

---

## 8. Gotchas críticos para Track B

Subconjunto del catálogo de gotchas del CONSOLIDADO original, filtrado a los que afectan Track B.

### 8.1 Vast.ai (G-VAST-*)

**G-VAST-01 — SSH falla con `Connection refused` justo tras `vastai create`**
- *Síntoma:* timeout o connection refused al SSH a la IP que muestra `vastai show instances`.
- *Causa:* el daemon SSH del container tarda 30-60 s en estar listo tras provisioning.
- *Fix:* esperar 1 minuto y reintentar. Si tras 3 minutos no responde, `vastai logs <ID>`.

**G-VAST-02 — `tmux: command not found` recién bootstrap**
- *Síntoma:* `bash: tmux: command not found` al primer comando.
- *Causa:* container base oficial Vast.ai NO incluye tmux.
- *Fix:* en `bootstrap.sh` la línea `apt-get install ... tmux ...` debe correr ANTES de cualquier `tmux new`.

**G-VAST-03 — `pip install` falla con `ImportError: cannot import name 'TYPE_CHECKING'` post-apt-upgrade**
- *Síntoma:* tras `apt upgrade` automático, pip queda roto.
- *Causa:* la actualización rompe el `pip` del sistema. uv evita este problema.
- *Fix:* usar `uv` en lugar de `pip` directo. Si rompiste pip, `python3 -m ensurepip --upgrade`.

**G-VAST-04 — CRLF line endings rompen scripts shell**
- *Síntoma:* `/bin/bash^M: bad interpreter: No such file or directory`.
- *Causa:* el script se copió desde Windows con CRLF.
- *Fix:* `dos2unix script.sh` o `git config --global core.autocrlf input`.

**G-VAST-05 — Instance no se destruye con `vastai destroy`**
- *Síntoma:* `vastai destroy <ID>` retorna 200 OK pero la instance sigue en `show instances`.
- *Causa:* delay propagación Vast.ai (~30 s).
- *Fix:* esperar 1 min. Si persiste >5 min, soporte Vast.ai.

**G-VAST-06 — `vastai search offers` devuelve lista vacía**
- *Síntoma:* búsqueda con filtros restrictivos no encuentra GPUs.
- *Causa:* RTX 4090 on-demand <0.50 USD/hora pueden no estar disponibles en horarios pico.
- *Fix:* relajar `dph_total<0.60` o probar RTX 3090 con `dph_total<0.30`.

### 8.2 Ultralytics + ONNX + TRT (G-ULTRA-*)

**G-ULTRA-01 — Ultralytics rompe con `numpy 2.0`**
- *Síntoma:* `AttributeError: module 'numpy' has no attribute 'float'`.
- *Causa:* numpy 2.0 removió aliases deprecated.
- *Fix:* `pip install "numpy<2.0"` (D2).

**G-ULTRA-02 — `onnxslim` simplificación silenciosa rompe el modelo**
- *Síntoma:* el `.onnx` se exporta sin error pero la inferencia da resultados distintos.
- *Causa:* `onnxslim<0.1.34` simplifica nodos `Slice` incorrectamente.
- *Fix:* `pip install "onnxslim>=0.1.34"` (D2).

**G-ULTRA-03 — TRT 8.2 falla compilación con `UNSUPPORTED_NODE: NonMaxSuppression`**
- *Síntoma:* `Network has dynamic or shape inputs, but no optimization profile has been defined`.
- *Causa:* `dynamic=True` o NMS opset 11+ no soportado.
- *Fix:* exportar con `dynamic=False`, `simplify=True`, `nms=False`.

**G-ULTRA-04 — Engine TRT compila pero inferencia produce NaN**
- *Síntoma:* outputs son `[nan, nan, ...]`.
- *Causa:* FP16 overflow en alguna capa (común con BatchNorm + LeakyReLU).
- *Fix:* compilar con `--strict-types` que fuerza algunas capas a FP32. En Nano: aceptar limitación y validar con FP32 si la latencia lo permite (en YOLOv8n FP32 ~50 ms, todavía da 20 FPS).

**G-ULTRA-05 — `model.export(format='onnx', opset=12)` falla con `unsupported`**
- *Síntoma:* error explícito al exportar.
- *Causa:* opset 12 no soportado por TRT 8.2.
- *Fix:* `model.export(format='onnx', opset=11, ...)`.

**G-ULTRA-06 — `requirements:` mensajes en rojo NO son errores**
- *Síntoma:* Ultralytics imprime `requirements: ... not found, attempting AutoUpdate...` en rojo durante export.
- *Causa:* convención visual de Ultralytics (rojo = atención, no error).
- *Fix:* ver log completo. Buscar `Export complete (X.Ys)` o `export success` para confirmar éxito. NO interpretar el rojo como error.

### 8.3 Roboflow (G-RF-*)

**G-RF-01 — `Dataset.location` es `null` en respuesta API REST**
- *Síntoma:* `location: null` en lugar de URL S3.
- *Causa:* bug API REST Roboflow #473.
- *Fix:* usar SDK Python `roboflow>=1.1.27` (D26).

**G-RF-02 — `download()` no descarga imágenes, solo labels**
- *Síntoma:* dataset tiene `train/labels/*.txt` pero no `train/images/*.jpg`.
- *Causa:* dataset Roboflow tiene imágenes por URL, no por upload.
- *Fix:* re-subir imágenes al workspace vía UI/SDK. Verificar flag "images included".

**G-RF-03 — Filter Null elimina demasiadas imágenes**
- *Síntoma:* esperado ~10 439 imgs, real <8 000.
- *Causa:* Modify Classes + Filter Null en exceso si muchas imágenes solo tenían clases-delete.
- *Fix:* revisar matriz de co-ocurrencia con jupyter notebook + pandas. Si crítico, considerar dataset suplementario.

**G-RF-04 — Clases "fantasma" tras export yolov8**
- *Síntoma:* `data.yaml` reporta 7 clases (cardboard, miscellaneous, etc.) pero deberían ser 3.
- *Causa:* bug histórico del Roboflow Python SDK con datasets filtrados por Modify Classes.
- *Fix:* post-download, validar manualmente que `train/labels/*.txt` solo contienen IDs 0, 1, 2. Si aparecen otros IDs, re-generar el dataset version desde la UI.

### 8.4 Nano runtime (G-NANO-*)

**G-NANO-01 — TRT engine compilado en x86 NO carga en Nano**
- *Síntoma:* `Cuda failure: the provided PTX was compiled with an unsupported toolchain`.
- *Causa:* engines TRT son específicos de versión + GPU compute capability. Engine compilado en x86 con sm_86 (RTX 30) no carga en Nano sm_53.
- *Fix:* compilar el engine EN LA NANO (D14). Usar Polygraphy en NGC TRT 21.11 solo para dry-run en x86 (Gate 4), no para producir engine final.

**G-NANO-02 — Nano queda lenta (latencia 5× peor de lo esperado)**
- *Síntoma:* inferencia que debería ser 40 FPS está en 8 FPS.
- *Causa:* thermal throttling (sin disipador activo o fan apagado).
- *Fix:* aplicar disipador con fan. Activar pwm-fan: `sudo sh -c 'echo 128 > /sys/devices/pwm-fan/target_pwm'`. Verificar `tegrastats`.

**G-NANO-03 — `apt upgrade` rompe boot en JetPack 4.6.x**
- *Síntoma:* tras `sudo apt upgrade`, Nano no arranca.
- *Causa:* `apt upgrade` actualizó `nvidia-l4t-bootloader` o `nvidia-l4t-kernel`.
- *Fix:* NO ejecutar `apt upgrade` masivo. Solo `apt update` + `apt install paquete=version` específico.

**G-NANO-04 — Swap insuficiente: TRT build mata procesos con OOM**
- *Síntoma:* `Killed` aparece random durante `trtexec --fp16`.
- *Causa:* Nano tiene 4 GB RAM + 2 GB swap default.
- *Fix:* agregar 4 GB swap: `sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile`. Agregar a `/etc/fstab` para persistencia.

**G-NANO-05 — uv `command not found` por SSH no-login**
- *Síntoma:* `ssh nano 'uv --version'` → `bash: uv: command not found`.
- *Causa:* SSH no-login no carga `~/.profile` (donde Nicolas exportó PATH).
- *Fix:* usar `ssh nano 'bash -lc "CMD"'` (D30).

### 8.5 Acceso remoto (G-REMOTE-*)

**G-REMOTE-03 — Tailscale "deadlock" de DNS en Nano ARM64**
- *Síntoma:* `tailscale up` cuelga indefinidamente.
- *Causa:* bug #14902 (DNS resolution falla en boot).
- *Fix:* `sudo tailscale up --accept-dns=false --ssh` (D27).

**G-REMOTE-04 — WireGuard kernel-module no carga en k4.9-tegra**
- *Síntoma:* `sudo modprobe wireguard` → `FATAL: Module wireguard not found`.
- *Causa:* kernel `4.9.337-tegra` NO incluye módulo WireGuard.
- *Fix:* usar `wireguard-go` userspace via Tailscale (que lo trae embebido). NO intentar compilar wireguard.ko en Nano.

### 8.6 Generales / proyecto (G-PROJ-*)

**G-PROJ-01 — `git push` falla con `Permission denied (publickey)` en Vast.ai**
- *Síntoma:* desde el container, `git push` falla.
- *Causa:* SSH key de GitHub no cargada en el container.
- *Fix:* generar nueva key en el container y agregarla a GitHub `Settings → SSH keys`. O usar HTTPS con PAT.

**G-PROJ-02 — `jupyter nbconvert --execute` falla silenciosamente al llegar a celda con error**
- *Síntoma:* el `.ipynb` no se actualiza ni levanta excepción visible.
- *Causa:* `nbconvert` por default ignora errores si flag no está.
- *Fix:* agregar `--ExecutePreprocessor.allow_errors=False` explícitamente.

**G-PROJ-03 — HF Hub upload falla con `Bad credentials`**
- *Síntoma:* `huggingface_hub.utils._errors.HfHubHTTPError: 401 Client Error`.
- *Causa:* `HF_TOKEN` no exportada o sin permiso write.
- *Fix:* `export HF_TOKEN=hf_...` antes de correr. Verificar en `huggingface.co/settings/tokens` que el token tenga "write".

**G-PROJ-04 — W&B run no aparece en dashboard tras `wandb init`**
- *Síntoma:* `wandb.init(...)` no error pero run no aparece.
- *Causa:* `WANDB_API_KEY` no configurado o modo offline.
- *Fix:* `wandb login` interactivo (una vez) o `export WANDB_API_KEY=...`.

**G-PROJ-05 — `vastai destroy` quema todo el filesystem sin warning**
- *Síntoma:* tras destroy, código local NO está en GitHub porque olvidaste push.
- *Causa:* expected behavior.
- *Fix:* `git push` ANTES de destroy. La última celda del notebook (D11) hace push + destroy en ese orden.

---

## 9. Próximos pasos sugeridos (ordenados por dependencia)

### Sprint 1 — Validación bootstrap end-to-end (1 sesión, ~3 h)

**Objetivo:** lograr que un comando único (`./bootstrap.sh`) levante el entorno Track B completo en Vast.ai.

1. **Refactorizar el `bootstrap.sh`** eliminando todo lo relacionado con Track A (TF 2.15, TFOD API, Pillow<10, protobuf==3.20, Coral wheel, gates 1-2). El script debe reducirse a ~40 líneas.
2. **Provisionar una instance Vast.ai temporal** para validar el bootstrap. Costo estimado: ~0.50 USD si todo va bien al primer intento.
3. **Validar que el venv Track B se crea** con todas las deps esperadas, registra ipykernel "Track B (YOLOv8)", y un notebook puede importar `from ultralytics import YOLO; from roboflow import Roboflow; from huggingface_hub import CommitScheduler`.
4. **Validar el cron watchdog** dejándolo correr 35 minutos con la GPU idle. Debe destruir la instance automáticamente.
5. **Destroy manual** si el cron no disparó.

### Sprint 2 — Training base + export ONNX (1 sesión, ~3 h interactiva + ~2 h cómputo)

**Objetivo:** producir el primer `best.onnx` validado y subido a HF Hub.

1. Provisionar instance Vast.ai (`bootstrap.sh` ya validado en Sprint 1).
2. Descargar dataset Roboflow 1-B con SDK.
3. **Validar dataset:** conteo imágenes train/val/test, distribución de clases en labels, 30 imágenes random visualmente OK.
4. Lanzar training Ablation 1 (100 epochs, defaults Ultralytics, `imgsz=416`, batch=16, patience=20).
5. Mientras corre, configurar CommitScheduler en background.
6. Al terminar training:
   - Evaluar sobre test split. Anotar mAP@0.5 y mAP@0.5:0.95 por clase.
   - Generar 20 sample predictions visualmente.
   - Export ONNX con flags canónicos (D2).
   - Verificar `onnx.checker.check_model()` OK.
7. Gate 3 (script Python verificando ops blacklist) directamente en el container.
8. Gate 4 (Polygraphy NGC) directamente en el container si Docker daemon está disponible. Si no, copiar el ONNX a la Win11 local y ejecutar Gate 4 con Docker Desktop.
9. Trigger HF Hub final sync. Stop scheduler. Auto-destroy.

### Sprint 3 — Compilación + smoke test en Nano (1 sesión, ~2 h)

**Objetivo:** producir el `best.engine` y validar latencia esperada en la Nano real.

1. Descargar `best.onnx` desde HF Hub a Win11.
2. SCP del ONNX a la Nano vía `nano:/home/jetson/embebidos-3/models/`.
3. SSH a Nano, sesión tmux.
4. `trtexec --onnx=best.onnx --fp16 --workspace=1024 --saveEngine=best.engine --verbose 2>&1 | tee logs/build.log`. Esperar ~8 min.
5. `trtexec --loadEngine=best.engine --iterations=100`. Verificar mean latency ~25 ms.
6. **Si la latencia es ≥30 ms con buen margen sobre 100 ms (el límite para 10 FPS):** ✅ Track B está listo en el modelo lado.
7. **Si la latencia es >100 ms:** investigar throttling térmico, usage de GPU otra cosa, o regresión en YOLOv8n fine-tuned vs baseline COCO.

### Sprint 4 — Pipeline inferencia con cámara (1 sesión, ~3 h)

**Objetivo:** loop completo cámara → engine → NMS CPU → render funcionando en la Nano.

1. **Conectar físicamente la cámara C920 a la Nano.**
2. Verificar `lsusb` reconoce `046d:082d Logitech Webcam C920`.
3. Verificar `v4l2-ctl --device=/dev/video0 --list-formats-ext` soporta MJPG 1280×720 @ 30 FPS.
4. Test mínimo OpenCV: `cv2.VideoCapture(<pipeline GStreamer>, cv2.CAP_GSTREAMER)` lee frames consistentemente, mostrar en una ventana X (con dummy HDMI plug o vía NoMachine si está configurado).
5. Escribir un script Python `inference_trackb.py` que:
   - Carga el engine con TRT Python bindings.
   - Allocate buffers GPU + host.
   - Abre cámara con pipeline GStreamer.
   - Loop con preprocess + inferencia + decode + NMS CPU + render.
6. Medir FPS end-to-end (no solo inferencia). Target: ≥15 FPS.

### Sprint 5 — Integración servos PCA9685 (1 sesión, ~3 h, depende de Sprint 4)

**Objetivo:** acción servo basada en clase detectada.

1. **Conectar físicamente PCA9685 al bus I²C de la Nano.**
2. `sudo i2cdetect -y 1` (probable bus 1) para confirmar dirección (default 0x40).
3. Instalar `Adafruit-PCA9685` o `adafruit-circuitpython-pca9685` en el venv del binario de inferencia.
4. Test mínimo: mover un servo de 0° a 90° con un script simple.
5. **Integrar con el loop de inferencia** del Sprint 4: si la clase detectada corresponde al bin de plástico (configurable), enviar pulso al servo correspondiente.
6. Validar que el loop no se bloquea durante la activación del servo. Si bloquea, mover el control de servo a un thread separado con queue.

### Sprint 6 — Stress test + demo dry-run (1 sesión, ~2 h)

**Objetivo:** validar que el sistema soporta una sesión de 30 minutos continuos sin throttling ni crashes.

1. Lanzar `inference_trackb.py` con la cámara apuntando a la banda real (o un proxy aceptable).
2. Monitor en tmux:
   - `tegrastats` continuo: ver RAM, GPU util, CPU util, temperaturas.
   - `htop` en otra ventana: confirmar no hay procesos zombie.
3. Mantener corriendo 30 minutos. Verificar:
   - FPS estable (no degrada con el tiempo).
   - Temperaturas <60°C (margen al throttling threshold 80°C).
   - RAM libre >1 GB en todo momento.
4. Si hay throttling antes de 30 min, activar `pwm-fan` a 50% (`target_pwm=128`) y repetir.

### Sprint 7 — Demo final (2026-05-26)

**Objetivo:** la demo en sí.

Preparación día anterior:
- Validar SSH `nano` desde Win11 funciona.
- Validar Tailscale auth no caducó.
- Cargar batería o tener cable de poder.
- Tener la cámara C920 apuntando al setup correcto.
- Pre-arrancar `inference_trackb.py` en sesión tmux.

Durante la demo:
- Si tiene WiFi UAO disponible → SSH directo.
- Si no → hotspot del celular del operador + Tailscale.
- Backup último recurso: cable HDMI + teclado para acceso físico.

### Estimaciones agregadas

| Sprint | Tiempo operador | Costo cloud | Hardware requerido |
|---|---|---|---|
| 1 — Bootstrap | 3 h | ~0.50 USD | — |
| 2 — Training + export | 3 h interactivo + 2 h cómputo | ~1 USD | — |
| 3 — Engine + smoke test Nano | 2 h | 0 USD | — |
| 4 — Pipeline cámara | 3 h | 0 USD | **C920 conectada** |
| 5 — Servos PCA9685 | 3 h | 0 USD | **PCA9685 conectado** |
| 6 — Stress test | 2 h | 0 USD | mismo de Sprint 5 |
| 7 — Demo | n/a | 0 USD | mismo + setup banda |
| **Total** | **~18 h operador** | **~2 USD** | |

Margen Ablations 2-5: agregar ~3-5 h operador + ~3 USD cloud.

---

## 10. Glosario

- **Backend (en contexto de ML deploy):** infraestructura de cómputo que ejecuta la inferencia. Para Track B = "GPU Maxwell vía TensorRT". Para Track A (descartado) hubiera sido "CPU + XNNPACK + NEON SIMD vía TFLite".
- **`dp4a`:** instrucción CUDA (introducida en sm_61 Pascal 2016) que realiza dot product de 4 INT8 acumulando a INT32 en un único ciclo. Es la base del speedup INT8 en GPUs NVIDIA. Ausente en Maxwell sm_53 (Tegra X1 / Jetson Nano).
- **Engine TensorRT:** archivo binario `.engine` que contiene un grafo de inferencia compilado y optimizado para una GPU específica (compute capability) y una versión específica de TRT. NO portable entre GPUs ni versiones.
- **EfficientNMS_TRT:** plugin TensorRT que aplica NMS dentro del grafo. Originalmente roto en Maxwell con TRT 8.0.x (issue NVIDIA/TensorRT#1538). **Fix incluido en TRT 8.2.1.8** que trae el JP 4.6.1 (commit `3235cc2`, julio 2021) — verificado leyendo el binary vía SSH 2026-05-14. Queda como path V1 (smoke test pendiente); V0 default sigue siendo NMS CPU NumPy por compatibilidad.
- **`fixed_shape_resizer` / `keep_aspect_ratio_resizer`:** modos de resize del TF Object Detection API. Track A los hubiera usado; Track B no aplica (Ultralytics usa LetterBox interno).
- **Fit-black (Roboflow):** modo de resize que preserva aspect ratio + agrega padding negro (valor 0) para hacer la imagen cuadrada. Equivalente conceptual al LetterBox de Ultralytics pero con padding 0 en lugar de 114.
- **Gate (en contexto de validación pre-deploy):** chequeo binario `pass`/`fail` que un artefacto (ONNX, TFLite) debe pasar antes de copiarse a la Nano. Track B tiene Gate 3 (ops blacklist) y Gate 4 (Polygraphy NGC). Gates 1-2 (TFLite) son Track A, fuera de alcance.
- **JetPack:** stack de software de NVIDIA para Jetson, incluye L4T (Linux for Tegra) + CUDA + cuDNN + TensorRT + librerías multimedia. Nuestra Nano corre JetPack 4.6.1 (L4T R32.7.1, verificado vía SSH 2026-05-14).
- **LetterBox (Ultralytics):** transformación que resize preservando aspect ratio y agrega padding (default valor 114 = ImageNet mean RGB averaged) para hacer la imagen cuadrada. Default `auto=False` en `val` y `predict`, `True` (=Stretch) en `train`.
- **`looker-remote-desktop`:** herramienta hecha por Nicolas Cuaran (`github.com/NicolasCuaran/looker-remote-desktop`) que combina x11vnc + Tailscale `--ssh` + GDM autologin para acceso remoto headless a la Nano. Ya operativa.
- **Maxwell sm_53:** arquitectura de la GPU del Tegra X1 (Jetson Nano). 128 CUDA cores, sin Tensor Cores, sin `dp4a`. SOLO acelera FP32 y FP16 en hardware; INT8 cae a software (más lento).
- **MJPG:** formato de codec USB de la C920 que es OBLIGATORIO para alcanzar 30 FPS. Sin MJPG, USB 2.0 satura ancho de banda con frames YUV crudos.
- **Modify Classes + Filter Null (Roboflow):** combinación que elimina anotaciones de clases no-deseadas (cardboard, metal, etc.) + elimina las imágenes que quedan sin ninguna anotación. Usado para reducir el dataset original de 7 clases a 3.
- **NMS (Non-Maximum Suppression):** algoritmo post-detection que elimina bboxes redundantes. En Track B se aplica en CPU NumPy con `cv2.dnn.NMSBoxes` porque el plugin TRT está roto en Maxwell.
- **NVFBC (NVIDIA Frame Buffer Capture):** API NVIDIA para captura de framebuffer GPU-side. Existe solo en GPUs discretas Quadro/GeForce; **no existe en Tegra X1**. Por eso Sunshine no funciona como servidor host en la Nano.
- **ONNX opset:** versión del conjunto de operadores ONNX. TRT 8.2 soporta hasta opset 13 nominalmente, pero en la práctica opset 11 es el más seguro.
- **Polygraphy:** herramienta NVIDIA (parte del repo TensorRT) para validar y comparar modelos ONNX vs engines TRT vs ONNX Runtime. No funciona en JetPack 4.6.1 (Py 3.6.9 incompat); SÍ funciona en Docker NGC `tensorrt:21.11-py3` (Ubuntu 20.04 + Py 3.8 + TRT 8.2.1).
- **QAT (Quantization Aware Training):** método de entrenamiento que simula los efectos de la cuantización INT8 durante el forward pass, lo que permite recuperar accuracy. Track A lo hubiera usado obligatoriamente; Track B no aplica (no usamos INT8).
- **Representative dataset:** muestras del val split usadas durante la calibración INT8 PTQ para estimar las distribuciones de activaciones. Track A: 300-500 muestras distribuidas proporcionalmente. Track B INT8 experimental: idem.
- **`tegrastats`:** comando Linux específico de Tegra que reporta usage RAM/CPU/GPU/EMC/PMIC + temperaturas + frequencies. Reemplazo de `nvidia-smi` (que NO existe en Tegra).
- **Tegra X1:** SoC NVIDIA que combina ARM Cortex-A57 quad-core + GPU Maxwell 128 cores + 4 GB LPDDR4 unificada. Usado en Jetson Nano (B01) y Nintendo Switch.
- **TensorRT (TRT):** runtime y librería NVIDIA para inferencia ML acelerada en GPU. Compila el grafo ONNX a un engine binario optimizado para la GPU + versión TRT específicas.
- **TFLite_Detection_PostProcess:** custom op de TensorFlow Lite que embebe NMS + decode dentro del grafo. Track A lo hubiera usado; Track B no aplica.
- **Ultralytics:** empresa y framework Python que mantiene YOLOv5, YOLOv8, YOLO11. API consistente para training, evaluation, export.
- **uv:** gestor de paquetes y entornos virtuales Python escrito en Rust, 10-100× más rápido que pip. Default del usuario per regla global.
- **uvx:** alias de `uv tool run`. Ejecuta una herramienta Python en un venv efímero sin instalación global.
- **Vast.ai:** marketplace de GPU on-demand. Pricing variable, RTX 4090 ~0.40 USD/hora. Container base oficial con CUDA preinstalado.
- **WireGuard kernel module:** módulo del kernel Linux que implementa el protocolo WireGuard. **NO existe en kernel 4.9-tegra.** Tailscale lo reemplaza con `wireguard-go` userspace automáticamente.
- **YOLOv8n (nano):** variante más pequeña de YOLOv8, ~3.1 M parámetros, 8.7 GFLOPs. Diseñada para edge inference. Variantes superiores: s (11 M), m (25 M), l (43 M), x (68 M).

---

## 11. Referencias clave

Filtrado a Track B + acceso remoto Nano. Para el catálogo completo (~270 URLs), ver historial git de `CONSOLIDADO-embebidos-3.md` antes de su eliminación.

### 11.1 Stack training Track B

- Ultralytics docs: https://docs.ultralytics.com/
- Ultralytics ONNX integration: https://docs.ultralytics.com/integrations/onnx/
- Ultralytics TensorRT integration: https://docs.ultralytics.com/integrations/tensorrt/
- Ultralytics CLI/Python API: https://docs.ultralytics.com/usage/cli/, https://docs.ultralytics.com/usage/python/
- Ultralytics default.yaml: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml
- Ultralytics issue #14751 (workspace OOM en Nano): https://github.com/ultralytics/ultralytics/issues/14751
- Ultralytics PR #21652 (padding_value configurable): https://github.com/ultralytics/ultralytics/pull/21652
- Ultralytics PR #24028 (INT8 calibration fix imgsz no-square): https://github.com/ultralytics/ultralytics/pull/24028
- Ultralytics issue #14530 (imgsz reduce latencia Nano): https://github.com/ultralytics/ultralytics/issues/14530
- onnxslim: https://github.com/inisis/OnnxSlim

### 11.2 Validación ONNX + TensorRT

- ONNX operators reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md
- TRT operators matrix: https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html
- TRT support matrix Maxwell: https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html
- onnx-tensorrt 8.2-GA operators.md: https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md
- Polygraphy CHANGELOG: https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/CHANGELOG.md
- Polygraphy tool: https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy
- NGC TensorRT container catalog: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
- Issue NVIDIA/TensorRT#1538 (EfficientNMS_TRT roto Maxwell): https://github.com/NVIDIA/TensorRT/issues/1538
- Issue NVIDIA/TensorRT#3762 (`--int8 means Enable int8 precision`): https://github.com/NVIDIA/TensorRT/issues/3762
- Foro NVIDIA #349598 (Polygraphy falla JP 4.6.1): https://forums.developer.nvidia.com/t/how-to-generate-and-verify-an-int8-calibration-cache-cache-for-trtexec-on-on-jetson-nano-tensorrt-8-2-1-8-polygraphy-failing-on-device/349598

### 11.3 Benchmarks Jetson Nano

- Nature Sci Reports 2024 (Tabla 4 YOLOv8n imgsz vs FPS Nano): https://www.nature.com/articles/s41598-024-74798-3
- Qengineering YoloV8-TensorRT-Jetson_Nano: https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano
- Qengineering YoloV5-ncnn-Jetson-Nano (tabla benchmark 17 modelos): https://github.com/Qengineering/YoloV5-ncnn-Jetson-Nano
- Špeh Medium 2023 (YOLOv7-tiny FPS por imgsz): https://medium.com/@jurespeh/yolov7-with-tensorrt-on-jetson-nano-with-python-script-example-63099fa7c8a5
- imnuman/jetson-object-detection (YOLOv8n@640 28 FPS): https://github.com/imnuman/jetson-object-detection
- jkjung-avt/tensorrt_demos: https://github.com/jkjung-avt/tensorrt_demos
- dusty-nv/jetson-inference: https://github.com/dusty-nv/jetson-inference
- Foro NVIDIA "Object detection on Nano with yolov8": https://forums.developer.nvidia.com/t/object-detection-on-nano-with-yolov8-model/275370

### 11.4 Vast.ai + cloud training

- Vast.ai landing: https://vast.ai/
- Vast.ai CLI docs: https://vast.ai/docs/cli/commands
- Vast.ai base image registry: https://hub.docker.com/r/vastai/base-image/tags
- uv docs: https://docs.astral.sh/uv/
- HF Hub CommitScheduler: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/utilities#huggingface_hub.CommitScheduler
- HF Hub upload guide: https://huggingface.co/docs/huggingface_hub/main/en/guides/upload
- nbconvert: https://nbconvert.readthedocs.io/en/latest/usage.html
- tmux man: https://man.openbsd.org/tmux.1

### 11.5 Roboflow

- Roboflow docs root: https://docs.roboflow.com/
- Roboflow preprocessing: https://docs.roboflow.com/datasets/dataset-versions/image-preprocessing
- Roboflow augmentation: https://docs.roboflow.com/datasets/dataset-versions/image-augmentation
- Roboflow Python SDK: https://github.com/roboflow/roboflow-python
- Roboflow bug #473 (location null): https://github.com/roboflow/roboflow-python/issues/473
- discuss.roboflow.com #6892 (best resize method YOLOv8): https://discuss.roboflow.com/t/selecting-the-best-image-resizing-method-in-roboflow-for-training-a-yolov8/6892
- YOLOv5 issue #7454 (letterbox doble -4-5 pp): https://github.com/ultralytics/yolov5/issues/7454
- Ultralytics issue #7053 (mixed aspect ratio + Roboflow): https://github.com/ultralytics/ultralytics/issues/7053

### 11.6 Papers académicos clave

- Crasto 2024 (class imbalance object detection): https://arxiv.org/abs/2403.07113
- Yun & Wong CVPR 2021 (MobileNet QUINT8 catastrófico): https://arxiv.org/abs/2104.11849
- Jacob et al. CVPR 2018 (QAT integer-arithmetic): https://arxiv.org/abs/1712.05877
- Krishnamoorthi 2018 (quantizing CNNs whitepaper): https://arxiv.org/abs/1806.08342
- Karimov et al. 2025 (quantization robustness input degradations): https://arxiv.org/abs/2508.19600
- Alqahtani et al. 2024 (edge benchmarking ICSOC): https://arxiv.org/abs/2409.16808
- Swaminathan et al. 2024 (Jetson Nano benchmarks): https://arxiv.org/abs/2406.17749
- Chiam et al. 2025 (Energy Optimized YOLO): Semantic Scholar
- Boddu & Mukherjee 2025 (YOLOv4-Tiny RPi 5 deployment): https://arxiv.org/abs/2506.09300
- Bochkovskiy et al. 2020 (YOLOv4 + mosaic +2.3 pp): https://arxiv.org/abs/2004.10934
- Zhong et al. AAAI 2020 (Random Erasing +1.4 pp Fast-RCNN): https://arxiv.org/abs/1708.04896
- Bashkirova et al. CVPR 2022 (ZeroWaste dataset): https://arxiv.org/abs/2106.15279
- Li & Grammenos UCL 2022 (Smart Recycling Bin Jetson Nano): https://arxiv.org/pdf/2210.00448

### 11.7 Acceso remoto Nano

- OpenSSH: https://www.openssh.com/
- Tailscale: https://www.tailscale.com/
- Tailscale issue #14902 (DNS bug ARM64): https://github.com/tailscale/tailscale/issues/14902
- Tailscale SSH KB: https://tailscale.com/kb/1193/tailscale-ssh
- NicolasCuaran/looker-remote-desktop: https://github.com/NicolasCuaran/looker-remote-desktop
- NoMachine ARM64 download: https://www.nomachine.com/download/download&id=115
- NoMachine + Xfce4 + Jetson Nano video: https://youtu.be/vBMHS6FXBM4
- Foro NVIDIA "WireGuard kernel module Jetson Nano" (#184764): https://forums.developer.nvidia.com/t/wireguard-kernel-module-jetson-nano/184764

### 11.8 Hardware + L4T + JetPack 4.6.1

- Jetson Nano DevKit producto: https://developer.nvidia.com/embedded/jetson-nano-developer-kit
- JetPack landing: https://developer.nvidia.com/embedded/jetpack
- JetPack 4.6.1: https://developer.nvidia.com/jetpack-sdk-461
- Linux for Tegra R32.7.6: https://developer.nvidia.com/embedded/linux-tegra-r3276
- eLinux Jetson Nano wiki: https://elinux.org/Jetson_Nano
- JetsonHacks blog: https://www.jetsonhacks.com/

### 11.9 Cámara C920 + GStreamer

- Logitech C920 producto: https://www.logitech.com/en-us/products/webcams/c920-pro-hd-webcam.html
- GStreamer v4l2src docs: https://gstreamer.freedesktop.org/documentation/video4linux2/v4l2src.html
- OpenCV VideoCapture GStreamer: https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html

### 11.10 Roboflow dataset original

- Workspace: https://app.roboflow.com/embebidos3/waste-3class-lwld8

---

## 12. Tabla de cambios respecto al CONSOLIDADO original

| Aspecto | Original CONSOLIDADO | Este HANDOFF | Razón |
|---|---|---|---|
| Tracks cubiertos | A + B | **Solo B** | Decisión usuario 2026-05-13 §1.3 |
| Decisiones activas | D1-D28 | D2, D3, D5, D6, D8, D9, D11, D13, D14, D17, D18, D26, D27 + D29, D30 (nuevas) | Filtro Track B + 2 nuevas de sesión 2026-05-13 |
| Sección AP-mode hostapd-rtw | §38-§39 | **Eliminada** | Usuario descartó AP mode |
| Sección driver TP-Link | §35-§37 | **Eliminada** | Usuario descartó (D20-D25 archivadas) |
| Sección bastion Contabo autossh:443 | §30-§31 | **Eliminada** | Descartado por simplicidad (D28 archivada) |
| Sección Headscale fallback | §29 | **Eliminada** | Tailscale `--ssh` suficiente (D27 vigente) |
| Sección NoMachine + Xfce4 | §28 | Reducida a glosario | Looker-remote-desktop ya operativo, no necesita re-instalación |
| Runbooks ejecutables | A-E completos con comandos | **Solo §5 conceptual** | Por instrucción usuario: "registro de decisiones más que un documento que contenga comandos a ejecutar" |
| Gotchas | ~60 | ~30 filtrados Track B | Filtro por aplicabilidad |
| Fuentes consultadas | ~270 URLs | ~100 filtradas | Solo las relevantes a Track B |
| Líneas totales | 5467 | ~1900 (target <2000) | Restricción usuario |
| Aprendizajes sesión 2026-05-13 | No existían | §6 (10 lecciones) | Nueva contribución |
| Sprints sugeridos | No existían | §9 (7 sprints) | Nueva contribución |

---

## 13. Memory de mnemon — IDs relevantes para esta línea de trabajo

Cuando una sesión futura haga `mnemon recall "track b" --limit 5`, debería traer los siguientes IDs (verificar vigencia con `mnemon read <ID>`):

- `97dc5e7c-c2e0-40f4-8313-8ddcea0366c2` — negative constraint: SSH no-login + uv PATH issue (D30 racional).
- `9a29b10d-89a7-4bd7-9453-d28487d6ce0f` — última versión confirmada del CONSOLIDADO original (será eliminada cuando se elimine el archivo).
- `207d435d-...` — primera versión del CONSOLIDADO (histórica).

Otros memories relevantes para Track B (a buscar):
- Decisión Track B 416×416 + Ultralytics ≥8.4.46.
- Auto-destroy Vast.ai pattern.
- HF Hub CommitScheduler + repo privado.
- Bug TP-Link RTL8188EUS (NO RE-INVESTIGAR — usuario descartó).
- Tailscale `--accept-dns=false --ssh`.

Para sesiones futuras: al iniciar, ejecutar `mnemon recall "embebidos 3 track b" --limit 10` y consolidar el resultado con este HANDOFF.

---

## 14. Cierre

Este documento condensa **toda la investigación previa relevante a Track B** + **lo aprendido empíricamente en la sesión interactiva 2026-05-13** en un único registro de decisiones. Sustituye sin pérdida de contenido los 6 archivos previos listados en §0.

La siguiente sesión de trabajo debería:
1. Comenzar por §1 (contexto) → §3 (estado actual) → §9 (próximos pasos).
2. Recuperar el estado mental con §6 (aprendizajes 2026-05-13).w
3. Tomar el primer sprint sugerido (§9 Sprint 1: bootstrap Vast.ai end-to-end).

Para cualquier decisión que se cuestione, consultar §2 (ledger). Para cualquier problema durante implementación, consultar §8 (gotchas). Para entender el por qué de algo no presente aquí, ver historial git de los archivos eliminados.

**FIN DEL HANDOFF.**
