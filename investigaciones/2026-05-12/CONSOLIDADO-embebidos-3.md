# CONSOLIDADO embebidos-3 — fusión exhaustiva (sustituye los 6 docs originales)

**Proyecto:** `embebidos-3` — clasificador de residuos (`glass`, `paper`, `plastic`) en Jetson Nano B01.
**Materia:** IA en Embebidos (Prof. Juan Camilo Giraldo, UAO Cali, semestre 7).
**Entrega:** 2026-05-26.
**Fecha de consolidación:** 2026-05-12 (post-Ronda 8 + cierre empírico in-situ).
**Mantenedor:** Martín García (`mitgar14`).

> **Este documento es la fusión exhaustiva de los 6 archivos hermanos** (`decisiones-D1-D15-ledger.md`, `compatibilidad-stack-cloud-jetson.md`, `infraestructura-training-vastai-uv-hf.md`, `validacion-artefactos-pre-deploy.md`, `dataset-roboflow-yolov8.md`, `acceso-remoto-wifi-jetson-nano.md`). Preserva TODOS los snippets de código, comparativas, razonamiento de decisiones descartadas, URLs y gotchas. Reorganizado **por dominio temático**, no por documento fuente. Los originales son redundantes y pueden eliminarse tras validar este consolidado.

**Estado:** 28 decisiones vinculantes (D1-D28) + 6 obsoletas/refinadas (D4 → D9; D7 → D11; D16, D17, D18, D19 refinadas). 8 rondas de `/investiga` cerradas (R1-R8 + R7-bis in-situ).

**Repos relacionados:**
- Código del proyecto: `github.com/mitgar14/embebidos-3` (privado)
- Modelos: `huggingface.co/mitgar14/embebidos-3-models` (privado)

---

## Índice

### Parte I — Resumen y contexto
1. [Resumen ejecutivo](#1-resumen-ejecutivo)
2. [Hardware y contexto inmutable del Nano](#2-hardware-y-contexto-inmutable-del-nano)

### Parte II — Stack de training (Vast.ai)
3. [Por qué Vast.ai (Colab y Kaggle descartados)](#3-por-que-vastai)
4. [Container y GPU on-demand (D1, D2)](#4-container-y-gpu-on-demand)
5. [Track A — TF 2.15 + TFOD API + SSD MV2 + PTQ INT8](#5-track-a--tf-215--tfod-api)
6. [Track B — YOLOv8n + Ultralytics + ONNX](#6-track-b--yolov8n--ultralytics)
7. [Dataset Roboflow `embebidos3/waste-3class-lwld8`](#7-dataset-roboflow)
8. [Dual venv uv (D10)](#8-dual-venv-uv)
9. [Notebook persistente — tmux + nbconvert (D9)](#9-notebook-persistente)
10. [HF Hub persistence (D3, D5, D6)](#10-hf-hub-persistence)
11. [Auto-destroy — cron watchdog + última celda (D11)](#11-auto-destroy)
12. [Bootstrap.sh completo](#12-bootstrapsh-completo)
13. [Comandos `vastai create instance`](#13-comandos-vastai-create-instance)

### Parte III — Validación pre-deploy
14. [Mecánica del versioning TFLite](#14-mecanica-del-versioning-tflite)
15. [Gate 1 — TFLite `op_version` (D12)](#15-gate-1--tflite-op_version)
16. [Gate 2 — TFLite carga test (D12)](#16-gate-2--tflite-carga-test)
17. [D15 — Plan B Coral wheel CP36 aarch64](#17-d15--plan-b-coral-wheel)
18. [Workaround `flatbuffer_utils.py`](#18-workaround-flatbuffer_utils)
19. [Gate 3 — ONNX ops blacklist TRT 8.2 (D13)](#19-gate-3--onnx-ops-blacklist)
20. [Gate 4 — Polygraphy en Docker NGC (D13)](#20-gate-4--polygraphy-docker-ngc)
21. [INT8 Maxwell `sm_53` — cierre del gap (D14)](#21-int8-maxwell-cierre-del-gap)
22. [Pipeline `validate_artifacts.py`](#22-pipeline-validate_artifacts)

### Parte IV — Stack de deploy en el Nano
23. [Engine TRT compilado en el Nano (D8)](#23-engine-trt-compilado-en-nano)
24. [Pipeline de inferencia Track A + Track B](#24-pipeline-de-inferencia)

### Parte V — Acceso remoto y red
25. [Topología y escenarios A/B/C](#25-topologia-y-escenarios)
26. [SSH OpenSSH Windows → Nano](#26-ssh-openssh-windows-nano)
27. [Tailscale con workaround DNS (D27)](#27-tailscale-con-workaround-dns)
28. [NoMachine + Xfce4 + dummy HDMI (D26)](#28-nomachine--xfce4--dummy-hdmi)
29. [Sunshine + Moonlight — descartados](#29-sunshine--moonlight-descartados)
30. [Contabo VPC bastion (D20)](#30-contabo-vpc-bastion)
31. [autossh reverse tunnel a Contabo:443 (D28)](#31-autossh-reverse-tunnel)
32. [WireGuard kernel-module BROKEN + userspace](#32-wireguard-broken-y-userspace)
33. [Comparativa overlay networks + benchmarks ARM](#33-comparativa-overlay-networks)
34. [Headscale self-hosted (migration path)](#34-headscale-self-hosted)

### Parte VI — WiFi en el Nano
35. [Hardware dual Wi-Fi (Intel M.2 + TP-Link USB)](#35-hardware-dual-wi-fi)
36. [Driver TP-Link TL-WN722N v4 — `lwfinger/rtl8188eu` (D21, D23)](#36-driver-tp-link)
37. [Por qué `aircrack-ng/rtl8188eus` NO compila en k4.9-tegra](#37-por-que-aircrack-ng-no-compila)
38. [AP mode hostapd `driver=rtl871xdrv` (D25)](#38-ap-mode-hostapd)
39. [NetworkManager hotspot alternativa](#39-networkmanager-hotspot)

### Parte VII — Decisiones, runbooks, gotchas y referencias
40. [Ledger consolidado D1-D28 (vinculantes + obsoletas/refinadas)](#40-ledger-consolidado-d1-d28)
41. [Runbooks ejecutables (5 runbooks)](#41-runbooks-ejecutables)
42. [Gotchas catalogados (60+)](#42-gotchas-catalogados)
43. [Fuentes consultadas (~270 acumuladas)](#43-fuentes-consultadas)
44. [Historial de rondas R4-R8](#44-historial-de-rondas)
45. [Gaps de evidencia residuales (G1-G12)](#45-gaps-residuales)
46. [Cómo usar este consolidado](#46-como-usar-este-consolidado)

---

## 1. Resumen ejecutivo

**Objetivo:** desplegar un clasificador de residuos en Jetson Nano B01 que detecta 3 clases (`glass`, `paper`, `plastic`) en tiempo real (>10 FPS) con cámara USB Logitech C920 OG, accesible remotamente desde Windows 11 vía SSH + escritorio.

### 1.1 Stack consolidado

| Capa | Decisión | Razón principal |
|---|---|---|
| **Training cloud** | Vast.ai container `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` + RTX 4090 on-demand (D1, D2) | Forward-compat para producir `.tflite` y `.onnx` ejecutables en JP 4.6.5 runtime; un solo container para dos tracks |
| **Track A (TF)** | TF 2.15 + TFOD API SHA `9cafa3d150` + SSD MobileNetV2 plain 320×320 + PTQ INT8 | Stack maduro con `TFLite_Detection_PostProcess` embebido; CPU TFLite + XNNPACK ~14-18 FPS |
| **Track B (PyTorch)** | YOLOv8n 416×416 + Ultralytics 8.4.46 + ONNX opset 11 + TRT FP16 engine (D14) | GPU Maxwell sin `dp4a` → FP16-only; engine compilado en Nano (D8) |
| **Dataset** | Roboflow `embebidos3/waste-3class-lwld8` v1-B + workaround bug `location` | 3 clases visualmente distintas curadas; bug SDK aún sin fix a 2026-05 |
| **Dual venv** | `uv venv /opt/venv/tracka` + `/opt/venv/trackb`, ipykernels distintos (D10) | TF 2.15 protobuf 3.20.3 + torch 2.1+cu121 numpy<2.0 son irreconciliables |
| **Notebook persistente** | `.ipynb` + `nbconvert --execute --inplace` dentro de `tmux` (D9) | Único patrón que sobrevive cierre de pestaña con outputs preservados |
| **Persistencia** | HF Hub `mitgar14/embebidos-3-models` privado + `CommitScheduler(every=5)` + `upload_folder` + W&B Track B (D3, D5, D6) | Tolerancia a fallos de red, sin pérdida >5min de progreso |
| **Auto-destroy** | Cron watchdog (1 min) + última celda → `vastai destroy instance` (D11) | Vast.ai NO tiene `--auto-stop` ni idle shutdown |
| **Validación pre-deploy** | 4 gates: TFLite op_version + carga, ONNX ops blacklist, Polygraphy Docker NGC `21.11-py3` (D12, D13) | Evitar ciclo "bajar al Nano → falla → re-train 2h" |
| **Acceso remoto** | SSH OpenSSH + NoMachine + Xfce4 + dummy HDMI + Tailscale + autossh→Contabo:443 (D17→D27, D18→D26, D20→D28) | Stack dual primario+backup; Sunshine descartado por NVFBC ausente en Tegra |
| **WiFi en Nano** | Chip Intel M.2 interno (STA, driver `iwlwifi`) + TP-Link TL-WN722N v4 (USB, RTL8188EUS, driver `lwfinger/rtl8188eu`) (D21, D22, D24) | Dual: Intel para cliente Wi-Fi; TP-Link para AP mode hostapd battery-powered |
| **VPN/overlay** | Tailscale `--accept-dns=false --ssh` (workaround bug #14902) + Headscale en Contabo migration path (D27) | WireGuard kernel-module BROKEN en 4.9-tegra; Tailscale usa `wireguard-go` userspace |
| **Reverse tunnel** | autossh systemd → Contabo VPS:443 (D28) | VPS ya pagado; puerto 443 indistinguible HTTPS = bypass DPI universitario |

### 1.2 Constraints críticas (orden de importancia)

1. **Runtime JP 4.6.5 inmutable**: Python 3.6.9, TFLite 2.5, TRT 8.2.1.8, CUDA 10.2, Maxwell `sm_53` sin `dp4a`.
2. **WireGuard kernel-module BROKEN en 4.9-tegra** ([foro #184764](https://forums.developer.nvidia.com/t/kernel-error-when-using-wireguard/184764)) → Tailscale usa `wireguard-go` userspace automáticamente.
3. **`aircrack-ng/rtl8188eus` NO compila en kernel 4.9-tegra** (API mismatch `NL80211_TIMEOUT_UNSPECIFIED`, kernel ≥4.14) → usar `lwfinger/rtl8188eu`.
4. **Sunshine + Moonlight NO funcionan en Tegra X1** (NVFBC ausente, no .deb arm64 Bionic) → NoMachine es única opción.
5. **TFLite forward-compat NO garantizada**: `op_version` puede exceder runtime 2.5 max → Gate 1 obligatorio.
6. **TRT engine GPU-arch-specific**: compilar siempre en Nano, nunca transferir (D8).
7. **TF OD API requiere TF ≤ 2.15** (TF 2.16+ rompe `tf.estimator`) y TF 2.15 NO tiene wheel cp312 → Python 3.10 (no 3.12 de Colab).

### 1.3 Roadmap de implementación (14 días)

| Fase | Días | Tareas | Tiempo estimado |
|---|---|---|---|
| **Fase 1** | 1-3 | Bootstrap Vast.ai + Track A + Track B + HF Hub persistence | 6-10 h cloud + 2 h dev local |
| **Fase 2** | 4-7 | Validación pre-deploy (4 gates) + transferir artefactos al Nano | 2-3 h |
| **Fase 3** | 8-12 | Pipeline inferencia en Nano + acceso remoto stack + dummy HDMI plug | 8-12 h |
| **Fase 4** | 13-14 | Demo prep + hostapd setup para battery-powered | 4-6 h |

### 1.4 Decisiones obsoletas / refinadas (resumen)

| Original | Estado | Reemplaza por | Razón |
|---|---|---|---|
| **D4** (jupytext `--to py:percent`) | ❌ Obsoleta | **D9** (nbconvert + tmux) | Usuario pidió mantener `.ipynb` interactivo con tolerancia a cierre de pestaña |
| **D7** (`trap EXIT` en `run.sh`) | ❌ Obsoleta | **D11** (cron watchdog + última celda) | `trap EXIT` requiere bash longevo; kernel detached lo elude |
| **D16** (driver aircrack-ng/rtl8188eus) | 🔄 Refinada | **D21** (driver lwfinger/rtl8188eu) | aircrack-ng NO compila en k4.9 (API mismatch verificado in-situ) |
| **D17** (Tailscale standard) | 🔄 Refinada | **D27** (Tailscale `--accept-dns=false`) | Bug ARM64 #14902 sin fix oficial |
| **D18** (NoMachine genérico) | 🔄 Refinada | **D26** (NoMachine + Xfce4 + dummy HDMI físico) | Receta validada en JP 4.6.5 |
| **D19** (escenarios demo) | 🔄 Refinada | **D22** (dual Wi-Fi) | Descubrimiento chip Intel M.2 interno cambia estrategia |
| **D20** (Contabo bastion básico) | 🔄 Refinada | **D28** (autossh:443) | Puerto 443 indistinguible de HTTPS = bypass DPI |

Las 28 decisiones vigentes finales: **D1, D2, D3, D5, D6, D8, D9, D10, D11, D12, D13, D14, D15, D21, D22, D23, D24, D25, D26, D27, D28** (21 vigentes + 7 obsoletas/superseded).

---

## 2. Hardware y contexto inmutable del Nano

### 2.1 Specs hardware del Jetson Nano B01

| Componente | Especificación |
|------------|----------------|
| SoC | NVIDIA Tegra X1 (TX1) |
| CPU | ARM Cortex-A57 quad-core @ 1,43 GHz (aarch64) |
| GPU | NVIDIA Maxwell, **128 CUDA cores**, `sm_53` |
| RAM | 4 GB LPDDR4 64-bit @ 1600 MHz, **compartida CPU/GPU** |
| Tensor Cores | **❌ Ninguno** (Maxwell predates Volta 2017) |
| Instrucción `dp4a` (INT8 SIMD) | **❌ NO disponible** (introducida en Pascal `sm_61` en 2016) |
| Storage | microSD (recomendado U3/V30, 64 GB+) |
| Power modes | 5 W (`MAXN-5W`) y 10 W (`MAXN`) |
| Cámara | USB UVC v4l2 — **Logitech C920 OG Rev 1** (PID `046d:082d`) |
| Wi-Fi internal | **Intel M.2 Key E** (MAC OUI `3c:64:cf`, driver `iwlwifi` in-tree) — descubierto Ronda 7-bis |
| Wi-Fi USB externo | **TP-Link TL-WN722N v4** (USB ID `2357:010c`, chip RTL8188E rev D 1T1R) |

### 2.2 Software inmutable (JetPack 4.6.5 tras `apt upgrade` Ronda 7-bis)

> El usuario hizo `sudo apt upgrade` y el paquete `nvidia-l4t-kernel` se actualizó de `4.9.253-tegra-32.7.1` (JP 4.6.1) a `4.9.337-tegra-32.7.6` (JP 4.6.5). Esto no cambia el runtime de inferencia (TFLite/TRT siguen siendo los mismos), pero sí provee `nvidia-l4t-kernel-headers` con symlink válido en `/lib/modules/$(uname -r)/build` que facilita DKMS.

| Componente | Versión exacta | Notas |
|------------|----------------|-------|
| L4T | **R32.7.6** (post-upgrade; R32.7.1 inicial) | kernel Linux 4.9.337-tegra |
| OS | Ubuntu 18.04 LTS bionic | EOL upstream pero Nano sigue soportada |
| **Python** | **3.6.9** | system, no actualizable sin riesgo de romper L4T |
| **TensorFlow** | **2.5.0+nv21.8** | wheel oficial NVIDIA |
| **TFLite runtime** | **2.5** | bundled con TF; fallback wheel Coral CP36 aarch64 (D15) |
| **TensorRT** | **8.2.1.8** | bundled JetPack; APIs Python + C++ |
| **CUDA** | **10.2** | runtime + dev |
| **cuDNN** | **8.2.1** | bundled CUDA |
| **OpenCV** | **4.1.1** | sin GStreamer support compilado por default |
| **TFLite schema** | **v3** | estable desde TF 2.x (`TFLITE_SCHEMA_VERSION = 3` en `tensorflow/lite/version.h` HEAD = TF 2.21 master) |

### 2.3 Implicación crítica: ausencia de `dp4a` → INT8 sin speedup

Maxwell `sm_53` **carece de la instrucción `dp4a`** (dot product de 4×INT8 acumulando a INT32) introducida en Pascal `sm_61`. Sin `dp4a`, TensorRT en INT8 tiene 3 opciones, ninguna útil:

1. **Kernels CUDA INT8 SIMD vía `dp4a`** → no disponible.
2. **Emular INT8 vía FP16/FP32** → elimina beneficio + overhead conversiones.
3. **Mixed precision fallback** → revierte capa a FP16, grafo mixto.

**Confirmación NVIDIA** (issue [`NVIDIA/TensorRT#3762`](https://github.com/NVIDIA/TensorRT/issues/3762)): *"`--int8` means Enable int8 precision, in addition to fp32."* INT8 nunca reemplaza FP32, lo complementa; las capas no cuantizables revierten.

**Consecuencia operativa:** Track B **FP16-only por default** (D14).

### 2.4 Cámara y pipeline de captura

Hardware: **Logitech C920 OG Rev 1** (PID `046d:082d`). Validado en investigación previa (`investigaciones/2026-05-10/2026-05-10-camara-usb-jetson-nano.md`).

Pipeline GStreamer canónico ya verificado:

```bash
gst-launch-1.0 -v \
  v4l2src device=/dev/video0 \
  ! image/jpeg,width=1280,height=720,framerate=30/1 \
  ! jpegdec \
  ! videoconvert \
  ! video/x-raw,format=BGR \
  ! appsink
```

**Quirk JetPack 4.6.x:** el control v4l2 se llama `focus_auto` (legacy), **no `focus_automatic_continuous`** (memoria mnemon `355dd78a`). Comando: `v4l2-ctl -d /dev/video0 --set-ctrl=focus_auto=0`.

---

## 3. Por qué Vast.ai

### 3.1 Por qué NO Colab (decisión cerrada, informativo)

Track A se intentó en Google Colab con `condacolab` para bajar Python a 3.10. La cadena de fallos reproducibles fue:

#### 3.1.1 Fallo 1 — Colab 2026.04 default Python 3.12

| Plataforma | Release | Python | TF default | PyTorch default | CUDA host |
|------------|---------|--------|-----------|-----------------|-----------|
| Google Colab | 2026.04 | **3.12.13** | 2.19.0 | 2.10.0 | 12.x |
| Google Colab | 2026.01 | 3.12.12 | 2.19.0 (pre) | 2.9.0 | 12.x |

Fuente verbatim FAQ Colab: *"2026.04: Ubuntu 22.04.5 LTS, Python 3.12.13, numpy 2.0.2, TensorFlow 2.19.0"* — [research.google.com/colaboratory/runtime-version-faq.html](https://research.google.com/colaboratory/runtime-version-faq.html).

**Pero TF 2.13/2.14/2.15 no tienen wheels cp312** (issue [tensorflow/tensorflow#62003](https://github.com/tensorflow/tensorflow/issues/62003); verbatim del usuario: `ERROR: Could not find a version that satisfies the requirement tensorflow==2.15.0 (from versions: 2.16.0rc0, 2.16.1, ...)`).

| TF version | Wheel cp312 | Notas |
|------------|-------------|-------|
| 2.13.x | ❌ NO | Primer fallo del usuario |
| 2.15.x | ❌ NO | Issue #62003 |
| 2.16.1 | ✅ primera versión cp312 | Blog TF marzo 2024 |
| 2.16–2.20 | ✅ | Pero **TF 2.16+ rompe `tf.estimator` → OD API roto** |
| 2.21.0 (mar 2026) | ✅ | Mantiene 3.10–3.12 |

**Cuello de botella crítico:** TF OD API legacy (`research/object_detection/model_main_tf2.py`) requiere TF ≤ 2.15 (limit duro). TF 2.15 no tiene wheel cp312 → necesitamos bajar Colab a Python 3.10.

#### 3.1.2 Fallo 2 — `condacolab.check()` lanza `AssertionError`, no devuelve `False`

Source verbatim ([`condacolab.py` línea 320](https://github.com/conda-incubator/condacolab/blob/main/condacolab.py)):

```python
assert find_executable("conda"), "Conda not found!"
```

El patrón `if not condacolab.check(): condacolab.install()` rompe con `AssertionError: Conda not found!` en el primer run. Solución: llamar `install_from_url(URL)` directamente — ya hace `try: check(); except AssertionError: pass` internamente (`condacolab.py` líneas 132–136).

#### 3.1.3 Fallo 3 — Miniforge 23.11.0-0 trae Python 3.12

Observado empíricamente 2026-05-12: tras `condacolab.install_from_url("23.11.0-0/Miniforge3-23.11.0-0-Linux-x86_64.sh")` y kernel restart, `sys.version_info` sigue en 3.12.13.

- Release notes 23.11.0-0 **no especifican** versión de Python explícitamente.
- Hipótesis 1: el installer fue rebuild con Python 3.12 retroactivamente.
- Hipótesis 2: el wrapper de `condacolab` no exec el conda Python correctamente en Colab moderno (`/usr/bin/python3` es ahora un wrapper shell que hace `exec /opt/conda/bin/python`).
- Intento de fix: `mamba install python=3.10` post-restart → **falla** porque `google-colab` está pinneado a 3.12 en el env conda → solver de mamba no encuentra resolución.

#### 3.1.4 Fallos auxiliares confirmados (gotchas informativos)

| # | Gotcha | Fuente |
|---|--------|--------|
| C1 | `condacolab.install_miniforge` NO acepta `python_version`. Firma real: `install_miniforge(prefix, env, run_checks, restart_kernel)` | `condacolab.py` línea 233, rama `main` |
| C2 | `do_shutdown(True)` es **asíncrono** → necesita `sys.exit(0)` después | `install_from_url` líneas 132–136 |
| C3 | Tras kernel restart, NO se mantienen las definiciones de celdas previas | Comportamiento estándar de Jupyter |
| C4 | `list(NEW_SRC)` produce caracteres individuales al asignar `nb['cells'][idx]['source']`. Usar `splitlines(keepends=True)` | Memoria mnemon `a8c6ef5b` |
| C5 | TF 2.16+ rompe TF OD API por removal de `tf.estimator` | Issues [tensorflow/models#13575](https://github.com/tensorflow/models/issues/13575), [#13599](https://github.com/tensorflow/models/issues/13599) |
| C6 | `graph_rewriter` para QAT silently broken en TF2 desde 2021 | Issue [tensorflow/models#9835](https://github.com/tensorflow/models/issues/9835) |

**Decisión vinculante:** Track A abandona Colab. Decisión cerrada. No re-investigar.

### 3.2 Por qué NO Kaggle (decisión cerrada, informativo)

#### 3.2.1 Stack Kaggle GPU v168 (marzo 2026)

Verificado leyendo `Dockerfile.tmpl` y releases v167/v168 de [Kaggle/docker-python](https://github.com/Kaggle/docker-python):

| Componente | Versión mayo 2026 |
|------------|-------------------|
| Python | 3.12 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| CUDA host | 12.8.1 |
| cuDNN | 9.8.0.87 |
| NumPy | **2.4.x (default sin flag para mantener 1.x)** |
| NCCL | 2.25.1+cuda12.8 |

#### 3.2.2 Bloqueadores para Track A

- **Sin soporte TF OD API legacy:** TF 2.15 no tiene wheel cp312 (idem Colab).
- **Mismo problema de Python:** bajar a 3.10 vía conda en Kaggle es aún más fricción que en Colab (no tiene `condacolab` equivalente).
- **Quota GPU 30 h/sem** (vs Colab Pro adaptativa). Aceptable para Track B pero restrictivo para múltiples experimentos.

#### 3.2.3 Bloqueadores para Track B (no fatales, pero frágiles)

- **NumPy 2.x rompe `ultralytics` en runtime** aunque `pyproject.toml` declare `numpy<2.0`. Deps transitivas pueden dejar 2.x. Issue [`ultralytics/ultralytics#22346`](https://github.com/ultralytics/ultralytics/issues/22346) "NumPy 2.2.6 import errors when running on Kaggle T4x2".
- **Conflicts de pip** con preinstalled Kaggle packages: issue [`#22336`](https://github.com/ultralytics/ultralytics/issues/22336) "dependency conflicts when install ultralytics" en Python 3.11.13.
- **PyTorch 2.10 con CUDA 12.8** no afecta el `.onnx` exportado (export con `device='cpu'`), pero introduce variables no controladas en el grafo si Ultralytics regresiona.

#### 3.2.4 Quotas comparativas (informativo)

| Plataforma | Sesión máx | Cuota semanal | RAM | GPU típica |
|------------|-----------|---------------|-----|-----------|
| Colab Free | 12 h | adaptativa | ~12 GB | T4 / K80 compartida |
| Colab Pro | 12 h | mejor disponibilidad | High-RAM | T4 / V100 prioritaria |
| Kaggle GPU | 12 h | 30 h/sem | 29 GB | T4×2 o P100 |
| **Vast.ai (elegida)** | sin cap | sin cuota | 32–128 GB | **RTX 4090 24 GB** |

**Decisión vinculante:** ni Colab ni Kaggle. Vast.ai resuelve los dos tracks con un solo container.

### 3.3 Alternativas si Vast.ai falla (memoria mnemon `1d6d237e`)

1. **RunPod** — limitado a PyTorch 2.6–2.9 (eliminó 2.1 del catálogo en 2025). Útil para Track B si Vast.ai cae temporalmente; **no apto para Track A** (TF 2.15 con torch 2.6+ tiene conflictos de protobuf).
2. **Lambda Labs** — buena GPU disponibility pero menos flexibilidad en imágenes Docker.
3. **Paperspace** — opciones gratuitas con tier limitado.
4. **Cluster UAO `uaodeepia11306`** — si está disponible para uso académico.

### 3.4 Quote del usuario en R4 (motivación de migración)

> *"Quiero mantener las arquitecturas y frameworks que han sido demostradas como estables y compatibles dentro de la Jetson Nano. Usar la mejor GPU. No me importan costos, pero mantener logging robusto y persistencia de TODO en HF Hub."*

### 3.5 Quote del usuario en R5 (motivación de renegociar D4 y D7)

> *"Necesito negociar [...] correrlo como notebook [...] que su ejecución no se afecte porque cerré la pestaña [...] todo sea lo más compatible entre sí (ver uv) [...] garantizar que los productos de esta fase sean compatibles/se puedan usar desde la Jetson Nano de forma estable y robusta."*

Contexto: D4 (jupytext) y D7 (`trap EXIT`) no satisfacían "ejecutar como `.ipynb` con tolerancia a cierre de pestaña". Se renegociaron a D9, D10, D11. Se añadieron D12–D15 para cerrar gaps de validación pre-deploy.

---

## 4. Container y GPU on-demand

### 4.1 Container elegido — `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` (D1)

**Tag exacto:**
```
vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310
```

- `last_pushed`: 2026-03-26
- `tag_status`: active
- Variante `cudnn-devel` → cuDNN 8.9+ incluido (vs `cudnn-runtime` que solo trae libs)
- Python 3.10 base (no 3.12 ni 3.11)
- Ubuntu 22.04 LTS

**Justificación:**

- TF 2.15.0 oficial pide CUDA 12.2 / cuDNN 8.9 según [tensorflow.org/install/source](https://www.tensorflow.org/install/source) → CUDA 12.4 satisface minor compat con drivers 12.x.
- Wheel `tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (PyPI subido 2023-11-14) confirmado existente.
- Python 3.10 evita el cuello de botella de Colab (3.12) sin necesidad de `condacolab`.
- `cudnn-devel` permite compilar contra cuDNN si algún paquete legacy lo requiere (Track A puede requerirlo).
- Un solo container con dos virtualenvs (Track A + Track B) reduce complejidad operativa frente a dos containers separados.

**Alternativa robusta** si la build custom da problemas: imagen oficial `tensorflow/tensorflow:2.15.0-gpu` preinstalada para Track A.

### 4.2 Quirks confirmados de Vast.ai

#### 4.2.1 Naming de imágenes Docker Hub (memoria mnemon `fba73ac3`, 2026-04-16)

- El repo `vastai/pytorch` es **OBSOLETO** (PyTorch 1.0 + CUDA 10.0). No usar.
- Usar `vastai/base-image` para construir imágenes custom.
- Tags compuestos para apps específicas: `vastai/openwebui:v0.5.7-cuda-12.1-pytorch-2.5.1-py311`, `vastai/vllm:v0.8.1-cuda-12.1-pytorch-2.5.1-py312`.
- Patrón de nombres de tags `vastai/base-image`: `cuda-X.Y.Z-cudnn-devel-ubuntu22.04-py310`.

#### 4.2.2 Subcomandos del CLI

- Subcomando `vastai search templates` **NO existe** en el CLI. Los `template_hash` se extraen de la UI `cloud.vast.ai/templates/` y se pasan con `--template_hash`.
- Verificación reciente: `gh api repos/vast-ai/vast-python/releases?per_page=5` confirma el alcance del CLI.

#### 4.2.3 SSH y `--ssh` flag (memoria mnemon `d5f717eb`, 2026-04-16)

- `--ssh` **reemplaza ENTRYPOINT** del container con proceso propio que INYECTA `sshd` desde fuera.
- Si `/usr/sbin/sshd` NO está en la imagen (caso `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`), Vast.ai intenta `apt-get install openssh-server` EN RUNTIME dentro del container.
- Falla con "Connection refused" permanente cuando el host bloquea `archive.ubuntu.com:80` (IP `185.125.190.82`).
- Confirmado por `rolandtannous` (colaborador Vast.ai) en [`unslothai/unsloth#4682`](https://github.com/unslothai/unsloth/issues/4682) (marzo 2026), [`vast-ai/base-image#141`](https://github.com/vast-ai/base-image/issues/141) (abril 2026), [`vast-ai/vast-cli#336`](https://github.com/vast-ai/vast-cli/issues/336) (feb 2026, reproducido en Corea y China).
- **Mitigación para nuestro caso:** la imagen `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` ya incluye `sshd`. No aplica.

#### 4.2.4 Conversión CRLF en Windows (memoria mnemon `27b66a6b`)

- Git en Windows convierte LF → CRLF al checkout por default (`core.autocrlf=true`).
- Los `.sh` y `.py` shipped a Vast.ai Linux tienen `\r\n`, y bash falla con `set: pipefail: invalid option name` porque ve `pipefail\r`.
- **FIX obligatorio:** `.gitattributes` con `*.sh text eol=lf` y `*.py text eol=lf` antes de pushear bootstrap.
- Aplicar también a `.ipynb` y `.json` si se editan en Windows.

### 4.3 GPU RTX 4090 on-demand (D2)

| GPU | Arquitectura | VRAM | TFLOPS FP16 | Precio Vast.ai mayo 2026 |
|-----|--------------|------|-------------|---------------------------|
| **RTX 4090** | **Ada Lovelace `sm_89`** | **24 GB** | **1008** | **0,35–0,50 USD/h on-demand · 0,14–0,31 USD/h spot** |
| A100 40 GB | Ampere `sm_80` | 40 GB | 312 | 0,80–1,40 USD/h on-demand |
| RTX 3090 | Ampere `sm_86` | 24 GB | 285 | 0,25–0,40 USD/h on-demand |
| H100 80 GB | Hopper `sm_90` | 80 GB | 1979 | 2,00+ USD/h on-demand |

**Justificación:** SSD MV2 320 + YOLOv8n 416 son modelos pequeños (≤ 5 M parámetros, batch 32–64 entra en 4 GB VRAM); A100/H100 son desperdicio. Con 1,72 USD de saldo del usuario: 4–12 h de RTX 4090, holgado para 1–3 h de training por track.

**Quote del usuario (R4 verbatim):** *"Usar la mejor GPU. No me importan costos."* Se elige 4090 sobre 5090 por madurez de drivers CUDA 12.4 y disponibilidad consistente.

### 4.4 Credenciales y paths Vast.ai (referencia)

- API key dentro del container: precedencia `--api-key <KEY>` > `$VAST_API_KEY` (env var) > `~/.config/vastai/vast_api_key` (archivo). Inyectar con `--env '-e VAST_API_KEY=<key>'` al crear instancia o con `vastai set api-key <KEY>` en `onstart` (más seguro, escribe al archivo con permisos 600).
- SSH key del proyecto (Windows local): `C:\Users\mitgar14\.ssh\id_ed25519` (memoria mnemon `b13050ac`)
- Binario CLI Windows: `C:\Users\mitgar14\AppData\Roaming\Python\Python312\Scripts\vastai.exe`
- Convención `python3` no `python` en imágenes Vast.ai (memoria mnemon `67358204`)

---

## 5. Track A — TF 2.15 + TFOD API

### 5.1 Decisión de modelo y precisión

| Opción | QAT real | TF | Wheel | FPS estimado CPU TFLite Nano | Verdict |
|--------|----------|----|-------|-------------------------------|---------|
| 1. TFOD API + `graph_rewriter` | ❌ placebo (issue #9835) | 1.15 | Py 3.7 max | N/A | ❌ Descartada |
| 2. MediaPipe Model Maker MOBILENET_V2_I320 | ✅ real preintegrado | 2.15 | Py 3.10 | 10–11 FPS (ajustado) | ⚠️ Plan B (NMS no embebido → decoder custom Nano) |
| **3. TF 2.15 + TFOD API + PTQ post-train (SSD MV2 plain)** | ❌ no QAT (PTQ) | **2.15** | Py 3.10 (vía Vast.ai) | **14–18 FPS** | ✅ **ELEGIDA** |
| 4. TFMOT Keras-style QAT | ✅ real | 2.15+ | Py 3.10+ | N/A | ❌ No soporta SSD FPNLite, sin receta oficial |

**Justificación de la elección (Opción 3):**

- **Compatibilidad runtime:** `TFLite_Detection_PostProcess` se incluye en el export → drop-in en TF 2.5.0+nv21.8 del Nano. No requiere desarrollo adicional de decoder.
- **FPS holgado:** 14–18 FPS estimado vs threshold 10 FPS → margen del 40–80%.
- **mAP aceptable para 3 clases visualmente distintas:** dataset `waste-3class` tiene clases muy diferenciadas (vidrio brillante, papel mate, plástico variado) → la caída de PTQ debería ser pequeña en términos absolutos.
- **Stack maduro:** TF 2.15 + TFOD API + Python 3.10 es la última combinación estable documentada y con tutoriales completos.
- **Plan B factible (Opción 2):** si el mAP de PTQ no alcanza el threshold de calidad, pivotar a MediaPipe Model Maker es viable con el mismo entorno Python 3.10 y requiere ~1–2 días de trabajo adicional para el decoder en Nano.

### 5.2 SSD MV2 plain vs FPNLite (referencia)

| Modelo | Resolución | Precisión | Latencia (Nano TRT GPU) | mAP COCO | mAP@50 |
|--------|-----------|-----------|-------------------------|----------|--------|
| **SSD MV2 plain** | 320×320 | FP32 | 26,96 ms | 20,18 | 34,74 |
| **SSD MV2 plain** | 320×320 | FP16 | 24,31 ms | 20,18 | 34,74 |
| SSD MV2 FPNLite | 320×320 | FP32 | 39,54 ms | 21,97 | 36,95 |
| SSD MV2 FPNLite | 320×320 | FP16 | 37,99 ms | 21,97 | 36,95 |

Fuente: [NobuoTsukamoto/benchmarks](https://github.com/NobuoTsukamoto/benchmarks) (Jetson Nano B01 JetPack 4.6.1 + TRT). Gap mAP@50: **+2,21 pp para FPNLite, 47% más lento.**

**Drop empírico esperado PTQ INT8:**

- Jacob et al. CVPR 2018 ([arXiv:1712.05877](https://arxiv.org/abs/1712.05877)): QAT mantiene caída < 1,5 pp mAP.
- Karimov et al. 2025 ([arXiv:2508.19600](https://arxiv.org/abs/2508.19600)): PTQ INT8 cae 3–7 pp mAP50-95.
- Diferencia neta esperada PTQ − QAT ≈ 2–5 pp en favor de QAT.

**Confirmación dominio específico** (Trisuwita et al. 2024, [doi:10.34010/komputika.v13i1.10333](https://doi.org/10.34010/komputika.v13i1.10333)): en detección de cascos, SSD MV2 plain (80,12% mAP) supera al FPNLite (71,59%). En datasets pequeños balanceados con clases visualmente distintas, FPN puede no aportar.

### 5.3 Pin de dependencias Track A (kernel `tracka`)

```bash
# Activación
source /opt/venv/tracka/bin/activate

# Stack core
uv pip install tensorflow==2.15.0 \
              tf-models-official==2.15.0 \
              "tensorflow-model-optimization>=0.7.5,<0.8.0" \
              "numpy==1.26.4" \
              "protobuf==3.20.3" \
              "Pillow==10.4.0"

# Imagen y tooling
uv pip install "opencv-python-headless==4.10.0.84" \
              "pycocotools==2.0.7" \
              "lvis==0.5.3" \
              "tensorflow-addons==0.23.0" \
              "tensorflow-text==2.15.0"

# Compilador de protos compatible con protobuf 3.20.x
uv pip install grpcio-tools==1.64.1

# HF Hub + ipykernel
uv pip install huggingface_hub ipykernel
python -m ipykernel install --user --name tracka --display-name "Track A (TF 2.15)"
```

### 5.4 Clone TF Models en pin SHA `9cafa3d150`

```bash
# Ronda 2 vinculante: clonar master y checkout a SHA pre-Pillow12-patch
git clone --filter=blob:none --no-checkout \
  https://github.com/tensorflow/models.git /workspace/tf_models
cd /workspace/tf_models
git checkout 9cafa3d150
# Validación dura
test -d research/object_detection || { echo "ERROR: research/ no existe"; exit 1; }
```

**Justificación del pin SHA:**

- **Tag `v2.15.0` de `tensorflow/models` NO contiene `research/`** — release note verbatim: *"Note that Research/tutorial/sample models have been removed."* Solo `official/` + `orbit/`.
- **Patch que rompe Pillow 10.4:** commit `971ded9e16` (2026-04-29, *"Support Pillow 12's stricter type checking in Image.fromarray"*).
- **Último commit estable pre-Pillow12-patch:** **`9cafa3d150`** (2026-03-17, *"Merge PR #13619 ai-gsutil-migration"*).
- Pin estable alternativo: `f9fdc4faef47af76351204b6d8df576f0e79baab` (2024-06-07, pre-NumPy-2.0).

### 5.5 Compilación de protos (evitar `runtime_version` bug)

```bash
cd /workspace/tf_models/research

# protoc 5.x+ genera `from google.protobuf import runtime_version as _runtime_version`
# que NO existe en protobuf 3.20.x. Usar grpcio-tools (protoc 4.x bundled).
python -m grpc_tools.protoc \
  --python_out=. \
  --proto_path=. \
  object_detection/protos/*.proto

# Instalar OD API sin deps (evita que apache-beam traiga protobuf 5.x)
cp object_detection/packages/tf2/setup.py .
pip install --no-deps -e .

# Red de seguridad: re-pin defensivo + env var
pip install --force-reinstall --no-deps "Pillow==10.4.0" "protobuf==3.20.3"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # parser puro Python
```

**`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`:** red de seguridad robusta contra el bug `runtime_version`. Pequeño overhead (5–10x más lento que C++), negligible para training overhead.

**Shim `PIL.Image.ANTIALIAS = LANCZOS`:** `ANTIALIAS` deprecado en Pillow 9.1, removido en Pillow 10.0. Si código legacy de `research/object_detection/utils/visualization_utils.py` lo invoca:

```python
import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
```

Confirmado en commit `a01d02f9` de `tensorflow/hub` (julio 2023).

### 5.6 Tabla `op_version` TFLite 2.5 max vs TF 2.15 export

| Op | Versión máx TFLite 2.5 | Versión generada TF 2.15 INT8 PTQ | Riesgo | Issues asociados |
|----|------------------------|------------------------------------|--------|------------------|
| `CONV_2D` | 5 | 5 con activaciones estándar | Bajo | [#41943](https://github.com/tensorflow/tensorflow/issues/41943), [#50652](https://github.com/tensorflow/tensorflow/issues/50652), [#43232](https://github.com/tensorflow/tensorflow/issues/43232) |
| `DEPTHWISE_CONV_2D` | 4 | 4–5 según flags | **Medio** | (sin issue público concluyente) |
| `FULLY_CONNECTED` | 4 | 4 para MobileNet v2 | Bajo | [#80736](https://github.com/tensorflow/tensorflow/issues/80736) (ejemplo análogo TF 2.17 → 2.5) |
| `QUANTIZE` / `DEQUANTIZE` | 2 | 2 | Bajo | — |
| `PAD`, `ADD`, `MUL`, `RESHAPE`, `CONCATENATION` | 1–2 | 1–2 | Ninguno | — |
| `TFLite_Detection_PostProcess` | custom | custom (post-process embebido) | Ver D15 ledger | — |
| `Cast v2+` | (requiere TF 2.7+) | (debería ser v1) | Bajo si NMS embebido evita Cast | — |
| `BatchMatMul v5+` | (requiere TF 2.6+) | (no en MV2 SSD plain) | Ninguno | — |

**Error de runtime esperado si `op_version` excede 2.5 max:**

```
ERROR: Didn't find op for builtin opcode 'CONV_2D' version '5'.
An older version of this builtin might be supported.
Are you using old TFLite binary with newer model?
```

**Flags conservadores del converter** (D12 del ledger):

```python
converter.experimental_new_quantizer = False  # cuantizador legacy
converter.experimental_new_converter  = False # converter TOCO legacy
```

Tradeoff: el converter legacy puede producir cuantización de menor calidad, pero garantiza compatibilidad con runtimes antiguos.

**Workaround vía `flatbuffer_utils.py`:** archivo [`tensorflow/lite/tools/flatbuffer_utils.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/flatbuffer_utils.py) permite parsear y reescribir flatbuffers. No existe script oficial de downgrade de versión, pero es técnicamente posible bajar manualmente el campo `version` de cada operador. Frágil, no documentado oficialmente.

### 5.7 MediaPipe Model Maker (Plan B, informativo)

Restricciones del wheel `mediapipe-model-maker`:

- `requirements.txt` verbatim: `tensorflow>=2.10,<2.16`, `tf-models-official>=2.13.2,<2.16.0`, `tensorflow-model-optimization<0.8.0`.
- `setup.py` Python classifiers: **3.8, 3.9, 3.10** (no 3.11/3.12).
- `MOBILENET_V2_I320` usa checkpoint `gs://tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i320_ckpt.tar.gz` — QAT preintegrado.
- **NMS omitido en TFLite export:** `tflite_post_processing=configs.common.TFLitePostProcessingConfig(omit_nms=True)` → el `.tflite` no contiene `TFLite_Detection_PostProcess`. Requiere decoder custom (decoding de anchors + NMS) en Nano.
- Dataset solo COCO / PASCAL VOC. Roboflow exporta ambos formatos → conversión trivial desde TFRecord.
- Caveat de accuracy: [`discuss.ai.google.dev/t/mediapipe-massive-accuracy-loss-with-quantization-aware-training/23177`](https://discuss.ai.google.dev/t/mediapipe-massive-accuracy-loss-with-quantization-aware-training/23177).

**No activar como Plan B sin nueva ronda `/investiga`.**

---

## 6. Track B — YOLOv8n + Ultralytics

### 6.1 Decisión de export (R1 cerrado)

**Veredicto:** mantener YOLOv8n 416×416 + Ultralytics ≥ 8.4.46 + ONNX opset 11 explícito + onnxslim ≥ 0.1.82 + FP16 TRT engine construido en Nano + NMS en CPU NumPy.

Razones:

- **Kaggle/Colab traen PyTorch 2.9–2.10 con CUDA 12 host.** No afecta el `.onnx` exportado con `device='cpu'` (PyTorch exporta por trazado en CPU; el ONNX es portátil).
- **ONNX opset 11 es soportado por TRT 8.2.1.** [`onnx-tensorrt/docs/operators.md?ref=release/8.2-GA`](https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md) verbatim: *"TensorRT 8.2 supports operators up to Opset 13."*
- **Ultralytics 8.4.x default opset es 20 con torch 2.9+.** Hay que forzar `opset=11` explícitamente.
- **`onnxsim==0.4.36` no compila en Python 3.12** (issue [`#334`](https://github.com/daquexian/onnx-simplifier/issues/334) `daquexian/onnx-simplifier`). Pin a `>=0.6.2,<0.7`.
- **Ultralytics 8.3+ migró de `onnxsim` a `onnxslim`** (verificado en source de `ultralytics/engine/exporter.py`). El flag `simplify=True` llama `onnxslim.slim(model_onnx)`. Pin `onnxsim` es irrelevante para el exporter.
- **`EfficientNMS_TRT` plugin no funciona estable en Maxwell** (issue [`NVIDIA/TensorRT#1538`](https://github.com/NVIDIA/TensorRT/issues/1538)) → NMS en CPU NumPy con `cv2.dnn.NMSBoxes` o `torchvision.ops.nms` en el Nano.

### 6.2 Pin de dependencias Track B (kernel `trackb`)

```bash
# Activación
source /opt/venv/trackb/bin/activate

# Defensa contra NumPy 2.x ANTES de ultralytics
uv pip install "numpy<2.0"

# PyTorch + CUDA 12.1 (CUDA del host 4090 es 12.4 compatible)
uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Ultralytics + onnxslim (el real, no onnxsim)
uv pip install "ultralytics>=8.4.46,<8.5" \
              "onnxslim>=0.1.82" \
              "onnx>=1.16,<1.18" \
              "onnxruntime>=1.18,<1.21"

# Dataset + logging + tooling
uv pip install "roboflow>=1.3.6,<1.4" \
              wandb huggingface_hub ipykernel

python -m ipykernel install --user --name trackb --display-name "Track B (YOLOv8)"

# Validación post-install
python - <<'EOF'
import numpy, torch, ultralytics, onnx, onnxslim, roboflow
assert int(numpy.__version__.split('.')[0]) < 2, f"NumPy {numpy.__version__} viola numpy<2.0"
print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"ONNX: {onnx.__version__}, onnxslim: {onnxslim.__version__}")
print(f"Roboflow: {roboflow.__version__}")
EOF
```

### 6.3 Hyperparameters de training YOLOv8n (canónico)

```python
from ultralytics import YOLO
import os

# Asegurar W&B autenticado (D6)
assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY no presente — set en bootstrap"

# Cargar modelo pretrained COCO
model = YOLO("yolov8n.pt")  # descarga automática desde Ultralytics CDN

# Training
results = model.train(
    data="/workspace/embebidos-3/datasets/waste-3class-1/data.yaml",

    # === Geometría ===
    imgsz=416,                  # PR #24028 fix; resolución Track B
    rect=False,                 # rectangular training off (dataset ya stretch 416×416)

    # === Schedule ===
    epochs=100,                 # arrancar con 100; early stopping vía patience
    patience=20,                # detener si val no mejora 20 epochs
    batch=32,                   # 4090 24 GB holgado para YOLOv8n 416
    workers=8,                  # data loader threads

    # === Optimizer ===
    optimizer="auto",           # selecciona SGD/AdamW según task
    lr0=0.01,
    lrf=0.01,                   # final lr = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,

    # === Augmentations (training) ===
    mosaic=1.0,                 # 4-image mosaic 100% epochs (off al final)
    close_mosaic=10,            # desactivar mosaic en últimas 10 epochs
    mixup=0.15,                 # 15% chance mixup
    fliplr=0.5,                 # 50% horizontal flip
    flipud=0.0,                 # NO vertical flip (waste-3class no es invariante)
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0,                # rotation off (Roboflow ya aplicó ±15°)
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,

    # === Logging ===
    project="embebidos-3",
    name="track_b_yolov8n",
    save=True,
    save_period=10,             # checkpoint cada 10 epochs
    plots=True,
    wandb=True,                 # D6 — W&B nativo

    # === Reproducibilidad ===
    seed=42,
    deterministic=True,

    # === Device ===
    device=0,                   # GPU id (single GPU)
    amp=True,                   # mixed precision FP16 training (no afecta export)
)
```

### 6.4 Justificación de los augmentations

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `mosaic=1.0` | 4-img mosaic 100% epochs | Augmentation estándar Ultralytics; mejora robustez a contextos diversos |
| `close_mosaic=10` | desactivar en últimas 10 epochs | Mosaic distorsiona objetos; al final mejor train sin él para que el modelo aprenda detección "limpia" |
| `mixup=0.15` | 15% chance mixup | Pequeño boost de generalización en clases visualmente distintas |
| `fliplr=0.5` | 50% flip horizontal | Waste invariante a flip horizontal (vidrio brillante igual de izquierda o derecha) |
| `flipud=0.0` | NO vertical flip | Etiquetas se invierten al voltear vertical (top → bottom) — datos perderían validez espacial |
| `degrees=0.0` | NO rotación | Roboflow ya aplicó ±15° en augmentation. Doble rotación destruiría señal |
| `translate=0.1` | 10% translación | Mejora robustez a objetos no centrados |
| `scale=0.5` | escala ±50% | Mejora robustez a tamaño del objeto en cámara |

### 6.5 Export ONNX canónico (pre-D13)

```python
from ultralytics import YOLO
model = YOLO('runs/detect/track_b_yolov8n/weights/best.pt')

# Export ONNX con flags explícitos
output_path = model.export(
    format='onnx',
    opset=11,           # CRÍTICO: default 20 con torch 2.9+; TRT 8.2 max opset 13
    simplify=True,      # llama onnxslim.slim() internamente (Ultralytics 8.3+)
    dynamic=False,      # shapes fijas para TRT engine determinístico
    imgsz=416,          # PR #24028 fix INT8 calib non-square (cuadrado, OK)
    device='cpu',       # CUDA host irrelevante para .onnx (export por trazado en CPU)
    half=False,         # FP16 se aplica en trtexec en Nano, no en ONNX
    int8=False,         # FP16-only por D14 (sin dp4a en sm_53)
    nms=False,          # NMS en CPU NumPy en Nano (EfficientNMS_TRT roto Maxwell #1538)
)
print(f"✅ ONNX exportado: {output_path}")
# Output: runs/detect/track_b_yolov8n/weights/best.onnx
```

### 6.6 Justificación verbatim de cada flag del export

| Flag | Razón | Issue / PR de referencia |
|------|-------|--------------------------|
| `opset=11` | TRT 8.2-GA soporta hasta opset 13; opset 11 es conservador y evita issue [`NVIDIA/TensorRT#4383`](https://github.com/NVIDIA/TensorRT/issues/4383) (Gather rank-0 bug con opset ≥17) | `onnx-tensorrt operators.md release/8.2-GA`, PR Ultralytics #23808 |
| `simplify=True` | Reduce nodos del grafo (constant folding, dead-code elimination) → engine TRT más compacto y rápido | Default Ultralytics 8.3+ usa `onnxslim` |
| `dynamic=False` | Shapes fijas → `ConstantOfShape` genera initializers FP32 (TRT 8.2 solo soporta FP32 en esta op); engine TRT determinístico | onnx-tensorrt operators.md `ConstantOfShape` restriction |
| `imgsz=416` | Resolución entrenada, debe coincidir en export | PR Ultralytics #24028 (INT8 calib non-square fix; no afecta 416×416 cuadrado pero pin defensivo) |
| `device='cpu'` | PyTorch exporta ONNX por trazado en CPU; el CUDA del host es irrelevante para el `.onnx` resultante | Ultralytics docs export |
| `half=False` | FP16 se aplica al compilar engine TRT (`trtexec --fp16` en Nano), NO en ONNX | Validación pre-deploy §23 |
| `int8=False` | FP16-only por D14 (Maxwell sin `dp4a` → INT8 sin speedup) | Decisión ledger D14 |
| `nms=False` | `EfficientNMS_TRT` plugin roto en Maxwell con TRT 8.x; NMS hecho en CPU NumPy con `cv2.dnn.NMSBoxes` en el Nano | Issue [`NVIDIA/TensorRT#1538`](https://github.com/NVIDIA/TensorRT/issues/1538) |

### 6.7 Verificación post-export (Gate 3 input)

```python
import onnx
m = onnx.load('runs/detect/track_b_yolov8n/weights/best.onnx')

# Validaciones duras
assert m.opset_import[0].version == 11, \
    f"❌ Opset {m.opset_import[0].version} ≠ 11 — flag `opset=11` no aplicado"
assert m.ir_version <= 10, \
    f"❌ IR version {m.ir_version} > 10 puede romper TRT 8.2 / onnxslim 0.6.x"

ops_present = sorted({n.op_type for n in m.graph.node})
print(f"✅ Opset: {m.opset_import[0].version}, IR: {m.ir_version}")
print(f"✅ Ops únicos en grafo: {ops_present}")

# Sanity check input shape
assert m.graph.input[0].type.tensor_type.shape.dim[2].dim_value == 416
assert m.graph.input[0].type.tensor_type.shape.dim[3].dim_value == 416
print(f"✅ Input shape: [1, 3, 416, 416]")
```

### 6.8 Por qué pinear Ultralytics `>=8.4.46,<8.5`

- **PR [`ultralytics/ultralytics#24028`](https://github.com/ultralytics/ultralytics/pull/24028)** "INT8 calibration non-square imgsz fix" merged 2026-03-28 en v8.4.31. Bug previo: si `imgsz` no era cuadrado (e.g., 640×480), la calibración INT8 producía resultados inválidos. Como usamos `imgsz=416` (cuadrado), el bug no nos afecta directamente, pero el pin garantiza que si en el futuro se cambia `imgsz` a no-cuadrado, no regresione.
- **v8.4.48** (2026-05-08) es la última release a 2026-05-12 (verificado con `gh api repos/ultralytics/ultralytics/releases?per_page=10`). El pin `<8.5` captura v8.4.46 → v8.4.48 sin permitir saltos a 8.5 (que aún no existe).
- **No existe 8.5** a fecha 2026-05-12; el cap es defensivo contra breaking changes futuros.

### 6.9 Por qué `numpy<2.0` ANTES de `ultralytics`

`ultralytics/pyproject.toml` declara `numpy<2.0.0` con comentario *"TF 2.20 compatibility"*. Heredado automáticamente al instalar `ultralytics`, pero en Kaggle/Colab base la cadena de deps transitivas puede dejar NumPy 2.x ya instalado **antes** de que ultralytics se evalúe. Resultado: NumPy 2.4.x quedó instalado y `import ultralytics` falla con:

```
ImportError: numpy.core.multiarray failed to import
```

Confirmado en issue [`ultralytics/ultralytics#22346`](https://github.com/ultralytics/ultralytics/issues/22346) "Installation + runtime failure: pip dependency conflicts & NumPy 2.2.6 import errors when running on Kaggle T4x2".

**Defensa robusta:** instalar `numpy<2.0` **explícitamente ANTES** de `ultralytics`:

```bash
uv pip install "numpy<2.0"  # PRIMERO
uv pip install "ultralytics>=8.4.46,<8.5"  # DESPUÉS
```

En Vast.ai con `uv venv` limpio el problema no debería ocurrir (no hay deps preinstaladas), pero el pin explícito es defensa en profundidad.

### 6.10 `onnxslim` (no `onnxsim`) — Ultralytics 8.3+

Verificado leyendo source de [`ultralytics/engine/exporter.py`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py) y [`pyproject.toml`](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml) rama `main` 2026-05-12:

Desde Ultralytics **8.3+**, `model.export(simplify=True)` llama internamente a `onnxslim.slim(model_onnx)`, **NO** a `onnxsim`. El pin `onnxsim` en `pyproject.toml` del notebook (heredado de templates antiguos) es **irrelevante** para el exporter.

| Herramienta | Estado | Compatibilidad Py 3.12 | Usado por Ultralytics 8.3+ |
|-------------|--------|------------------------|-----------------------------|
| `onnxsim` (`daquexian/onnx-simplifier`) | Mantenimiento de comunidad | 0.4.x ❌ no compila (issue [`#334`](https://github.com/daquexian/onnx-simplifier/issues/334)); 0.6.x ✅ wheels manylinux | ❌ No (deprecado) |
| **`onnxslim`** (`inisis/OnnxSlim`) | **Mantenimiento activo** | ✅ Sí | ✅ **Sí (default desde 8.3)** |

**Pin recomendado:**

```bash
uv pip install "onnxslim>=0.1.82"
# NO usar onnxsim, mantener solo si código custom lo necesita
```

**Breaking changes históricos `onnxsim 0.5.0+` (informativo):**

Si por alguna razón el código de usuario llama `onnxsim` directamente (no vía Ultralytics):

- `--dynamic-input-shape` removido (era flag deprecated)
- `--input-shape` → `--overwrite-input-shape`
- `--enable-fuse-bn` removido (default `True` ahora)

### 6.11 Stack moderno Ultralytics (referencia)

PRs recientes relevantes (R3 R4):

- **PR [`#23807`](https://github.com/ultralytics/ultralytics/pull/23807)** (2026-03-05): Docker base actualizado a `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime`. Confirma torch 2.10 como runtime moderno. **No usamos torch 2.10**: pinneamos a 2.1.0+cu121 por compat con TRT 8.2 en validación.
- **PR [`#23808`](https://github.com/ultralytics/ultralytics/pull/23808)** (2026-03-05): "safer ONNX opset cap for Torch 2.9+ exports". Protección adicional al pin `opset=11` explícito del notebook.

### 6.12 Gates de pre-train

Antes de lanzar `model.train()` validar (en una celda anterior del notebook):

```python
# Pre-train sanity check
import yaml
from pathlib import Path

DATA_YAML = Path("/workspace/embebidos-3/datasets/waste-3class-1/data.yaml")
assert DATA_YAML.exists(), f"data.yaml no encontrado en {DATA_YAML}"

with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

# Validar 3 clases
assert cfg["nc"] == 3, f"nc={cfg['nc']} ≠ 3 (issue #88)"
assert sorted(cfg["names"]) == ["glass", "paper", "plastic"]

# Validar splits con ≥ 30 imágenes (≈ 10 por clase mínimo)
for split in ["train", "val"]:
    split_path = Path(cfg.get(split, ""))
    if not split_path.is_absolute():
        split_path = DATA_YAML.parent / split_path
    n_imgs = len(list(split_path.parent.glob("**/*.jpg"))) + len(list(split_path.parent.glob("**/*.png")))
    assert n_imgs >= 30, f"{split} tiene solo {n_imgs} imágenes"
    print(f"✅ {split}: {n_imgs} imágenes")

print("✅ Pre-train sanity OK — proceder con model.train()")
```

### 6.13 Gates de post-train

Antes de exportar a ONNX, validar el `.pt`:

```python
# Post-train sanity
results = model.val()  # corre val split
assert results.box.map50 > 0.5, f"mAP50={results.box.map50:.3f} < 0.5 — modelo no aprendió"

# fuse() no debe errorear
fused = model.fuse()
print(f"✅ mAP50={results.box.map50:.3f}, mAP50-95={results.box.map:.3f}, fuse OK")

# Mejor checkpoint disponible
best_pt = Path("runs/detect/track_b_yolov8n/weights/best.pt")
assert best_pt.exists(), "best.pt no generado"
```

### 6.14 Upload final a HF Hub

Tras validación exitosa de Gates 3 y 4 (ver §19, §20):

```python
from huggingface_hub import HfApi
import json
from pathlib import Path

EXPORT_DIR = Path("/workspace/embebidos-3/track_b/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Copiar artefactos
import shutil
shutil.copy("runs/detect/track_b_yolov8n/weights/best.pt", EXPORT_DIR / "best.pt")
shutil.copy("runs/detect/track_b_yolov8n/weights/best.onnx", EXPORT_DIR / "best.onnx")

# Manifest
manifest = {
    "model": "YOLOv8n",
    "track": "B",
    "imgsz": 416,
    "classes": ["glass", "paper", "plastic"],
    "opset": 11,
    "ir_version": 10,
    "nms": "external (CPU NumPy)",
    "precision_export": "FP32",
    "precision_deploy": "FP16",  # aplicado en Nano via trtexec --fp16
    "dataset": "embebidos3/waste-3class-lwld8 v1-B",
    "ultralytics_version": "8.4.46",
    "training_run": "track_b_yolov8n",
    "metrics": {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    },
    "wandb_run": "<URL del run W&B>",
}
(EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

# Push asíncrono
api = HfApi()
future = api.upload_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path=str(EXPORT_DIR),
    path_in_repo="track_b/exports",
    commit_message=f"Track B — best.onnx (mAP50={results.box.map50:.3f})",
    run_as_future=True,
)
print(f"Upload iniciado: {future}")
result = future.result(timeout=600)
print(f"✅ Upload completado: {result}")
```

---

## 7. Dataset Roboflow

### 7.1 Identificación del dataset

| Campo | Valor |
|-------|-------|
| **Workspace** | `embebidos3` |
| **Project slug** | `waste-3class-lwld8` |
| **Project type** | Object Detection |
| **Clases** | 3: `glass`, `paper`, `plastic` |
| **IDs de clase** | 0=glass, 1=paper, 2=plastic (ordenamiento alfabético YOLOv8) |
| **Version usada** | **Version 1-B**, formato `yolov8`, resolución 416×416 |

> **Nota:** el ID exacto del dataset puede no coincidir si el usuario re-curó/exportó tras Ronda 1. Validar antes del primer training con:
>
> ```bash
> python -c "
> import os; os.environ['ROBOFLOW_API_KEY']='<KEY>'
> from roboflow import Roboflow
> rf = Roboflow()
> print([w for w in rf.workspaces])
> "
> ```

### 7.2 Spec de la Version 1-B (preprocessing + augmentations)

**Preprocessing:**

| Stage | Configuración |
|-------|---------------|
| Auto-orient | Activado (corrige EXIF rotation tags) |
| Resize | Stretch a **416×416** (no `Fit (white edges)` para evitar márgenes blancos en train) |
| Tile | (deshabilitado) |
| Modify classes | (sin remapeo: 3 clases tal cual) |

**Augmentation** (aplicado solo a train split, no val/test):

| Augmentation | Valor / rango |
|--------------|---------------|
| Flip horizontal | ✅ activado |
| Flip vertical | ❌ desactivado (waste-3class no es invariante a vertical flip — etiquetas se invierten) |
| 90° rotation | ❌ desactivado |
| Crop (zoom) | ±15% |
| Rotation | ±15° |
| Shear | ±5° H, ±5° V |
| Brightness | ±25% |
| Exposure | ±15% |
| Blur | hasta 1,5 px |
| Noise | hasta 2% pixels |
| Mosaic (Roboflow) | ❌ desactivado (Ultralytics aplica su propio mosaic en training, mejor calidad) |

**Generate version:** 3 outputs (3× train images con augmentations aplicadas). Split 70/20/10 (train/valid/test) por default Roboflow.

### 7.3 Variables de entorno requeridas

```bash
# .env (gitignored, NO commitear)
ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxxxxxxxx

# Set ANTES de instanciar Roboflow() para forzar dir de descarga
DATASET_DIRECTORY=/workspace/embebidos-3/datasets
```

Sin `DATASET_DIRECTORY` set, el SDK Roboflow descarga al CWD del proceso Python — que en JupyterLab + tmux + `nbconvert` puede no ser el directorio del notebook (es el dir donde se invocó `jupyter nbconvert`).

### 7.4 Bug crítico `dataset.location` y workaround

#### 7.4.1 Causa raíz

Source de [`roboflow/core/version.py`](https://github.com/roboflow/roboflow-python/blob/main/roboflow/core/version.py) (verificado en v1.3.9, SHA `1e4cbc04`):

```python
def download(self, model_format=None, location=None, overwrite: bool = False):
    if location is None:
        location = self.__get_download_location()  # ← devuelve path RELATIVO al CWD
```

Cuando `location is None`, `__get_download_location()` retorna algo como `"./waste-3class-1"` (relativo). Combinado con `os.chdir` interno o llamadas posteriores que cambian el CWD, el dataset termina en una ubicación inesperada.

#### 7.4.2 Bug residual cuando `location` SÍ se pasa

Aunque el usuario pase `location` explícito, el `data.yaml` generado por el SDK contiene paths **relativos** que asumen ubicación distinta. Issue [`roboflow/roboflow-python#240`](https://github.com/roboflow/roboflow-python/issues/240) verbatim:

> *"RuntimeError: Dataset 'TennisBallTracker-9/data.yaml' error [...] missing path '/.../datasets/TennisBallTracker-9/TennisBallTracker-9/valid/images'"*

El path resultante tiene **el slug del proyecto duplicado** (`TennisBallTracker-9/TennisBallTracker-9/...`) porque el `data.yaml` no respeta el `location` pasado.

#### 7.4.3 Issue #88 — clases fantasma en `data.yaml`

Issue [`roboflow/roboflow-python#88`](https://github.com/roboflow/roboflow-python/issues/88) (open desde 2022-12-21, **sin fix** a 2026-05-12): si se eliminan clases en Roboflow Universe sin regenerar la version, el `data.yaml` puede contener `names: [glass, paper, plastic, deprecated_class]` con `nc: 4` mal. **Mitigación:** validar `nc:` y `names:` manualmente post-download.

#### 7.4.4 Workaround completo (cascada de búsqueda)

Patrón canónico aplicado en `notebooks/train_track_b_yolov8.ipynb` cell-10:

```python
# === Track B — cell-10 Roboflow download con workaround ===
import os
import shutil
from pathlib import Path
from roboflow import Roboflow

# CRÍTICO: set ANTES de Roboflow()
WORK_DIR = Path("/workspace/embebidos-3/datasets")
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.environ["DATASET_DIRECTORY"] = str(WORK_DIR)

# Forzar CWD conocido ANTES de version.download()
os.chdir(WORK_DIR)

# Descargar
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("embebidos3").project("waste-3class-lwld8")
version = project.version(1)  # Version 1-B
ds = version.download("yolov8", location=str(WORK_DIR / "waste-3class-1"), overwrite=False)
# ds.location es lo que el SDK CREE haber descargado

# === Cascada de búsqueda para data.yaml ===
# (a) primero en ds.location (lo más confiable si el SDK no rompió)
# (b) luego en WORK_DIR (si el SDK lo descargó al CWD por bug)
# (c) fallback global con heurística por contenido

def find_data_yaml(ds_location: str, work_dir: Path) -> Path:
    """Busca data.yaml con cascada y heurística."""
    candidates = [
        Path(ds_location) / "data.yaml",                        # (a)
        work_dir / "waste-3class-1" / "data.yaml",              # (b1)
        work_dir / "data.yaml",                                  # (b2)
    ]

    # Heurística por contenido (c) — buscar globalmente cualquier data.yaml con las 3 clases
    if not any(c.exists() for c in candidates):
        print("[fallback] Buscando data.yaml globalmente con heurística glass+paper+plastic...")
        for yaml_path in Path("/workspace").rglob("data.yaml"):
            content = yaml_path.read_text(errors="ignore").lower()
            if all(cls in content for cls in ["glass", "paper", "plastic"]):
                print(f"[fallback] Encontrado: {yaml_path}")
                candidates.insert(0, yaml_path)
                break

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"data.yaml no encontrado en cascada:\n" +
        "\n".join(f"  - {c}" for c in candidates)
    )

DATA_YAML = find_data_yaml(ds.location, WORK_DIR)
print(f"✅ DATA_YAML resuelto: {DATA_YAML}")

# === Si está fuera del WORK_DIR esperado, copiar/mover ===
EXPECTED_DIR = WORK_DIR / "waste-3class-1"
if not str(DATA_YAML).startswith(str(EXPECTED_DIR)):
    print(f"[fix] Copiando dataset a {EXPECTED_DIR}...")
    src_root = DATA_YAML.parent
    shutil.copytree(src_root, EXPECTED_DIR, dirs_exist_ok=True)
    DATA_YAML = EXPECTED_DIR / "data.yaml"

# === Validación post-download (defensa contra issue #88) ===
import yaml
with open(DATA_YAML) as f:
    cfg = yaml.safe_load(f)

assert cfg["nc"] == 3, f"nc esperado 3, got {cfg['nc']} — issue #88 (clases fantasma)"
assert sorted(cfg["names"]) == ["glass", "paper", "plastic"], \
    f"Clases inesperadas: {cfg['names']}"
print(f"✅ data.yaml válido: nc={cfg['nc']}, names={cfg['names']}")

# === Validación de splits (cada uno con >= 10 img por clase) ===
ROOT = DATA_YAML.parent
for split in ["train", "valid", "test"]:
    img_dir = ROOT / split / "images"
    if img_dir.exists():
        n = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        print(f"   {split}: {n} imágenes")
        if split in ["train", "valid"]:
            assert n >= 30, f"{split} tiene solo {n} imágenes (< 30 = < 10 por clase)"
```

### 7.5 Estado del bug en versiones recientes (R3 verificado)

- **Roboflow SDK v1.3.9 (2026-05-07, SHA `1e4cbc04`):** SIN fix. Releases 1.3.7–1.3.9 enfocados en soft-delete y device-management, no en path resolution.
- **Roboflow SDK v1.3.6:** mismo bug, sin fix.
- **Workaround cascada de cell-10 es obligatorio** hasta nueva ronda `/investiga` que confirme fix.

Issues relacionados (todos open):

- [`#125`](https://github.com/roboflow/roboflow-python/issues/125) "data.yaml file has different references for image paths" (2022)
- [`#240`](https://github.com/roboflow/roboflow-python/issues/240) "Incorrect Data Path in YOLOv8 Dataset Configuration" (2024)
- [`#306`](https://github.com/roboflow/notebooks/issues/306) "dataset.location empty" (en `roboflow/notebooks`)
- [`#333`](https://github.com/roboflow/roboflow-python/issues/333) "Issue with relative paths in data.yaml file when trying to train yolo custom model"
- [`#108`](https://github.com/roboflow/roboflow-python/issues/108) ".download() re-downloads the same version even if it already exists on disk"

### 7.6 PR #113 (fix histórico, no aplicable a nuestro caso)

[`roboflow/roboflow-python#113`](https://github.com/roboflow/roboflow-python/pull/113) "Fix for v8>=8.0.29 breaking changed to dataset loader" — fix de 2023 para una breaking change distinta de Ultralytics 8.0.30. Mencionado por completitud, no resuelve nuestro caso.

---

## 8. Dual venv uv

### 8.1 Por qué dos virtualenvs separados (D10)

Las dependencias son intrínsecamente incompatibles:

- **Track A** requiere TF 2.15 + `protobuf==3.20.3` + Pillow 10.4 + numpy 1.26 + grpcio-tools 1.64.1.
- **Track B** requiere PyTorch 2.1+cu121 + Ultralytics 8.4.46 + `numpy<2.0` (deps ultralytics) + onnxslim.

El solver de `pip`/`uv` no encuentra una resolución única porque `protobuf 3.20.3` (pin Track A) entra en conflicto con deps transitivas de `torch 2.1+cu121` que arrastran `protobuf>=4`. Y `numpy<2.0` de Ultralytics colisiona con TF 2.15 que pide `numpy==1.26.4` específicamente.

### 8.2 Tres opciones evaluadas

| Opción | Lockfile | Aislamiento | Complejidad | Recomendado |
|--------|----------|-------------|-------------|-------------|
| Monorepo único con `[project.optional-dependencies]` + `[tool.uv] conflicts` + `[tool.uv.sources]` por extra | Único, compartido | Mismo venv, mismo proceso | Alta | ❌ Overhead para proyecto académico; deps incompatibles fundamentalmente |
| Dos `pyproject.toml` separados invocados con `uv sync --project <archivo>` | Independiente | Venvs separados | Media | ❌ **GAP:** `--project` acepta directorios, no archivos. No soportado |
| **Dos `uv venv` separados con `uv pip install`** | Sin lockfile (deps explícitas en bootstrap) | Procesos completamente aislados | Baja | ✅ **ELEGIDA (D10)** |

### 8.3 Capacidades de uv (referencia)

- **Workspaces** (`[tool.uv.workspace]`): inspirado en Cargo, `uv.lock` compartido — [docs.astral.sh/uv/concepts/projects/workspaces/](https://docs.astral.sh/uv/concepts/projects/workspaces/).
- **Dependency groups** (`[dependency-groups]`): PEP 735, soportado desde uv 0.5+ — [docs.astral.sh/uv/concepts/projects/dependencies/](https://docs.astral.sh/uv/concepts/projects/dependencies/).
- **Per-package index** con `[tool.uv.sources]` + `[[tool.uv.index]] explicit = true`:
  > *"An index can be marked as `explicit = true` to ensure it's only used for packages explicitly pinned to it in `tool.uv.sources`."* — [docs.astral.sh/uv/guides/integration/pytorch/](https://docs.astral.sh/uv/guides/integration/pytorch/).
- **Conflicts entre extras**: declaración explícita de incompatibilidad:
  > *"This tells `uv` to resolve them separately, preventing both from being installed simultaneously."* — DeepWiki `astral-sh/uv`.
- **`uv pip install` con `--index-url`** (caso PyTorch):
  > *"To use the same workflow with uv, replace `pip3` with `uv pip`: `$ uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`"* — [docs.astral.sh/uv/guides/integration/pytorch/](https://docs.astral.sh/uv/guides/integration/pytorch/).

### 8.4 Aislamiento de runtime entre TF 2.15 y torch 2.1+cu121

**No hay conflicto cuando se ejecutan en kernels separados.** JupyterLab abre un proceso Python independiente por cada kernel. `libcudart`, `libcudnn`, `libnvinfer` se cargan independientemente por cada proceso. El JupyterLab server en sí no carga ninguna de estas librerías — solo gestiona los kernels vía ZeroMQ.

Esto incluye:

- `libcudart.so.10.2` (TF 2.5 path, irrelevante aquí; en Vast.ai usamos CUDA 12.4) y `libcudart.so.12.4` (4090 driver).
- `libcudnn.so.8.9` (CUDA 12.4) — single version, ambos tracks lo usan.
- `libnvinfer.so.10` (CUDA 12.4 TRT 10) — para Track B en validación local (aunque la validación canónica de D13 usa Docker NGC con TRT 8.2.1).

### 8.5 Comandos canónicos de creación de venvs

```bash
# ============== Track A — venv /opt/venv/tracka ==============
uv venv /opt/venv/tracka --python 3.10
source /opt/venv/tracka/bin/activate

# Stack core (sin --no-deps, deja que uv resuelva)
uv pip install tensorflow==2.15.0 \
              tf-models-official==2.15.0 \
              "tensorflow-model-optimization>=0.7.5,<0.8.0" \
              "numpy==1.26.4" \
              "protobuf==3.20.3" \
              "Pillow==10.4.0" \
              "opencv-python-headless==4.10.0.84" \
              "pycocotools==2.0.7" \
              "lvis==0.5.3" \
              "tensorflow-addons==0.23.0" \
              "tensorflow-text==2.15.0" \
              grpcio-tools==1.64.1 \
              huggingface_hub ipykernel

# Registrar kernel
python -m ipykernel install --user --name tracka \
  --display-name "Track A (TF 2.15)"

deactivate

# ============== Track B — venv /opt/venv/trackb ==============
uv venv /opt/venv/trackb --python 3.10
source /opt/venv/trackb/bin/activate

# Defensa contra NumPy 2.x ANTES de ultralytics (issue #22346)
uv pip install "numpy<2.0"

# PyTorch CUDA 12.1 vía index URL
uv pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# Ultralytics + ONNX stack + Roboflow
uv pip install "ultralytics>=8.4.46,<8.5" \
              "onnxslim>=0.1.82" \
              "onnx>=1.16,<1.18" \
              "onnxruntime>=1.18,<1.21" \
              "roboflow>=1.3.6,<1.4" \
              wandb huggingface_hub ipykernel

# Registrar kernel
python -m ipykernel install --user --name trackb \
  --display-name "Track B (YOLOv8)"

deactivate

# ============== Validación de kernels ==============
jupyter kernelspec list
# Esperado:
#   tracka     ~/.local/share/jupyter/kernels/tracka
#   trackb     ~/.local/share/jupyter/kernels/trackb
```

### 8.6 `kernel.json` resultante (referencia)

Tras `python -m ipykernel install --user --name trackb`, el archivo `~/.local/share/jupyter/kernels/trackb/kernel.json` contiene:

```json
{
 "argv": [
  "/opt/venv/trackb/bin/python",
  "-Xfrozen_modules=off",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Track B (YOLOv8)",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```

El campo `argv[0]` debe apuntar a **`/opt/venv/trackb/bin/python`**, no a `/usr/bin/python3`. Si apunta al sistema, el kernel cargará deps del sistema (no del venv) y romperá con `ModuleNotFoundError: ultralytics`.

### 8.7 Variante con detección automática de backend CUDA (no usada aquí)

`uv` soporta `UV_TORCH_BACKEND=cu121` para autodetección:

```bash
UV_TORCH_BACKEND=cu121 uv pip install torch==2.1.0 torchvision==0.16.0
```

No la usamos porque conocemos el backend de antemano (`cu121` para 4090 con driver CUDA 12.4 compatible).

### 8.8 Por qué Python 3.10 (no 3.11 o 3.12)

- **TF 2.15.0 cp310** existe en PyPI: `tensorflow-2.15.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` (2023-11-14).
- **TF 2.15.0 cp311 y cp312 NO existen**: TF 2.15 fue el último con wheels cp310 antes de saltar a cp312 en TF 2.16.
- **Ultralytics 8.4.46** soporta Python ≥ 3.8 hasta 3.12; 3.10 está en el sweet spot maduro.
- **ipykernel** sin issues conocidos en Py 3.10.

El container `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` ya viene con Python 3.10 base. **Coincidencia perfecta.**

---

## 9. Notebook persistente

### 9.1 La decisión renegociada (D9 reemplaza D4)

R4 había definido D4: convertir los `.ipynb` a `.py` con `jupytext --to py:percent` para ejecutar headless con `python train_track_a.py`. R5 renegocia esta decisión a petición explícita del usuario (cita verbatim del HANDOFF §2): *"Necesito negociar [...] correrlo como notebook [...] que su ejecución no se afecte porque cerré la pestaña [...]"*.

### 9.2 Comparativa de cinco estrategias evaluadas

| Estrategia | Output post-disconnect | Requiere pestaña abierta | Pros | Contras |
|------------|------------------------|---------------------------|------|---------|
| JupyterLab + kernel detached "puro" desde UI | Solo lo guardado en `.ipynb` (autosave 120 s) | No | UI familiar, interactividad | **Output entre autosaves NO recuperable** |
| `papermill input.ipynb output.ipynb` | Sí, en `output.ipynb` completo | No | Output completo garantizado | Sin UI durante ejecución; archivo separado |
| `jupyter execute notebook.ipynb` | Sí, in-place | No | Simple, sin deps extra | Menos manejo de errores que papermill |
| **`nbconvert --execute --to notebook --inplace`** | **Sí, in-place** | **No** | **Estándar, distribuido con Jupyter** | Inicialización lenta |
| `nohup jupyter execute ... &` | Sí (background) | No | Sencillo | Sin reintentos ni logs estructurados |

**Conclusión:** `nbconvert --execute --inplace` dentro de `tmux` es el **único patrón** que combina:

1. Preservación del `.ipynb` con outputs incrustados navegable desde JupyterLab al reconectar.
2. Supervivencia del kernel al cierre de pestaña (tmux mantiene el proceso vivo).
3. Recuperación completa del output al volver horas después.

`jupytext --to py:percent` no satisface (1) porque el `.py` resultante no captura outputs en el `.ipynb` original (queda solo en stdout del proceso). Si se quiere ver el output al reconectar, hay que redirigir a archivo y abrir desde otra herramienta — fricción innecesaria.

### 9.3 Limitación crítica de "kernel detached puro"

> **Al reconectar a JupyterLab tras una desconexión, el output de celdas ya ejecutadas NO se recupera automáticamente del stream en vivo. Solo el output guardado en el `.ipynb` es visible.**

El kernel Python es un proceso independiente del WebSocket. Al cerrar el navegador, el proceso continúa; lo que se pierde es solo el stream IOPub. Sin `nbconvert`, los outputs emitidos entre el último autosave (cada 120 s) y la desconexión se pierden completamente.

Por eso `tmux + nbconvert --execute --inplace` es obligatorio: `nbconvert` escribe directamente al `.ipynb` en disco, no depende de IOPub stream.

### 9.4 Config crítica de JupyterLab (`/root/.jupyter/jupyter_server_config.py`)

```python
# /root/.jupyter/jupyter_server_config.py
# Aplica al lanzar JupyterLab desde Vast.ai (D9 — kernel-survives-disconnect)

# 1. Kernel sobrevive cierre de pestaña / disconnect
c.MappingKernelManager.cull_idle_timeout = 0       # 0 deshabilita el culler completamente
c.MappingKernelManager.cull_busy = False           # no cull si está busy
c.MappingKernelManager.cull_connected = False      # no cull si hay clients conectados
c.ServerApp.shutdown_no_activity_timeout = 0       # server no se apaga por inactividad

# 2. Sin truncado de output verboso (training loops loggean cada batch)
c.ZMQChannelsWebsocketConnection.iopub_msg_rate_limit = 0          # default 1000 msg/s
c.ZMQChannelsWebsocketConnection.iopub_data_rate_limit = 10_000_000  # 10 MB/s (default 1 MB/s)
```

**Justificación:**

- *"To disable kernel culling entirely, set `cull_idle_timeout` to 0 or lower. This ensures kernels survive browser disconnects."* — DeepWiki sobre `jupyter-server/jupyter_server`.
- *"Kernel culling is initialized lazily when the first kernel starts, so setting `cull_idle_timeout=0` prevents the culler from ever starting."* — Idem.
- *"These [rate limit settings] were deprecated in `ServerApp` in favor of configuring them directly on `ZMQChannelsWebsocketConnection`."* — Idem.

Sin estos ajustes, un training que loggea métricas cada batch genera `[stdout truncated]` en la celda y el kernel puede ser asesinado por el culler aunque el proceso siga vivo.

### 9.5 Comando canónico de ejecución (Track B ejemplo)

```bash
# Dentro del terminal de JupyterLab (o vía SSH)

# 1. Crear sesión tmux detached
tmux new-session -d -s training

# 2. Enviar comando de ejecución
tmux send-keys -t training '
  source /opt/venv/trackb/bin/activate
  cd /workspace/embebidos-3
  jupyter nbconvert --to notebook --execute --inplace \
    notebooks/train_track_b_yolov8.ipynb \
    --ExecutePreprocessor.timeout=10800 \
    --ExecutePreprocessor.kernel_name=trackb \
    2>&1 | tee /workspace/embebidos-3/logs/train_track_b.log
  touch /workspace/embebidos-3/.training_done
' Enter

# 3. Verificar que el proceso sigue vivo
tmux ls
# Output esperado: training: 1 windows (created ...) [80x24]

# 4. (Opcional) inspeccionar progreso
tmux attach-session -t training
# Ctrl+B, D para detach sin matar el proceso
```

### 9.6 Flags importantes de `nbconvert`

| Flag | Valor | Razón |
|------|-------|-------|
| `--to notebook` | (fijo) | Mantener formato `.ipynb` |
| `--execute` | (fijo) | Ejecutar todas las celdas |
| `--inplace` | (fijo) | Escribir outputs al mismo archivo `.ipynb` |
| `--ExecutePreprocessor.timeout` | `10800` | 3 h por celda. Default 30 s (rompe para training). |
| `--ExecutePreprocessor.kernel_name` | `tracka` o `trackb` | Forzar kernel custom (sino usa kernel por default, falla con TF/torch específico) |
| `--allow-errors` | (NO usar) | Default `false` aborta al primer error: queremos eso |

**Alternativa con `papermill`:** sintaxis similar pero con archivo separado y parametrización inyectable. `papermill input.ipynb output.ipynb -p PARAM_NAME PARAM_VALUE`. Útil si se quiere ejecutar múltiples variantes. Para nuestro caso (un run por track) `nbconvert` es suficiente y no añade deps extra.

### 9.7 Trampas conocidas y mitigaciones

| # | Trampa | Mitigación |
|---|--------|-----------|
| T1 | `"Notebook is too large to be saved"` cuando el output acumulado supera ~25 MB | `%%capture` en celdas de instalación; `IPython.display.clear_output(wait=True)` antes de loops verbosos; log verboso a archivo con `tqdm.write()` en vez de `print()`; mostrar solo métricas resumidas con `tqdm` |
| T2 | Autosave conflictivo si se abre el mismo `.ipynb` en dos pestañas | Una sola sesión activa por notebook (en general no se abrirá mientras está corriendo `nbconvert`) |
| T3 | WebSocket ping/pong timeout: el proxy de Vast.ai puede cerrar conexiones inactivas | `jupyter lab --ServerApp.tornado_settings='{"websocket_ping_interval": 30000}'` o `--ping-interval 30` al lanzar |
| T4 | Buffer overflow del frontend con `iopub_data_rate_limit=0` | El navegador se ralentiza con miles de líneas. Log verboso a archivo; mostrar pocos prints |
| T5 | `nbconvert` falla silenciosamente si kernel no existe | Pre-validar con `jupyter kernelspec list` antes; especificar `--ExecutePreprocessor.kernel_name=<tracka\|trackb>` explícito (issue R9 del HANDOFF) |
| T6 | `tmux` no preinstalado en imagen vastai | `apt-get install -y tmux` en bootstrap |
| T7 | `nbconvert` no respeta `cull_idle_timeout=0` si la config no se aplicó antes de lanzar JupyterLab | Reiniciar JupyterLab tras escribir `jupyter_server_config.py`: `supervisorctl restart jupyter` |

---

## 10. HF Hub persistence

### 10.1 Estructura del repo `mitgar14/embebidos-3-models` (D3)

Repo **privado** ya creado en HF Hub con la siguiente estructura:

```
mitgar14/embebidos-3-models/
├── README.md                  # generado en setup
├── track_a/
│   ├── runs/                  # logs intermedios y tfevents
│   │   └── .gitkeep
│   ├── checkpoints/           # ckpt-N (TF OD API)
│   │   └── .gitkeep
│   ├── exports/               # detect_int8.tflite, manifest.json, pipeline.config
│   │   └── .gitkeep
│   └── logs/                  # bootstrap.log, train_track_a.log
│       └── .gitkeep
└── track_b/
    ├── runs/                  # logs W&B mirror si necesario
    │   └── .gitkeep
    ├── checkpoints/           # best.pt, last.pt
    │   └── .gitkeep
    ├── exports/               # best.onnx, manifest.json
    │   └── .gitkeep
    └── logs/                  # bootstrap.log, train_track_b.log
        └── .gitkeep
```

**Free tier 2026:** 100 GB privado total, sin límite de repos privados, 500 GB max por archivo, sin cap de bandwidth. Xet storage por default desde mayo 2025 (dedup chunks). Artefactos totales del proyecto ~250–450 MB.

### 10.2 `CommitScheduler` cada 5 min (D5)

`CommitScheduler` es el patrón canónico para push automático periódico sin bloquear el kernel:

```python
from huggingface_hub import CommitScheduler
from pathlib import Path

# Track A
checkpoints_dir = Path("/workspace/embebidos-3/track_a/checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

scheduler_a = CommitScheduler(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path=str(checkpoints_dir),
    path_in_repo="track_a/checkpoints",
    every=5,                # minutos
    private=True,
    squash_history=False,   # mantener historial para rollback
    token=None,             # usa ~/.cache/huggingface/token
)

# Lo mismo para logs/tfevents
scheduler_a_logs = CommitScheduler(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_a/logs",
    path_in_repo="track_a/logs",
    every=5,
    private=True,
)
```

**Cómo funciona internamente:**

- Inicia un thread daemon que cada `every` minutos hace `snapshot` de `folder_path`.
- Calcula diff contra el último commit en `path_in_repo` del repo.
- Si hay cambios, hace `commit` con mensaje autogenerado (`Update from CommitScheduler [...]`).
- Si no hay cambios, no hace commit (no contamina historial).
- Resistente a errores de red: reintenta en el próximo intervalo.

### 10.3 `upload_folder(run_as_future=True)` para checkpoints finales

Al terminar el training, push asíncrono de exports + manifests:

```python
from huggingface_hub import HfApi
api = HfApi()

# Push asíncrono — no bloquea
future = api.upload_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_a/exports",
    path_in_repo="track_a/exports",
    commit_message="Track A — final exports (model.tflite + manifest)",
    run_as_future=True,
)

# ... otras tareas ...

# Bloquear solo cuando se necesite
result = future.result(timeout=600)  # 10 min max
print(f"Upload completo: {result}")
```

**`run_as_future=True`:** devuelve un `concurrent.futures.Future`. Permite ejecutar otras tareas en paralelo mientras el upload corre en background.

### 10.4 `upload_large_folder` para casos resumibles

Patrón resumible para artefactos grandes (no aplica a embebidos-3 con ~50 MB total por track, pero documentado por completitud):

```python
api.upload_large_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_b/exports",
    path_in_repo="track_b/exports",
    private=True,
    multi_commits=True,                # chunks separados
    multi_commits_verbose=True,
    create_pr=False,
)
```

Reanuda automáticamente si la red se cae a mitad de la subida.

### 10.5 TensorBoard hosted en HF Hub (D6 Track A)

HF Hub detecta automáticamente archivos `tfevents.*` y monta una instancia de TensorBoard gratis. Patrón:

```python
from huggingface_hub import HFSummaryWriter

writer = HFSummaryWriter(
    repo_id="mitgar14/embebidos-3-models",
    logdir="/workspace/embebidos-3/track_a/logs",
    commit_every=5,         # minutos
    repo_private=True,
)

# Uso idéntico a torch.utils.tensorboard.SummaryWriter
writer.add_scalar("train/loss", loss, step)
writer.add_scalar("train/mAP", map_value, step)
writer.add_image("samples/predictions", img, step)
```

URL de TensorBoard hosted: `https://huggingface.co/mitgar14/embebidos-3-models/tensorboard`. La detección y boot del board son automáticos al primer commit con `tfevents`.

### 10.6 W&B nativo (D6 Track B)

Ultralytics tiene integración nativa con Weights & Biases:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=416,
    batch=32,
    project="embebidos-3",
    name="track_b_yolov8n",
    wandb=True,             # ← magia: crea W&B run, loggea métricas, sube samples
)
```

**Variables requeridas:** `WANDB_API_KEY` (pasada al container vía `--env`).

**Free tier W&B 2026:** sin cap conocido en runs para proyectos personales. Confirmado en [`wandb.ai/site/pricing`](https://wandb.ai/site/pricing).

Crea automáticamente:

- **Run** en proyecto W&B `embebidos-3` con name `track_b_yolov8n`.
- **Métricas:** `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/precision`, `val/recall`, `val/mAP50`, `val/mAP50-95`.
- **Media:** confusion matrix, sample predictions (cada N epochs), labels.jpg, results.png.
- **System metrics:** GPU util, GPU memory, CPU util, network I/O.

URL típica: `https://wandb.ai/<user>/embebidos-3/runs/<run_id>`.

### 10.7 Heartbeat custom (red de seguridad)

Heartbeat custom de los notebooks actuales se conserva (logs cada N segundos a stdout) como red de seguridad si W&B o TensorBoard fallan. Variables de entorno requeridas: `WANDB_API_KEY` (Track B), `HF_TOKEN` (ambos).

### 10.8 Snippet completo Track A (Bootstrap CommitScheduler + HFSummaryWriter)

```python
import os
from pathlib import Path
from huggingface_hub import CommitScheduler, HFSummaryWriter

REPO_ID = "mitgar14/embebidos-3-models"
ROOT = Path("/workspace/embebidos-3")

# Asegurar directorios
for sub in ["track_a/checkpoints", "track_a/exports", "track_a/logs"]:
    (ROOT / sub).mkdir(parents=True, exist_ok=True)

# Scheduler para checkpoints (cada 5 min)
ckpt_scheduler = CommitScheduler(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=str(ROOT / "track_a/checkpoints"),
    path_in_repo="track_a/checkpoints",
    every=5,
    private=True,
)

# Scheduler para logs / tfevents (cada 5 min)
logs_scheduler = CommitScheduler(
    repo_id=REPO_ID,
    repo_type="model",
    folder_path=str(ROOT / "track_a/logs"),
    path_in_repo="track_a/logs",
    every=5,
    private=True,
)

# TensorBoard writer
tb_writer = HFSummaryWriter(
    repo_id=REPO_ID,
    logdir=str(ROOT / "track_a/logs"),
    commit_every=5,
    repo_private=True,
)

print(f"✅ HF Hub persistence activa para {REPO_ID}")
print(f"   Checkpoints: cada 5 min desde {ROOT / 'track_a/checkpoints'}")
print(f"   Logs/TB:     cada 5 min desde {ROOT / 'track_a/logs'}")
```

---

## 11. Auto-destroy

### 11.1 GAPs confirmados de Vast.ai (motivación de D11)

| GAP | Evidencia |
|-----|-----------|
| **`vastai create instance` NO tiene `--auto-stop`, `--max-runtime`, `--idle-timeout`** | Revisión de [`vast-ai/vast-python`](https://github.com/vast-ai/vast-python) source `vast.py` y de [docs.vast.ai/cli/reference/create-instance](https://docs.vast.ai/cli/reference/create-instance) |
| **Vast.ai NO tiene "Idle Shutdown" automático** basado en utilización de GPU | La doc no menciona ningún mecanismo de shutdown por inactividad. Vast.ai cobra por segundo de instancia activa pero no detiene automáticamente |
| **Flags `--end_date`, `--day`, `--hour` pertenecen a `add_scheduled_job`** (jobs programados), no a `create instance` | Verificación de subcomandos |

Quote de la doc oficial: *"Every offer has a maximum rental duration. When you rent an instance, the offer end date at the time of rental becomes your rental end date, the date your instance will run until."* — [docs.vast.ai/guides/reference/faq/instances](https://docs.vast.ai/guides/reference/faq/instances).

### 11.2 Patrón canónico de auto-destroy desde el container

La FAQ oficial documenta el patrón:

> *"A special instance API key is pre-installed. Install the CLI and use it: `pip install vastai` / `vastai stop instance $CONTAINER_ID`"*
> — [docs.vast.ai/guides/reference/faq/instances](https://docs.vast.ai/guides/reference/faq/instances)

El `$CONTAINER_ID` está disponible como env var `VAST_CONTAINERLABEL`, formato `C.<id>` (extraer con `${VAST_CONTAINERLABEL#C.}`).

### 11.3 Por qué NO `trap EXIT` (D7 reemplazada)

`trap EXIT` requiere un proceso bash longevo. Al ejecutar el notebook como kernel detached con `nbconvert --execute --inplace` dentro de `tmux`, el bash inicial (entrypoint del container) muere temprano y el `trap` se dispara antes de que termine el training. El proceso del kernel sobrevive a `trap`, pero el shutdown nunca se dispara.

### 11.4 Patrón de tres componentes (D11)

#### 11.4.1 Componente 1 — Cron watchdog instalado en bootstrap

```bash
# Registrar cron job que cada minuto verifica el archivo señal
echo "* * * * * test -f /workspace/embebidos-3/.training_done && \
  vastai destroy instance \${VAST_CONTAINERLABEL#C.} 2>&1 | \
  tee -a /workspace/embebidos-3/logs/watchdog.log" | crontab -

# Verificar
crontab -l
```

**Frecuencia 1 min** es suficiente para nuestro caso (latencia tope 60 s). Para latencia menor, podría usar systemd timer con `OnUnitActiveSec=10s`, pero overhead innecesario.

#### 11.4.2 Componente 2 — Última celda del notebook

```python
# Última celda — auto-destroy señaling + plan B inmediato
import os
import subprocess
import time
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()

# 1. Upload artefactos finales (espera a que termine, evitar pérdida)
print("[finalize] Subiendo exports a HF Hub...")
api.upload_folder(
    repo_id="mitgar14/embebidos-3-models",
    repo_type="model",
    folder_path="/workspace/embebidos-3/track_b/exports",
    path_in_repo="track_b/exports",
    commit_message="Track B — final exports",
    run_as_future=False,  # bloqueante: necesitamos confirmación antes de destroy
)
print("[finalize] Upload completado.")

# 2. Marcar como completo (cron watchdog detecta en < 60 s)
Path("/workspace/embebidos-3/.training_done").touch()
print(f"[finalize] {time.strftime('%H:%M:%S')} Training done. Watchdog disparará destroy.")

# 3. Plan B inmediato (si watchdog falla): destruir desde el notebook
container_id = os.environ.get("VAST_CONTAINERLABEL", "").lstrip("C.")
if container_id:
    print(f"[finalize] Plan B: vastai destroy instance {container_id}")
    subprocess.run(["vastai", "destroy", "instance", container_id], check=False)
else:
    print("[finalize] VAST_CONTAINERLABEL no presente; depende del watchdog")
```

#### 11.4.3 Componente 3 — Variable `$VAST_CONTAINERLABEL`

Disponible automáticamente como env var del container (inyectada por Vast.ai al crear instancia). Formato:

```
VAST_CONTAINERLABEL=C.12345678
```

Extracción: `${VAST_CONTAINERLABEL#C.}` → `12345678`.

### 11.5 API key dentro del container

Precedencia en `vast.py` (verificada en código fuente):

```
--api-key <KEY>  >  $VAST_API_KEY (env var)  >  ~/.config/vastai/vast_api_key (archivo)
```

**Inyección al crear instancia (Opción A — env var):**

```bash
vastai create instance <OFFER_ID> \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --env '-e VAST_API_KEY=<key> -e HF_TOKEN=<token> -e WANDB_API_KEY=<key> -e ROBOFLOW_API_KEY=<key> -p 8080:8080' \
  --jupyter --jupyter-lab \
  --jupyter-dir /workspace \
  --disk 30 \
  --onstart-cmd 'bash /workspace/embebidos-3/scripts/bootstrap.sh'
```

**Seguridad:** la API key queda visible en logs del container y en variables de entorno del usuario root. Para minimizar exposición, **Opción B — archivo:**

```bash
# En onstart, antes de cualquier vastai destroy:
pip install vastai
vastai set api-key "$VAST_API_KEY"
unset VAST_API_KEY
# Ahora el key vive en ~/.config/vastai/vast_api_key con permisos 600
```

### 11.6 HF Hub webhook → AWS Lambda → vastai destroy (descartado)

**No vale la pena para proyecto académico.** La cadena HF webhook → AWS Lambda/GitHub Action → `vastai destroy` añade tres puntos de fallo:

1. HF Hub webhook puede fallar / atrasarse.
2. Lambda cold start ~3 s.
3. Permisos cross-account.

Para ahorrar US$0,05 en un saldo de US$1,72 no compensa. El watchdog interno es suficiente.

---

## 12. Bootstrap.sh completo

Archivo final: `scripts/bootstrap.sh`. **Crítico:** debe pushearse con line endings `LF`, no `CRLF` (ver gotcha #28).

```bash
#!/usr/bin/env bash
set -euo pipefail

# scripts/bootstrap.sh — embebidos-3 Vast.ai entrypoint
# Invocado por --onstart-cmd al crear instancia Vast.ai.

LOG=/workspace/embebidos-3/logs/bootstrap.log
mkdir -p /workspace/embebidos-3/logs
exec > >(tee -a "$LOG") 2>&1

echo "==> [$(date -Iseconds)] Bootstrap iniciado"

# ============ 1. Variables requeridas ============
: "${HF_TOKEN:?Variable HF_TOKEN requerida (pasar con --env -e HF_TOKEN=...)}"
: "${WANDB_API_KEY:?Variable WANDB_API_KEY requerida}"
: "${VAST_API_KEY:?Variable VAST_API_KEY requerida}"
: "${ROBOFLOW_API_KEY:?Variable ROBOFLOW_API_KEY requerida}"

# ============ 2. Sistema y utilidades ============
apt-get update -qq
apt-get install -y -qq tmux cron curl git ca-certificates

# ============ 3. Clonar repo (si no está montado) ============
if [ ! -d /workspace/embebidos-3/.git ]; then
  cd /workspace
  # Token GitHub (PAT con scope repo) inyectado para repo privado
  git clone "https://${GITHUB_TOKEN:-}@github.com/mitgar14/embebidos-3.git" embebidos-3 || \
    git clone https://github.com/mitgar14/embebidos-3.git embebidos-3
fi
cd /workspace/embebidos-3

# ============ 4. Instalar uv ============
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# ============ 5. venv Track A ============
echo "==> Creando /opt/venv/tracka..."
uv venv /opt/venv/tracka --python 3.10
# shellcheck disable=SC1091
source /opt/venv/tracka/bin/activate
uv pip install --quiet \
  tensorflow==2.15.0 \
  tf-models-official==2.15.0 \
  "tensorflow-model-optimization>=0.7.5,<0.8.0" \
  "numpy==1.26.4" \
  "protobuf==3.20.3" \
  "Pillow==10.4.0" \
  "opencv-python-headless==4.10.0.84" \
  "pycocotools==2.0.7" \
  "lvis==0.5.3" \
  "tensorflow-addons==0.23.0" \
  "tensorflow-text==2.15.0" \
  grpcio-tools==1.64.1 \
  huggingface_hub ipykernel

python -m ipykernel install --user --name tracka --display-name "Track A (TF 2.15)"

# Clonar TF Models pinned SHA
if [ ! -d /workspace/tf_models ]; then
  git clone --filter=blob:none --no-checkout \
    https://github.com/tensorflow/models.git /workspace/tf_models
  cd /workspace/tf_models
  git checkout 9cafa3d150
  test -d research/object_detection || { echo "ERROR: research/ no existe"; exit 1; }
fi

# Compilar protos (grpcio-tools)
cd /workspace/tf_models/research
python -m grpc_tools.protoc \
  --python_out=. \
  --proto_path=. \
  object_detection/protos/*.proto

# Instalar OD API sin deps
cp object_detection/packages/tf2/setup.py .
pip install --quiet --no-deps -e .

# Re-pin defensivo
pip install --quiet --force-reinstall --no-deps "Pillow==10.4.0" "protobuf==3.20.3"

deactivate
cd /workspace/embebidos-3

# ============ 6. venv Track B ============
echo "==> Creando /opt/venv/trackb..."
uv venv /opt/venv/trackb --python 3.10
# shellcheck disable=SC1091
source /opt/venv/trackb/bin/activate

# Defensa NumPy ANTES de ultralytics
uv pip install --quiet "numpy<2.0"

# PyTorch CUDA 12.1
uv pip install --quiet torch==2.1.0+cu121 torchvision==0.16.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

uv pip install --quiet \
  "ultralytics>=8.4.46,<8.5" \
  "onnxslim>=0.1.82" \
  "onnx>=1.16,<1.18" \
  "onnxruntime>=1.18,<1.21" \
  "roboflow>=1.3.6,<1.4" \
  wandb huggingface_hub ipykernel

python -m ipykernel install --user --name trackb --display-name "Track B (YOLOv8)"
deactivate

# ============ 7. Vast.ai CLI + cron watchdog ============
pip install --quiet vastai

# Guardar API key en archivo (más seguro que env var persistente)
vastai set api-key "$VAST_API_KEY"

# Registrar cron watchdog (cada 1 min)
service cron start || cron
echo "* * * * * test -f /workspace/embebidos-3/.training_done && \
  vastai destroy instance \${VAST_CONTAINERLABEL#C.} 2>&1 | \
  tee -a /workspace/embebidos-3/logs/watchdog.log" | crontab -

crontab -l

# ============ 8. Config JupyterLab persistente ============
mkdir -p /root/.jupyter
cat > /root/.jupyter/jupyter_server_config.py <<'PYEOF'
c.MappingKernelManager.cull_idle_timeout = 0
c.MappingKernelManager.cull_busy = False
c.MappingKernelManager.cull_connected = False
c.ServerApp.shutdown_no_activity_timeout = 0
c.ZMQChannelsWebsocketConnection.iopub_msg_rate_limit = 0
c.ZMQChannelsWebsocketConnection.iopub_data_rate_limit = 10_000_000
PYEOF

# ============ 9. Variables env permanentes (para sesiones SSH/tmux) ============
{
  echo "export HF_TOKEN='${HF_TOKEN}'"
  echo "export WANDB_API_KEY='${WANDB_API_KEY}'"
  echo "export ROBOFLOW_API_KEY='${ROBOFLOW_API_KEY}'"
  # NO exportar VAST_API_KEY (ya está en ~/.config/vastai/vast_api_key)
} >> /root/.bashrc

# Autenticar HF CLI (escribe ~/.cache/huggingface/token)
hf auth login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# ============ 10. Reiniciar JupyterLab ============
if command -v supervisorctl >/dev/null 2>&1; then
  supervisorctl restart jupyter || echo "==> JupyterLab no controlado por supervisor; reinicia desde la UI"
else
  echo "==> supervisorctl no disponible; reinicia JupyterLab desde la UI de Vast.ai para aplicar config"
fi

echo "==> [$(date -Iseconds)] Bootstrap completo."
echo "    Kernels registrados: tracka (TF 2.15), trackb (YOLOv8)."
echo "    HF Hub repo: mitgar14/embebidos-3-models."
echo "    Watchdog cron activo (frecuencia 1 min)."
echo "    Próximo paso: tmux new -s training; jupyter nbconvert --execute --inplace ..."
```

### 12.1 Flujo de uso end-to-end

1. **Local:** `git push` con el notebook actualizado al repo `mitgar14/embebidos-3`.
2. **CLI Vast.ai:** (ver §13)
3. **Browser:** abrir la URL de JupyterLab que entrega Vast.ai. Confirmar que en `Kernel > Change Kernel` aparecen `Track A (TF 2.15)` y `Track B (YOLOv8)`.
4. **Terminal en JupyterLab:**
   ```bash
   tmux new -s training
   jupyter nbconvert --to notebook --execute --inplace \
     notebooks/train_track_b_yolov8.ipynb \
     --ExecutePreprocessor.timeout=10800 \
     --ExecutePreprocessor.kernel_name=trackb
   # Ctrl+B, D para detach
   ```
5. **Usuario cierra pestaña.** El proceso `nbconvert` sigue ejecutando en `tmux`. El kernel `trackb` sobrevive porque `cull_idle_timeout=0`.
6. **2 horas después:** usuario reabre la URL de JupyterLab. Abre `train_track_b_yolov8.ipynb`. Output completo está guardado en disco.
7. **Última celda** crea `/workspace/embebidos-3/.training_done`. Cron watchdog detecta en < 60 s. `vastai destroy` dispara. Instancia desaparece.

---

## 13. Comandos `vastai create instance`

### 13.1 Flags relevantes

| Flag CLI | Campo API | Descripción |
|----------|-----------|-------------|
| `--image <tag>` | `image_uuid` | Tag Docker (`vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`) |
| `--jupyter` | `runtype: jupyter` | Activa launch mode jupyter (abre puerto 8080 + 22 SSH) |
| `--jupyter-lab` | `use_jupyter_lab: true` | Lanza JupyterLab en vez de Notebook clásico |
| `--jupyter-dir <path>` | `jupyter_dir: <path>` | Directorio raíz del server (default `/workspace`) |
| `--direct` | — | Conexión HTTPS directa (no proxy). Requiere instalar certificado TLS local |
| `--env '<flags>'` | `env: {...}` | Variables de entorno + port forwarding (e.g., `-e KEY=val -p PORT:PORT`) |
| `--disk <GB>` | `disk` | Tamaño del volumen (default 10 GB; usamos 30 para datasets) |
| `--onstart-cmd <cmd>` | `onstart` | Comando a ejecutar al arrancar el container |
| `--ssh` | `run_with_ssh: true` | Habilita SSH (cuidado con quirk `d5f717eb` si imagen no incluye `sshd`) |

### 13.2 Crear instancia (template completo)

```bash
# Buscar offer barato con RTX 4090 (CLI Vast.ai)
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 dph_total<=0.50 reliability>0.95 inet_down>=200' \
  --order 'dph_total'

# Crear instancia (reemplazar <OFFER_ID> con el del search)
vastai create instance <OFFER_ID> \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --env '-e VAST_API_KEY=<VAST_KEY> -e HF_TOKEN=<HF_KEY> -e WANDB_API_KEY=<WANDB_KEY> -e ROBOFLOW_API_KEY=<RF_KEY> -e GITHUB_TOKEN=<GH_PAT> -p 8080:8080' \
  --jupyter --jupyter-lab \
  --jupyter-dir /workspace \
  --disk 30 \
  --onstart-cmd 'bash -c "curl -sSL https://raw.githubusercontent.com/mitgar14/embebidos-3/main/scripts/bootstrap.sh | bash"'

# Ver instancias activas
vastai show instances

# Conectar (printa URL de JupyterLab)
vastai show instance <ID>

# Destruir manualmente (si watchdog falla)
vastai destroy instance <ID>
```

### 13.3 GitHub token para clone de repo privado

Para que `git clone https://github.com/mitgar14/embebidos-3.git` funcione desde el container:

1. Generar **fine-grained PAT** en GitHub con scope `repo:read` (solo lectura).
2. Inyectar como `GITHUB_TOKEN` vía `--env`.
3. Bootstrap usa `https://${GITHUB_TOKEN}@github.com/mitgar14/embebidos-3.git`.

**Alternativa:** deploy key SSH del repo, pero requiere config `~/.ssh/known_hosts` y agente en el container. PAT es más simple para uso único.

### 13.4 Configuración local previa (Windows)

#### 13.4.1 Generar PAT GitHub (una sola vez)

1. https://github.com/settings/tokens?type=beta
2. Generate new token → scope `Contents: read` para `mitgar14/embebidos-3`.
3. Guardar como `GITHUB_TOKEN` en `.env` local (gitignored).

#### 13.4.2 Token HF Hub

Ya autenticado: `hf auth whoami` → `user: mitgar14` (verificado 2026-05-12).
Token en `~/.cache/huggingface/token` localmente. Para Vast.ai, leer y exportar:

```powershell
# PowerShell Windows
$HF_TOKEN = Get-Content "$env:USERPROFILE\.cache\huggingface\token"
# Inyectar en --env -e HF_TOKEN=$HF_TOKEN al crear instancia
```

#### 13.4.3 `.env` local recomendado (gitignored)

```bash
# .env (NO commitear)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ROBOFLOW_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx
VAST_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GITHUB_TOKEN=github_pat_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Verificar `.gitignore` incluye `.env*` (HANDOFF §4 confirma que sí).

#### 13.4.4 `.gitattributes` para line endings (obligatorio)

Para evitar el bug CRLF documentado (mnemon `27b66a6b`):

```gitattributes
# .gitattributes
*.sh   text eol=lf
*.py   text eol=lf
*.ipynb text eol=lf
*.json text eol=lf
*.yaml text eol=lf
*.yml  text eol=lf
*.md   text eol=lf
```

Validar tras commit con `git check-attr -a scripts/bootstrap.sh` (esperar `text: set`, `eol: lf`).

#### 13.4.5 Vast.ai CLI local

```powershell
# PowerShell Windows
pip install vastai
vastai set api-key "$env:VAST_API_KEY"
vastai search offers 'gpu_name=RTX_4090' --order 'dph_total' | Select-Object -First 10
```

Path del binario en Windows (memoria mnemon `b13050ac`): `C:\Users\mitgar14\AppData\Roaming\Python\Python312\Scripts\vastai.exe`.

---

## 14. Mecánica del versioning TFLite

### 14.1 Por qué importa `op_version`

Cada operador TFLite tiene una versión máxima registrada en el runtime. Si el converter (TF 2.15 export) genera un `op_version` superior al runtime (TFLite 2.5 del Nano), el modelo lanza al cargar:

```
ERROR: Didn't find op for builtin opcode 'CONV_2D' version '5'.
An older version of this builtin might be supported.
Are you using old TFLite binary with newer model?
```

El **schema flatbuffer** (`TFLITE_SCHEMA_VERSION = 3`) es estable entre TF 2.5 y TF 2.21 (verificado en `tensorflow/lite/version.h` HEAD = TF 2.21 master). La compatibilidad real depende de **op versioning por operador**, no del schema container.

### 14.2 Histórico de issues confirmados

Tres reports cruzados confirman que el escenario es real:

- Issue [`tensorflow/tensorflow#41943`](https://github.com/tensorflow/tensorflow/issues/41943) (`mgalgs`, 2020): *"ERROR: Didn't find op for builtin opcode 'CONV_2D' version '5'."* Modelo generado con `tf-nightly 2.4`, fallo en runtime 2.2/2.3.
- Issue [`#50652`](https://github.com/tensorflow/tensorflow/issues/50652) (`djbacad`, TF 2.5.0 Python 3.6.9, 2021): *"Quantized Version of tf-lite model returning `ERROR: Didn't find op for builtin opcode 'CONV_2D' version '5'`."*
- Issue [`#43232`](https://github.com/tensorflow/tensorflow/issues/43232) (`juanpbotero98`, 2020): *"I'm trying to run inference of a custom trained `mobilenet_v2_coco17_320x320_tpu-8` model on a Raspberry pi [...] `Didn't find op for builtin opcode 'CONV_2D' version '5'`."*

El patrón es: **converter moderno genera op version N**, **runtime viejo soporta hasta versión N-1**. La mitigación es forzar el converter legacy (D12 flags) o downgrade manual del flatbuffer.

### 14.3 Tabla de `op_version` por riesgo (TF 2.15 INT8 PTQ → runtime TFLite 2.5)

| Op | Versión máx TFLite 2.5 | Versión generada TF 2.15 INT8 PTQ | Riesgo | Mitigación si falla |
|----|------------------------|------------------------------------|--------|----------------------|
| `CONV_2D` | 5 | 5 con activaciones estándar | **Bajo** (al límite) | Converter legacy (D12) o `flatbuffer_utils` downgrade |
| `DEPTHWISE_CONV_2D` | 4 | 4–5 según flags | **Medio** (verificación obligatoria) | Converter legacy (D12) |
| `FULLY_CONNECTED` | 4 | 4 para MobileNet v2 | Bajo | — |
| `QUANTIZE` / `DEQUANTIZE` | 2 | 2 | Bajo | — |
| `PAD`, `ADD`, `MUL`, `RESHAPE`, `CONCATENATION` | 1–2 | 1–2 | Ninguno | — |
| `TFLite_Detection_PostProcess` | custom | custom | Depende del wheel runtime | Ver §17 (Plan B Coral wheel) |
| `Cast v2+` | (requiere TF 2.7+) | Probablemente v1 si NMS embebido evita Cast | Bajo | — |
| `BatchMatMul v5+` | (requiere TF 2.6+) | No aplica en MV2 SSD plain | Ninguno | — |

**GAP residual:** las versiones exactas de `DEPTHWISE_CONV_2D` y `FULLY_CONNECTED` que TF 2.15 genera con INT8 PTQ desde TFOD API **no están documentadas públicamente**. La inspección flatbuffer post-conversión (§15) es obligatoria.

### 14.4 Por qué no existe flag `target_runtime_version`

El converter TFLite no tiene flag `--target_runtime_version` ni `min_runtime_version`. Los únicos flags que reducen la probabilidad de versiones altas son los del converter legacy:

```python
converter.experimental_new_quantizer = False  # cuantizador legacy
converter.experimental_new_converter  = False # converter TOCO legacy
```

**Tradeoff:** el converter legacy puede producir cuantización de menor calidad (entre +1 y +3 pp de drop adicional según Karimov et al. 2025), pero garantiza compatibilidad con runtimes antiguos.

---

## 15. Gate 1 — TFLite `op_version`

### 15.1 Instalación del paquete `tflite==2.5.0`

```bash
source /opt/venv/tracka/bin/activate
pip install tflite==2.5.0  # PyPI: package "tflite" (no "tflite-runtime")
```

### 15.2 Script de inspección completo

```python
# scripts/validate_tflite_ops.py (o celda del notebook)
import tflite.Model
from pathlib import Path

MODEL = Path("track_a/exports/model_int8.tflite")

with open(MODEL, "rb") as f:
    buf = bytearray(f.read())

model = tflite.Model.Model.GetRootAsModel(buf, 0)
sg = model.Subgraphs(0)
op_codes = [model.OperatorCodes(i) for i in range(model.OperatorCodesLength())]

# Tabla de versiones máximas soportadas por runtime 2.5
MAX_VERSION_TFLITE_25 = {
    1: 5,   # CONV_2D
    4: 4,   # DEPTHWISE_CONV_2D
    9: 4,   # FULLY_CONNECTED
    # 114, 115 = QUANTIZE/DEQUANTIZE v2 OK
    # 1, 2, 3, 22, 41 = PAD/ADD/MUL/RESHAPE/CONCATENATION v1-2 OK
}

errors = []
for i in range(sg.OperatorsLength()):
    op = sg.Operators(i)
    code = op_codes[op.OpcodeIndex()]
    builtin = code.BuiltinCode()
    version = code.Version()
    max_v = MAX_VERSION_TFLITE_25.get(builtin)
    if max_v is not None and version > max_v:
        errors.append(f"Op {i}: builtin={builtin} version={version} > max {max_v}")
    print(f"Op {i}: builtin={builtin} version={version}")

if errors:
    print("\n❌ ERRORES op_version > runtime 2.5 max:")
    for e in errors:
        print(f"   {e}")
    print("\nAplicar flags D12 o flatbuffer_utils downgrade (§18).")
    raise SystemExit(1)
else:
    print(f"\n✅ Todos los operadores son compatibles con TFLite runtime 2.5")
```

### 15.3 Flags conservadores del converter (export Track A)

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

# Conversión INT8 con representative dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.uint8

# CRÍTICO (D12): flags conservadores para minimizar op_version
converter.experimental_new_quantizer = False
converter.experimental_new_converter  = False

tflite_model = converter.convert()
Path("track_a/exports/model_int8.tflite").write_bytes(tflite_model)
```

---

## 16. Gate 2 — TFLite carga test

### 16.1 Wheel Coral CP38 x86 (idéntico runtime al Nano)

El wheel Coral CP38 x86 es **el mismo runtime TFLite 2.5** que el Nano usaría (wheel Coral CP36 aarch64), pero compilado para Vast.ai x86. Permite validar sin tener acceso físico al Nano.

```bash
# Wheel verificado disponible en Google Coral PyPI repo
pip install \
  "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl"
```

### 16.2 Script de validación

```python
import tflite_runtime.interpreter as tflite

try:
    interp = tflite.Interpreter("track_a/exports/model_int8.tflite")
    interp.allocate_tensors()

    in_d = interp.get_input_details()
    out_d = interp.get_output_details()
    print(f"✅ Runtime 2.5 acepta el modelo")
    print(f"   Input dtype: {in_d[0]['dtype']}, shape: {in_d[0]['shape']}")
    # Si INT8: in_d[0]['dtype'] debe ser np.uint8, no np.float32

    # Validar que TFLite_Detection_PostProcess está embebido (4 outputs)
    assert len(out_d) == 4, \
        f"PTQ TFLite con NMS embebido debe tener 4 outputs (boxes/classes/scores/num_detections), got {len(out_d)}"
    print(f"   Outputs: {len(out_d)} tensores (boxes, classes, scores, num_detections)")
except ValueError as e:
    if "TFLite_Detection_PostProcess" in str(e):
        print("❌ FALLO: custom op TFLite_Detection_PostProcess no encontrado.")
        print("   Aplicar plan B D15 (§17) en el Nano.")
    else:
        raise
```

---

## 17. D15 — Plan B Coral wheel

### 17.1 Naturaleza del custom op

`TFLite_Detection_PostProcess` no es un Select TF op; es un **op nativo compilado** en `tensorflow/lite/kernels/detection_postprocess.cc`. **Debería estar incluido** en cualquier build completo de TFLite (no en builds minimal `tflite-micro` o similares).

> *"The `tflite_runtime` package is a fraction the size of the full `tensorflow` package and includes the bare minimum code required to run inferences with LiteRT — primarily the `Interpreter` Python class."*
> — Google AI Edge docs, 2025.

### 17.2 GAP del wheel NVIDIA

**No existe documentación pública** que confirme verbatim si el wheel NVIDIA `tensorflow==2.5.0+nv21.8` (preinstalado en JetPack 4.6.1) incluye `TFLite_Detection_PostProcess` compilado. La arquitectura es distinta al wheel Coral: NVIDIA compiló TF completo con GPU/CUDA; Coral compiló solo el runtime TFLite. Inferencia: **debería estar pero sin garantía**.

### 17.3 Wheel Coral CP36 aarch64 (D15)

Si el wheel NVIDIA en el Nano lanza `Didn't find custom op TFLite_Detection_PostProcess`, fallback al wheel oficial Coral verificado disponible:

```
URL:    https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
sha256: 7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df
```

Listado verbatim en [`google-coral.github.io/py-repo/tflite-runtime/`](https://google-coral.github.io/py-repo/tflite-runtime/).

**Configuración exacta del Nano:** Python 3.6, aarch64, JetPack 4.6.1 → coincide con `cp36-cp36m-linux_aarch64`.

### 17.4 Uso en el Nano

```bash
# En el Nano (post-deploy, si plan B requerido)
wget -q https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
echo "7c58b1a9fb2d2b24d6f0b0f8629ede7d288358e2cb93c68c3e4f78fd0ee7d1df  tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl" | sha256sum -c -
pip3 install tflite_runtime-2.5.0.post1-cp36-cp36m-linux_aarch64.whl
```

```python
# inference_nano.py (en el Nano)
import tflite_runtime.interpreter as tflite  # NO `from tensorflow.lite`
interp = tflite.Interpreter("model_int8.tflite")
interp.allocate_tensors()
# resto del pipeline de inferencia
```

### 17.5 Repos de referencia con build TF JetPack 4.6.1

- [`Qengineering/TensorFlow-JetsonNano`](https://github.com/Qengineering/TensorFlow-JetsonNano) — wheels TF para Jetson Nano (incluye TFLite).
- [`PINTO0309/Tensorflow-bin`](https://github.com/PINTO0309/Tensorflow-bin) — wheels alternativos con XNNPACK + Multi-Threads.
- [`xuhj-code/Tensorflow-bin`](https://github.com/xuhj-code/Tensorflow-bin) — fork con custom ops MediaPipe.

### 17.6 Detección anticipada en Vast.ai (Gate 2)

```python
try:
    interp = tflite.Interpreter("track_a/exports/model.tflite")
    interp.allocate_tensors()
    print("OK: TFLite_Detection_PostProcess registrado en runtime 2.5")
except ValueError as e:
    if "TFLite_Detection_PostProcess" in str(e):
        print("FALLO: custom op no encontrado. Aplicar D15 fallback en el Nano.")
    else:
        raise
```

---

## 18. Workaround `flatbuffer_utils.py`

Archivo [`tensorflow/lite/tools/flatbuffer_utils.py`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/flatbuffer_utils.py) permite parsear y reescribir flatbuffers. **No existe script oficial de downgrade de versión**, pero es técnicamente posible bajar manualmente el campo `version` de cada operador.

**Frágil, no documentado oficialmente.** Solo aplicar si:

1. Gate 1 detecta `op_version` excedido.
2. Re-export con D12 flags no resolvió.
3. Re-train con dataset distinto no resolvió (caso patológico).

```python
# Patrón (esqueleto, requiere adaptación)
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb

# Cargar
with open("model_int8.tflite", "rb") as f:
    buf = bytearray(f.read())
model = schema_fb.ModelT.InitFromBuf(buf, 0)

# Modificar (e.g., bajar CONV_2D v5 → v4 si runtime acepta v4)
for op_code in model.operatorCodes:
    if op_code.builtinCode == 1 and op_code.version > 4:  # CONV_2D
        print(f"Downgrade CONV_2D v{op_code.version} -> v4")
        op_code.version = 4

# Re-serializar
builder = flatbuffers.Builder(1024)
builder.Finish(model.Pack(builder), b"TFL3")
with open("model_int8_downgraded.tflite", "wb") as f:
    f.write(bytes(builder.Output()))
```

**Riesgo:** el campo `version` del op_code es solo metadata; las características reales del operador no cambian. Si TF 2.15 genera CONV_2D con features de v5 (e.g., parámetro `depthwise_multiplier` o `dilation`), bajar version a 4 hace que el runtime intente parsearlo como v4 y falle de forma distinta.

**Veredicto:** workaround último recurso. Prefiere re-export con D12 flags o re-train.

---

## 19. Gate 3 — ONNX ops blacklist

### 19.1 Lista soportada por TRT 8.2-GA

Recuperada verbatim de [`onnx-tensorrt/docs/operators.md release/8.2-GA`](https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md):

> *"TensorRT 8.2 supports operators up to Opset 13."*

Tabla parcial relevante para YOLOv8n:

| Op ONNX | Soportado TRT 8.2 | Tipos | Restricciones |
|---------|--------------------|-------|---------------|
| `Add` | Sí | FP32, FP16, INT32 | — |
| `Concat` | Sí | FP32, FP16, INT32, INT8, BOOL | — |
| **`ConstantOfShape`** | Sí | **FP32 únicamente** | Riesgo si onnxslim genera tipos INT64 |
| `Conv` | Sí | FP32, FP16, INT8 | 2D o 3D, pesos como initializer |
| `Equal` | Sí | FP32, FP16, INT32 | — |
| `Gather` / `GatherND` | Sí | FP32, FP16, INT8, INT32 | Issue [`#4383`](https://github.com/NVIDIA/TensorRT/issues/4383) Gather rank-0 con opsets ≥17; opset 11 lo evita |
| **`GridSample`** | **❌ AUSENTE** | — | Solo en YOLOv8-seg, no detección |
| `NonMaxSuppression` | Sí `[EXPERIMENTAL]` | FP32, FP16 | Inputs como initializers; evitar con `nms=False` |
| `Range` | Sí | FP32, FP16, INT32 | Inputs flotantes solo como initializers |
| `Reshape` | Sí | FP32, FP16, INT32, INT8, BOOL | — |
| `Resize` | Sí | FP32, FP16 | Modos: `half_pixel`, `pytorch_half_pixel`, `tf_half_pixel_for_nn`, `asymmetric`, `align_corners`; interpolación: `nearest`, `linear` |
| `ScatterND` | Sí | FP32, FP16, INT8, INT32 | — |
| `Sigmoid` / `HardSigmoid` | Sí | FP32, FP16, INT8 | — |
| `Softmax` | Sí | FP32, FP16 | — |
| `Upsample` | Sí | FP32, FP16 | — |
| `Where` | Sí | FP32, FP16, INT32, BOOL | — |

### 19.2 Blacklist TRT 8.2 (ops NO soportadas, evitar)

| Op | Por qué bloquear |
|----|------------------|
| `GridSample` | **No aparece en operators.md** (sí en TRT 8.4+). Solo YOLOv8-seg lo usa |
| `DFT`, `IsInf`, `IsNaN` | Operaciones numéricas avanzadas; no soportadas en TRT 8.2 |
| `MelWeightMatrix`, `STFT` | Audio/señales, no aplicable |
| `SequenceInsert` | Operadores de secuencia (NLP); no aplicable |
| `CumSum` | Suma acumulada; no soportada en TRT 8.2 (sí en 8.4+) |
| `NonZero` | No soportada |
| `RoiAlign` | Solo segmentación; no aplica YOLOv8 detect |
| `QLinearConv`, `QLinearMatMul` | Solo INT8 ONNX (QDQ flow), no aplica |
| `Reciprocal` | No soportada (improbable en YOLOv8n FP32 estático) |

### 19.3 Análisis YOLOv8n detección pura con flags actuales

Con el export canónico de Track B (D13 + §6.5):

```python
model.export(format="onnx", imgsz=416, opset=11,
             simplify=True, dynamic=False, nms=False)
```

- `GridSample` aparece **solo en YOLOv8-seg**, no en detección. ✅ No problema.
- `NonMaxSuppression` **NO está en el grafo** con `nms=False`. ✅
- `ConstantOfShape` con tipo no-FP32 → **riesgo bajo** si `dynamic=False` (genera initializers estáticos). Validación requerida.
- `Resize` con `mode='linear'` o `'nearest'` → soportado por opset 11.

### 19.4 Script de inspección Gate 3

```bash
source /opt/venv/trackb/bin/activate
pip install onnx
```

```python
# scripts/validate_onnx_ops.py
import onnx
from pathlib import Path

MODEL = Path("track_b/exports/best.onnx")

m = onnx.load(str(MODEL))

# Verificar IR y opset
assert m.opset_import[0].version == 11, f"Opset {m.opset_import[0].version} ≠ 11"
assert m.ir_version <= 10, f"IR version {m.ir_version} > 10 puede romper onnxslim 0.6.x / TRT 8.2"
print(f"✅ Opset: {m.opset_import[0].version}, IR: {m.ir_version}")

# Inspeccionar ops
ops = sorted({n.op_type for n in m.graph.node})
print(f"Ops en el grafo: {ops}")

BLACKLIST_TRT82 = {
    "GridSample", "DFT", "IsInf", "IsNaN",
    "MelWeightMatrix", "STFT",
    "SequenceInsert", "CumSum",
    "NonZero", "RoiAlign",
    "QLinearConv", "QLinearMatMul",
    "Reciprocal",
}
hits = set(ops) & BLACKLIST_TRT82
if hits:
    print(f"❌ Ops problemáticas TRT 8.2 encontradas: {hits}")
    raise SystemExit(1)
else:
    print("✅ Ninguna op de la blacklist TRT 8.2 presente.")

# Verificar ConstantOfShape NO usa tipos exóticos
for n in m.graph.node:
    if n.op_type == "ConstantOfShape":
        for a in n.attribute:
            if a.name == "value":
                dt = a.t.data_type
                # ONNX TensorProto.FLOAT == 1
                if dt != 1:
                    print(f"❌ ConstantOfShape con dtype {dt} (debe ser 1=FP32)")
                    raise SystemExit(1)
                print(f"✅ ConstantOfShape dtype: {dt} (FP32)")

# Verificar NMS no embebido (debería ser False con nms=False)
nms_ops = [n for n in m.graph.node if n.op_type == "NonMaxSuppression"]
if nms_ops:
    print(f"⚠️  NonMaxSuppression encontrado ({len(nms_ops)} nodos). Verificar flags del export (nms=False).")
else:
    print("✅ Sin NonMaxSuppression embebido (NMS se hará en CPU NumPy en Nano).")

print("\n✅ Gate 3 OK — modelo listo para Gate 4 (Polygraphy).")
```

---

## 20. Gate 4 — Polygraphy Docker NGC

### 20.1 Por qué Docker NGC y no `pip install`

**Crítico:** si en el container Vast.ai (CUDA 12.4) corres `pip install polygraphy tensorrt`, obtienes **TRT 10+, no 8.2**. Para validar contra el TRT exacto del Nano (8.2.1.8), **es obligatorio** usar el Docker NGC `21.11-py3` que tiene TRT 8.2.1 + CUDA 11.5 + Ubuntu 20.04.

### 20.2 Imágenes NGC TensorRT candidatas

| Imagen NGC | Versión TensorRT | CUDA | Ubuntu | Python | Coincide JetPack 4.6.1 |
|------------|-------------------|------|--------|--------|------------------------|
| `nvcr.io/nvidia/tensorrt:21.10-py3` | 8.0.x | 11.4 | 20.04 | 3.8 | No |
| **`nvcr.io/nvidia/tensorrt:21.11-py3`** | **8.2.1** | 11.5 | 20.04 | 3.8 | ✅ **Sí (versión exacta)** |
| `nvcr.io/nvidia/tensorrt:22.01-py3` | 8.2.3 | 11.5 | 20.04 | 3.8 | Aproximado (no exacto) |

### 20.3 Polygraphy 0.49.x

[`NVIDIA/TensorRT/tools/Polygraphy/CHANGELOG.md`](https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/CHANGELOG.md) — versión actual `v0.49.27` (2025).

> *"Fixed a bug where `explicit_batch` would be provided by default on TRT 10.0, where it has been removed."*
> — v0.49.5, 2024-01-16.

Esto confirma que **polygraphy 0.49.x funciona con TRT 8 Y TRT 10**. No existe serie 0.50+. PyPI lista hasta `0.49.26`.

### 20.4 Comando de validación canónico

```bash
# 1. Pull imagen NGC (~6 GB, una sola vez)
docker pull nvcr.io/nvidia/tensorrt:21.11-py3

# 2. Validación completa: parser ONNX + comparación numérica TRT vs ORT
docker run --rm --gpus all \
  -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tensorrt:21.11-py3 \
  bash -c "
    pip install -q polygraphy onnx &&
    polygraphy run /workspace/track_b/exports/best.onnx \
      --onnxrt --trt \
      --atol 1e-2 --rtol 1e-2 \
      --input-shapes images:[1,3,416,416]
  "
```

**Flags clave:**

- `--onnxrt` — corre el modelo con ONNX Runtime (referencia).
- `--trt` — corre con TensorRT 8.2.1 dentro del container.
- `--atol 1e-2 --rtol 1e-2` — tolerancia absoluta y relativa para comparar outputs (suficiente para FP16; FP32 usaría 1e-5).
- `--input-shapes images:[1,3,416,416]` — shape fijo. Para dynamic shapes usar formato `--input-shapes images:[min,opt,max]`.

### 20.5 Alternativas (informativo)

```bash
# Solo validación ligera del parser (sin Docker, instala TRT 10 inútil para Nano)
pip install onnx onnxruntime polygraphy
polygraphy inspect model track_b/exports/best.onnx --display-as=trt

# trtexec dentro del Docker NGC (alternativa a polygraphy)
docker run --rm --gpus all \
  -v "$(pwd)":/workspace \
  nvcr.io/nvidia/tensorrt:21.11-py3 \
  trtexec --onnx=/workspace/track_b/exports/best.onnx \
          --shapes=images:1x3x416x416 \
          --fp16 \
          --verbose 2>&1 | grep -E "ERROR|WARNING|Parsing|Building"
```

### 20.6 Limitación arquitectural — x86 + TRT 8.2 ≠ aarch64 Maxwell `sm_53`

Este gate es **necesario pero no suficiente**:

| Aspecto | x86 + TRT 8.2 vía Docker | aarch64 `sm_53` Nano real |
|---------|--------------------------|----------------------------|
| Validación parser ONNX | ✅ | ✅ |
| Detección ops fuera de opset 13 | ✅ | ✅ |
| Comparación numérica `--onnxrt` | ✅ | ✅ |
| Tiempos reales en Maxwell | ❌ | ✅ |
| Fusiones de kernels específicas Maxwell | ❌ | ✅ |
| Comportamiento INT8 calibrador (sin TC) | ❌ | ✅ |

**Implicación:** después de pasar Gate 4 en Vast.ai, ejecutar una **corrida rápida** en el Nano (10–15 min de smoke test) antes del primer ciclo de training completo. Compilar el `.engine` con:

```bash
# En el Nano (post-deploy, primer smoke test)
trtexec --onnx=best.onnx \
        --fp16 \
        --workspace=1024 \
        --saveEngine=best.engine \
        --verbose 2>&1 | tee trt_build.log

# Validar latencia
trtexec --loadEngine=best.engine \
        --shapes=images:1x3x416x416 \
        --iterations=100
# Esperar latencia ~30-50 ms = 20-33 FPS
```

### 20.7 Polygraphy en el Nano (no funcional)

**GAP confirmado:** Polygraphy **NO funciona en JetPack 4.6.1** porque Python 3.6.9 es incompatible con polygraphy 0.45+ (requiere Py 3.8+). En el Nano, validar con `trtexec` directo (no polygraphy).

Referencia: [foro NVIDIA #349598](https://forums.developer.nvidia.com/t/how-to-generate-and-verify-an-int8-calibration-cache-cache-for-trtexec-on-on-jetson-nano-tensorrt-8-2-1-8-polygraphy-failing-on-device/349598) "How to generate and verify an INT8 calibration cache (.cache) for trtexec on Jetson Nano (TensorRT 8.2.1.8) — Polygraphy failing on-device".

### 20.8 ONNX Runtime + TRT EP en Nano (no viable)

ORT + TRT EP requiere CUDA 11.4 (ORT 1.11+); Nano tiene CUDA 10.2. **Ruta no viable.** Inferencia en el Nano usa TRT Python bindings directos + `cuda-python 11.0`.

---

## 21. INT8 Maxwell cierre del gap

### 21.1 Resumen del problema

La literatura 2024–2026 presenta **evidencia directamente contradictoria** sobre INT8 PTQ en Jetson Nano B01 (Maxwell `sm_53`, TensorRT 8.2.1). No existe ningún paper peer-reviewed ni preprint arXiv que haya caracterizado el trade-off directamente sobre Maxwell `sm_53`. El gap declarado en Ronda 4 es **irreductible con la literatura actual**.

### 21.2 Mecanismo teórico decisivo: ausencia de `dp4a`

Maxwell `sm_53` **carece de la instrucción `dp4a` (dot product de 4 × INT8 acumulando a INT32)** introducida en Pascal `sm_61` (2016). Sin `dp4a`, TensorRT tiene tres opciones:

1. **Usar kernels CUDA INT8 SIMD vía `dp4a`** → **no disponible en `sm_53`**.
2. **Emular INT8 vía FP16/FP32** → elimina cualquier beneficio de velocidad y añade overhead.
3. **Mixed precision fallback** → TensorRT revierte la capa a FP16, generando grafo mixto con conversiones adicionales.

Confirmación oficial NVIDIA (issue [`NVIDIA/TensorRT#3762`](https://github.com/NVIDIA/TensorRT/issues/3762)):

> *"`--int8` means Enable int8 precision, in addition to fp32."*

Es decir, INT8 nunca reemplaza FP32: lo complementa, y las capas no cuantizables revierten.

**Conclusión mecánica:** el speedup INT8 en Maxwell `sm_53` es **estructuralmente nulo** por arquitectura de hardware. La degradación de mAP ocurre igualmente (cuantización modifica pesos y activaciones independientemente del hardware), pero sin la contraparte de velocidad que la justifique.

### 21.3 Evidencia primaria (fuentes contradictorias)

#### 21.3.1 Fuente 1 — Qengineering (confianza media-alta, repo activo TRT 8.x para Nano B01)

[`Qengineering/YoloV8-TensorRT-Jetson_Nano`](https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano), rama `tensorrt8`:

> *"All models are quantized to `FP16`. The `int8` models don't give any increase in FPS, while, at the same time, their mAP is significantly worse."*

Tabla FP16 reportada:

| Modelo | Nano B01 (FPS) | Orin Nano (FPS) |
|--------|----------------|------------------|
| YOLOv8n | 19 | 100 |
| YOLOv8s | 9,25 | 100 |

El autor **no publica tabla INT8** porque concluye que el upside es nulo y el daño a mAP es significativo.

#### 21.3.2 Fuente 2 — espstack.com (confianza baja, sin metodología verificable)

[`espstack.com/blogs/posts/yolov8-jetson-nano.html`](https://espstack.com/blogs/posts/yolov8-jetson-nano.html):

| Modelo | Formato | FPS | Latencia (ms) | mAP50 |
|--------|---------|-----|---------------|-------|
| YOLOv8n | PyTorch FP32 | 7–9 | 110–140 | 0,887 |
| YOLOv8n | TRT FP16 | 18–22 | 45–55 | 0,885 |
| YOLOv8n | TRT INT8 | 28–32 | 31–36 | 0,878 |

Esta fuente reporta caída de mAP50 FP16 vs INT8 de solo 0,7 pp y +50% FPS. Pero:

- (a) dataset de calibración no especificado,
- (b) mAP50 0,887 coherente con COCO no con custom 3 clases,
- (c) sin código reproducible,
- (d) **contradice directamente a Qengineering**.

#### 21.3.3 Fuente 3 — `the0807/YOLOv8-ONNX-TensorRT` (Orin Nano `sm_87`, con TC INT8 reales)

| Cuantización | FPS | mAP val 50–95 |
|--------------|-----|---------------|
| FP16 | 60 | 37,1 |
| INT8 | 63 | 33,0 |

Drop mAP50-95: −4,1 pp con apenas +5% FPS. Sobre hardware **con** Tensor Cores INT8. Esto sugiere que incluso en arquitecturas modernas el speedup es modesto para nano/small.

### 21.4 Decisión D14 — FP16-only por default

**Track B se queda en FP16-only por default.** Experimento INT8 opcional 45–60 min en el propio Jetson Nano, **únicamente si hay margen de tiempo antes de la entrega**.

### 21.5 Protocolo del experimento INT8 opcional en Nano

```bash
# 1. Compilar engine INT8 con calibración
trtexec --onnx=yolov8n_custom.onnx \
        --saveEngine=yolov8n_int8.engine \
        --int8 \
        --calib=calib_list.txt \
        --workspace=1024

# Generar calib_list.txt con paths a imágenes de calibración (~100-500 imágenes del val set)

# 2. Medir mAP@0.5 en val set completo
python validate_engine.py --engine yolov8n_int8.engine --data data.yaml

# 3. Medir FPS empírico
trtexec --loadEngine=yolov8n_int8.engine --iterations=100
# Comparar contra FP16:
trtexec --loadEngine=yolov8n_fp16.engine --iterations=100
```

### 21.6 Criterio binario de decisión

Definido en D14 (ledger):

- **Si** `FPS_INT8 < FPS_FP16 × 1,10` (menos de 10% de ganancia en FPS) **O** `mAP_INT8 < mAP_FP16 − 5 pp`, **abandonar INT8** y consolidar FP16-only.
- **Si NO** (FPS gana ≥ 10% **Y** mAP cae < 5 pp), **adoptar INT8**.

**Importante:** zona gris (entre +0% y +10% FPS) → abandonar por el criterio del 10%. Evitamos optimización ambigua que añade complejidad sin gain claro.

### 21.7 Calibración INT8 — quirks documentados

- Foro NVIDIA #349598 confirma que **Polygraphy + INT8 falla en JetPack 4.6.1** (incompat. Py 3.6). Usar `IInt8EntropyCalibrator2` Python custom + `trtexec --int8 --calib=<cache>`.
- Foro NVIDIA #331356 confirma que `TRT INT8 conversion fails with assertion error using Ultralytics` en Orin → quirk de Ultralytics export, no del runtime. Mitigación: re-export ONNX manualmente y feed a trtexec directo.

---

## 22. Pipeline validate_artifacts

### 22.1 Pipeline Track A

```bash
source /opt/venv/tracka/bin/activate

# Gate 1 — inspección op_version
pip install tflite==2.5.0
python scripts/validate_tflite_ops.py track_a/exports/model_int8.tflite
# Exit code != 0 si alguna op excede runtime 2.5

# Gate 2 — carga test
pip install "https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl"
python scripts/validate_tflite_load.py track_a/exports/model_int8.tflite

# Si Gate 2 falla con TFLite_Detection_PostProcess missing:
#   → Documentar para aplicar D15 (wheel Coral CP36) en el Nano
#   → NO abortar export, el Plan B existe
```

### 22.2 Pipeline Track B

```bash
source /opt/venv/trackb/bin/activate

# Gate 3 — inspección ops contra blacklist TRT 8.2
pip install onnx
python scripts/validate_onnx_ops.py track_b/exports/best.onnx
# Exit code != 0 si hay ops blacklisted o ConstantOfShape no-FP32

# Gate 4 — Polygraphy en Docker NGC
docker pull nvcr.io/nvidia/tensorrt:21.11-py3
docker run --rm --gpus all -v "$(pwd)":/ws nvcr.io/nvidia/tensorrt:21.11-py3 \
  bash -c "pip install -q polygraphy onnx && \
           polygraphy run /ws/track_b/exports/best.onnx \
             --onnxrt --trt --atol 1e-2 --rtol 1e-2 \
             --input-shapes images:[1,3,416,416]"
# Exit code != 0 si TRT 8.2 no puede construir el engine o si los outputs divergen
```

### 22.3 Script `validate_artifacts.py` consolidado

CLI consolidado con flags `--track {A,B}` y `--model <path>`. Esquema:

```python
# scripts/validate_artifacts.py
"""
Validación pre-deploy unificada para artefactos embebidos-3.

Uso:
    python validate_artifacts.py --track A --model track_a/exports/model_int8.tflite
    python validate_artifacts.py --track B --model track_b/exports/best.onnx
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def validate_track_a(model_path: Path) -> dict:
    """Gate 1 (op_version) + Gate 2 (load test) para Track A."""
    result = {"track": "A", "model": str(model_path), "gates": {}}

    # Gate 1
    # ... (código de §15)
    result["gates"]["op_version"] = {"status": "pass", "ops_inspected": N, "violations": []}

    # Gate 2
    # ... (código de §16)
    result["gates"]["load_test"] = {"status": "pass", "outputs": 4, "input_dtype": "uint8"}
    result["gates"]["tflite_detection_postprocess"] = {"status": "pass"}  # o "fallback_d15_required"

    return result

def validate_track_b(model_path: Path) -> dict:
    """Gate 3 (ops blacklist) + Gate 4 (polygraphy en Docker NGC) para Track B."""
    result = {"track": "B", "model": str(model_path), "gates": {}}

    # Gate 3
    # ... (código de §19.4)
    result["gates"]["ops_blacklist"] = {"status": "pass", "ops_present": [...], "violations": []}

    # Gate 4
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{model_path.parent.parent}:/ws",
        "nvcr.io/nvidia/tensorrt:21.11-py3",
        "bash", "-c",
        f"pip install -q polygraphy onnx && "
        f"polygraphy run /ws/{model_path.name} --onnxrt --trt "
        f"--atol 1e-2 --rtol 1e-2 --input-shapes images:[1,3,416,416]"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    result["gates"]["polygraphy"] = {
        "status": "pass" if proc.returncode == 0 else "fail",
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
    }

    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=["A", "B"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("validation_report.json"))
    args = parser.parse_args()

    if args.track == "A":
        report = validate_track_a(args.model)
    else:
        report = validate_track_b(args.model)

    # Persistir JSON + log markdown
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    # Exit code != 0 si algún gate falló
    all_pass = all(g.get("status") == "pass" for g in report["gates"].values())
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()
```

### 22.4 Riesgos residuales del pipeline de validación

| # | Riesgo / GAP | Mitigación |
|---|--------------|------------|
| R1 | TF 2.15 puede generar `op_version > 2.5 max` para `DEPTHWISE_CONV_2D` o `FULLY_CONNECTED` sin documentación pública | Inspección obligatoria del flatbuffer (Gate 1). Workaround `flatbuffer_utils.py` si falla. Re-train con converter legacy. |
| R2 | Wheel NVIDIA `tensorflow==2.5.0+nv21.8` puede no incluir `TFLite_Detection_PostProcess` (sin confirmación verbatim) | Fallback Coral wheel CP36 aarch64 (D15). Verificación anticipada en Gate 2. |
| R3 | Drop INT8 YOLOv8n Maxwell `sm_53` no caracterizado en literatura | Confirmado FP16-only por mecanismo (`dp4a` ausente). Experimento opcional 1 h en Nano (D14). |
| R4 | Polygraphy en container Vast.ai (CUDA 12.4) instala TRT 10 por defecto, no 8.2 | Docker NGC `tensorrt:21.11-py3` obligatorio (Gate 4). |
| R5 | Validación x86 con TRT 8.2.1 no equivale a Maxwell `sm_53` (fusiones de kernels difieren) | Gate necesario pero no suficiente. Smoke test rápido en Nano (5–15 min) antes del primer training completo. |
| R6 | EfficientNMS_TRT plugin roto en Maxwell con TRT 8.x (issue #1538) | `nms=False` en export ONNX; NMS en CPU NumPy con `cv2.dnn.NMSBoxes` en Nano. |
| R7 | TF 2.15 INT8 PTQ puede no preservar `TFLite_Detection_PostProcess` si el SavedModel se reconstruye | Validar pre-export con `tf.saved_model.load()` y revisar signatures. Gate 2 (carga test) confirma post-export. |
| R8 | `ConstantOfShape` con tipos no-FP32 en ONNX por upgrades futuros de Ultralytics | Inspección Gate 3 detecta y bloquea. Re-export con flags conservadores si aparece. |
| R9 | Docker daemon no disponible en Vast.ai (Docker-in-Docker) | Verificación: `docker info` en bootstrap. Si falla, ejecutar Gate 4 en máquina x86 separada (local Windows con Docker Desktop). |
| R10 | Build TRT engine en Nano falla por OOM (issue #14751) | `trtexec --workspace=1024` (vs default que puede ser mayor); cerrar JupyterLab/desktop antes de build. |

---

## 23. Engine TRT compilado en Nano

### 23.1 Regla vinculante (D8)

**El `.engine` siempre se compila en el Nano**, nunca en Vast.ai ni transferido. Justificación: TensorRT engines son **GPU-architecture-specific y TRT-version-specific**. Un engine compilado en RTX 4090 (Ada `sm_89`, TRT 10) NO ejecutará en Jetson Nano (Maxwell `sm_53`, TRT 8.2.1). Incluso con la misma versión de TRT, las fusiones de kernels difieren entre `sm_89` y `sm_53` por catálogo de tactics.

### 23.2 Comando build en el Nano

```bash
# En el Nano
trtexec --onnx=best.onnx \
        --fp16 \
        --workspace=1024 \
        --saveEngine=best.engine \
        --verbose 2>&1 | tee trt_build.log
```

**Quirk OOM (issue [`ultralytics/ultralytics#14751`](https://github.com/ultralytics/ultralytics/issues/14751)):** TRT build necesita workspace; el Nano tiene 4 GB compartida CPU/GPU. `--workspace=1024` MB es el sweet spot. Si OOM persiste, bajar a 512.

### 23.3 Track A no aplica

Track A corre en **CPU TFLite + XNNPACK + NEON**, no usa TensorRT. No hay paso de "compilar engine". El `.tflite` se carga directamente vía `tflite_runtime.Interpreter`.

---

## 24. Pipeline de inferencia

### 24.1 Track A — TFLite + XNNPACK en CPU

```python
# inference_track_a.py (en el Nano)
import tflite_runtime.interpreter as tflite  # wheel Coral CP36 si D15 aplica
import cv2
import numpy as np

# Cargar modelo INT8
interp = tflite.Interpreter("model_int8.tflite", num_threads=4)
interp.allocate_tensors()

in_d = interp.get_input_details()[0]
out_d = interp.get_output_details()

# Captura GStreamer (ver §2.4)
cap = cv2.VideoCapture(
    "v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720 ! "
    "jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink",
    cv2.CAP_GSTREAMER
)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Preprocess: 320×320 + INT8 quant
    inp = cv2.resize(frame, (320, 320))
    inp = inp.astype(np.uint8)  # input INT8
    interp.set_tensor(in_d['index'], inp[np.newaxis, ...])
    interp.invoke()

    # Outputs (TFLite_Detection_PostProcess embebido):
    # 0: boxes [1, N, 4], 1: classes [1, N], 2: scores [1, N], 3: num_detections [1]
    boxes = interp.get_tensor(out_d[0]['index'])[0]
    classes = interp.get_tensor(out_d[1]['index'])[0]
    scores = interp.get_tensor(out_d[2]['index'])[0]
    n = int(interp.get_tensor(out_d[3]['index'])[0])

    # Render
    for i in range(n):
        if scores[i] < 0.5: continue
        # boxes en formato [ymin, xmin, ymax, xmax] normalizado
        ymin, xmin, ymax, xmax = boxes[i]
        h, w = frame.shape[:2]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)
        cls_id = int(classes[i])
        cls_name = ["glass", "paper", "plastic"][cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {scores[i]:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Track A", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
```

### 24.2 Track B — TRT engine + NMS CPU NumPy

```python
# inference_track_b.py (en el Nano)
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Cargar engine
with open("best.engine", "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Allocate buffers
input_shape = (1, 3, 416, 416)
input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
output_shape = (1, 25200, 7)  # YOLOv8n detect: 25200 anchors × (cx,cy,w,h,conf,c0,c1,c2)
output_size = trt.volume(output_shape) * np.dtype(np.float32).itemsize

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)
h_output = np.empty(output_shape, dtype=np.float32)

stream = cuda.Stream()

def nms_numpy(boxes, scores, iou_threshold=0.45):
    """NMS en CPU NumPy (EfficientNMS_TRT roto en Maxwell, issue #1538)"""
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                               score_threshold=0.5, nms_threshold=iou_threshold)
    return indices.flatten() if len(indices) > 0 else []

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    # Preprocess
    inp = cv2.resize(frame, (416, 416))
    inp = inp.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)  # HWC -> CHW
    inp = np.ascontiguousarray(inp[np.newaxis, ...])

    # Inference
    cuda.memcpy_htod_async(d_input, inp, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # Postprocess (decode + NMS)
    output = h_output[0]  # (25200, 7)
    # Cada row: cx, cy, w, h, conf, c_glass, c_paper, c_plastic
    conf = output[:, 4]
    mask = conf > 0.5
    output = output[mask]

    if len(output) == 0:
        cv2.imshow("Track B", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    boxes_cxcywh = output[:, :4]
    classes_scores = output[:, 5:]
    classes = np.argmax(classes_scores, axis=1)
    scores = output[:, 4] * np.max(classes_scores, axis=1)

    # cx,cy,w,h -> x1,y1,x2,y2
    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    # NMS CPU
    keep = nms_numpy(boxes_xyxy.tolist(), scores.tolist())

    h, w = frame.shape[:2]
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i] * np.array([w/416, h/416, w/416, h/416])
        cls_name = ["glass", "paper", "plastic"][classes[i]]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{cls_name} {scores[i]:.2f}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Track B", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
```

---

## 25. Topología y escenarios

### 25.1 Hardware involucrado

| Componente | Especificación |
|---|---|
| Jetson Nano | Developer Kit 4 GB B01, JetPack 4.6.5 (post-apt-upgrade), kernel 4.9.337-tegra, Ubuntu 18.04 bionic, aarch64 |
| Antena Wi-Fi USB | **TP-Link TL-WN722N v4** — chipset Realtek **RTL8188EUS** (USB ID `2357:010c`, no `0bda:8179`) |
| Wi-Fi interno | **Intel M.2 Key E** (MAC OUI `3c:64:cf`, driver `iwlwifi` in-tree) |
| Equipo de desarrollo | Windows 11 Pro (mitgar14) |
| Servidor bastion | Contabo VPC (asumido Ubuntu 20.04+/22.04, IP pública, acceso root SSH) |

**Crítico:** el TL-WN722N v1 usaba Atheros AR9271 (soporte mainline, monitor + AP). El **v4 usa RTL8188EUS** (Realtek), que requiere driver out-of-tree para AP mode estable. NO asumir specs del v1.

### 25.2 Escenarios de uso a cubrir

| Escenario | Cuándo aplica | Conectividad necesaria |
|---|---|---|
| **Development en casa/UAO** | Próximas 2 semanas | SSH directo a IP local + JupyterLab port-forward + opcional VNC para ajustes UI |
| **Acceso desde fuera (cualquier red)** | Cualquier momento, sobre todo si trabajo desde otro sitio | Túnel overlay (Tailscale) + opcional reverse SSH como respaldo |
| **Demo final battery-powered en UAO** | 2026-05-26 | Una de tres opciones: WiFi UAO + Tailscale, hotspot móvil + Tailscale, **o** Nano en modo AP con laptop conectado directo |

### 25.3 Escenario A — WiFi UAO disponible

```
                    Internet (UAO WiFi)
                          │
   ┌──────────────────────┼──────────────────────┐
   │                      │                      │
Jetson Nano          Contabo VPC            Win11 Laptop
(192.168.x.x)        (backup only,         (DHCP UAO)
tailscale:           con Headscale)        tailscale:
100.64.X.2                                 100.64.X.3

Protocolo: Tailscale P2P directo (misma red → latencia mínima)
Alternativa: SSH directo a IP local Nano
Comando: ssh nano  (config con IP local)
        ssh nano-tail (config con IP tailscale)
```

### 25.4 Escenario B — Sin WiFi UAO, hotspot móvil

```
                  Hotspot móvil (LTE/5G)
                          │
   ┌──────────────────────┼──────────────────────┐
   │                      │                      │
Jetson Nano          Contabo VPC            Win11 Laptop
(NAT del hotspot)    (DERP relay si        (NAT del hotspot)
tailscale:           NAT estricto)         tailscale:
100.64.X.2                                 100.64.X.3

Protocolo: Tailscale (P2P primero; DERP fallback si NAT estricto)
Fallback: Reverse SSH via Contabo:2222 → autossh siempre activo
Comando: ssh nano-tail  (recomendado)
         ssh nano-bastion  (fallback)
```

### 25.5 Escenario C — Nano en modo AP, laptop directo (sin Internet)

```
Win11 Laptop ─── WiFi direct ─── Jetson Nano (AP)
(DHCP del AP                     192.168.42.1 (wlan1)
 192.168.42.x)                   hostapd + dnsmasq

Sin Internet → sin Tailscale, sin Contabo
Protocolo: SSH directo a 192.168.42.1
NoMachine: cliente apuntando a 192.168.42.1
Comando: ssh usuario@192.168.42.1
```

### 25.6 Stack recomendado consolidado

| Componente | Elección | Razón principal |
|---|---|---|
| **SSH desde Win11** | OpenSSH built-in + ed25519 + `~/.ssh/config` | Sin instalación adicional, autenticación por clave, alias para puertos custom |
| **Escritorio remoto** | **NoMachine** + Xfce4 + **HDMI dummy plug** (D26) | Único que combina OpenGL accel (preview inferencia OpenCV), bajo overhead red (protocolo NX), cliente Win11 oficial, autostart headless |
| **Túnel overlay** | **Tailscale free tier** con `--accept-dns=false --ssh` (D27) | Evita el bug WireGuard del kernel 4.9-tegra (usa `wireguard-go` userspace), NAT traversal automático, setup en < 10 min. Workaround bug ARM64 #14902 |
| **Fallback si Tailscale Inc. falla** | Headscale self-hosted en Contabo | Mismo binario cliente, sólo cambia `--login-server` |
| **Driver Wi-Fi TL-WN722N v4** | **`lwfinger/rtl8188eu`** (D21, no aircrack-ng) | aircrack-ng NO compila en k4.9 (API mismatch verificado in-situ) |
| **Demo final sin WiFi conocido** | Nano en modo AP (hostapd compilado + dnsmasq) **o** hotspot móvil + Tailscale (D25) | Dos rutas viables; preferir hotspot móvil por simplicidad |
| **Bastion en Contabo** | Reverse SSH autossh a puerto **443** (D28) + Headscale opcional | Doble vía: tunnel persistente clásico + control plane Tailscale. Puerto 443 = bypass DPI universitario |

---

## 26. SSH OpenSSH Windows Nano

### 26.1 Verificar OpenSSH client en Win11

```powershell
Get-WindowsCapability -Online -Name OpenSSH.Client*
ssh -V
# Esperado: OpenSSH_for_Windows_8.6p1, LibreSSL 3.4.3 (o más nuevo)
```

Si no está: `Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0`.

### 26.2 Generar clave ed25519 en Windows

```powershell
ssh-keygen -t ed25519 -C "mitgar14@embebidos3"
# Guarda en C:\Users\mitgar14\.ssh\id_ed25519 (privada) + id_ed25519.pub
# Pasphrase opcional (vacío para autoconexión sin prompt)
```

### 26.3 Copia clave a la Nano (sin `ssh-copy-id` en Windows)

```powershell
# Opción cmd nativa:
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh <usuario>@<IP_NANO> "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"

# Opción PowerShell pura:
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub" | ssh <usuario>@<IP_NANO> "mkdir -p ~/.ssh && tee -a ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### 26.4 `~/.ssh/config` en Windows

Crear `C:\Users\mitgar14\.ssh\config` (Windows usa el mismo formato que Linux):

```sshconfig
Host nano
    HostName 192.168.X.X
    User <tu_usuario>
    IdentityFile C:/Users/mitgar14/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
    LocalForward 8888 localhost:8888

Host nano-tail
    HostName 100.64.X.X
    User <tu_usuario>
    IdentityFile C:/Users/mitgar14/.ssh/id_ed25519
    ServerAliveInterval 60
    LocalForward 8888 localhost:8888

Host nano-bastion
    HostName <CONTABO_IP>
    User <tu_usuario>
    Port 2222
    IdentityFile C:/Users/mitgar14/.ssh/id_ed25519
```

Uso:
```powershell
ssh nano           # LAN local
ssh nano-tail      # Via Tailscale tailnet IP (100.x)
ssh nano-bastion   # Via reverse SSH tunnel en Contabo:2222
```

El bloque `LocalForward 8888 localhost:8888` ya port-forwarea JupyterLab — luego abrir `http://localhost:8888` en el browser.

### 26.5 sshd en la Nano

JetPack 4.6.5 trae `openssh-server` instalado y habilitado por default. Verificar:

```bash
sudo systemctl status ssh
# Si no está activo:
sudo systemctl enable ssh && sudo systemctl start ssh
```

---

## 27. Tailscale con workaround DNS

### 27.1 Bug crítico verificado: tailscale/tailscale#14902

[Issue tailscale/tailscale#14902](https://github.com/tailscale/tailscale/issues/14902), abierto 2025-02-04 por dmellosanjay. Reproducible en 1.78.1, 1.80.0, 1.82.5, **1.84.0** (verificado por múltiples reporters en 2025). Sin fix oficial en repo.

**Síntoma:** al boot, `tailscaled` arranca y los logs muestran:
```
dns: [rc=unknown ret=direct]
dns: using "direct" mode
logtail: dial "log.tailscale.io:443" failed: no DNS fallback candidates remain
```
Sin DNS, Tailscale no puede contactar el control plane. Workaround empírico: `sudo systemctl restart tailscaled` post-boot funciona; pero el bug puede reaparecer en cualquier link change.

**Causa raíz inferida (comentario @ph1048 sept 2025):** Tailscale modifica `/etc/resolv.conf` al iniciar y si la shutdown anterior no fue limpia, el archivo queda apuntando al resolver interno de Tailscale (`100.100.100.100`) que no es accesible antes de que Tailscale esté up — deadlock circular.

### 27.2 Fix robusto (D27) — combinación de tres flags

```bash
# Paso 1: instalar Tailscale por repo bionic oficial
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.noarmor.gpg | \
    sudo tee /usr/share/keyrings/tailscale-archive-keyring.gpg >/dev/null
curl -fsSL https://pkgs.tailscale.com/stable/ubuntu/bionic.tailscale-keyring.list | \
    sudo tee /etc/apt/sources.list.d/tailscale.list
sudo apt update && sudo apt install -y tailscale

# Paso 2: prevenir el bug con --accept-dns=false (NO usamos MagicDNS, no necesitamos)
sudo systemctl enable --now tailscaled
sudo tailscale up --accept-dns=false --ssh

# Paso 3: backup defensivo — asegurar que resolv.conf NO es overritten por Tailscale
sudo rm /etc/resolv.conf
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
sudo systemctl enable --now systemd-resolved
```

**`--accept-dns=false`** previene el bug #14902 por completo porque Tailscale NO toca `/etc/resolv.conf`. Trade-off: pierdes MagicDNS (`nano.tailnet.ts.net` resolverá sólo si la usás directo via IP `100.x.x.x`). **Para nuestro caso es aceptable** — sólo necesitamos el túnel; el nombre lo manejamos con un alias SSH local en el Win11:

```ssh-config
# C:\Users\mitgar14\.ssh\config en Win11
Host nano-tailscale
    HostName 100.78.140.33  # IP fija de Tailscale para la Nano
    User jetson
    IdentityFile ~/.ssh/id_ed25519
```

**`--ssh`** activa **Tailscale SSH** — el daemon de Tailscale acepta conexiones SSH entrantes directamente, así no dependemos del `sshd` del sistema y tenemos auth federada (cualquier device autenticado en el tailnet puede conectarse sin gestionar claves manualmente). **Bonus para demo**: si Win11 también tiene Tailscale, podés conectarte por `ssh nano` directo sin password ni clave (Tailscale verifica identidad).

### 27.3 WireGuard userspace automático (D27 implicit)

Tailscale detecta automáticamente que el kernel no tiene módulo WireGuard (ver §32) y usa **userspace networking** (`wireguard-go` embebido). Overhead CPU ~15-25% mayor pero estable. No requiere intervención.

### 27.4 Tailscale en Win11

Descargar instalador `.exe` de [tailscale.com/download/windows](https://tailscale.com/download/windows). Login con la misma cuenta. La Win11 obtiene su propia IP `100.64.X.Y`.

Conexión: `ssh nano-tail` (ver §26.4) que apunta a `100.64.X.X`.

### 27.5 Tailscale SSH (opcional, simplifica más)

Gestiona auth SSH a través de Tailscale, sin necesidad de gestionar claves manualmente:

```bash
# En la Nano:
sudo tailscale up --ssh
# Ahora desde Win11: ssh <usuario>@100.64.X.X funciona sin clave configurada
```

Ver [tailscale.com/kb/1193/tailscale-ssh](https://tailscale.com/kb/1193/tailscale-ssh).

---

## 28. NoMachine + Xfce4 + dummy HDMI

### 28.1 Tabla comparativa (Jetson Nano + JetPack 4.6.5)

| Criterio | vino / RealVNC bundled | x11vnc | TigerVNC | **NoMachine NX** | xrdp |
|---|---|---|---|---|---|
| Latencia WiFi 100 Mbps | Media | Media | Media-baja | **Baja** (protocolo NX) | Media |
| OpenGL hardware accel | Sí (display real) | Sí (display real) | No (sesión virtual) | **Sí** con dummy HDMI | **No** (EGL roto) |
| Autostart headless sin dummy | **No** (bug conocido) | Sí con workaround | Sí | Sí (mejor con dummy) | Sí |
| Cliente Win11 | VNC Viewer / TightVNC | VNC Viewer | TigerVNC viewer | **App oficial NX** | Remote Desktop nativo |
| RAM idle en Nano | ~30 MB | ~15 MB | ~20 MB | ~80-100 MB (con Xfce: ~40-50) | ~25 MB |
| Bugs JetPack 4.6.5 | Sin HDMI no arranca | Resolución incorrecta sin dummy | Sesión nueva sin CUDA | Alto CPU si GNOME (Xfce lo arregla) | **Crash inmediato + EGL roto** |

### 28.2 Bugs específicos confirmados

- **xrdp descartado**: moderador NVIDIA `linuxdev` confirma que "EGL does not work with rdp backend" → rompe OpenGL/CUDA en la sesión RDP ([forums.developer.nvidia.com/t/xrdp-login-profile-different-from-boot-with-monitor/217306](https://forums.developer.nvidia.com/t/xrdp-login-profile-different-from-boot-with-monitor/217306)). Crash inmediato reportado en JetPack 4.6 ([#259902](https://forums.developer.nvidia.com/t/jetson-nano-headless-fails/259902)).
- **vino + GNOME headless**: requiere display físico al arrancar, sin él no inicia. Workaround: **HDMI dummy plug** (~1 USD en AliExpress, "4K HDMI emulator").
- **NoMachine + GNOME**: alto CPU (20-30% por core idle) reportado en [foro NVIDIA #77399](https://forums.developer.nvidia.com/t/jetson-nano-vnc-headless-connections/77399). Fix: instalar Xfce4 y apuntar NX al desktop Xfce ([NoMachine KB AR02R01074](https://kb.nomachine.com/AR02R01074)).

### 28.3 Receta consolidada (D26) — verificada para JP 4.6.5

```bash
# === 1. Xfce4 + dependencias (NO descomenta xdg-utils en bionic, ya viene) ===
sudo apt update
sudo apt install -y xfce4 xfce4-goodies xfce4-terminal

# === 2. Login automático en Xfce4 para autostart ===
sudo mkdir -p /etc/lightdm
sudo tee /etc/lightdm/lightdm.conf.d/12-autologin.conf <<EOF
[Seat:*]
autologin-user=$USER
autologin-user-timeout=0
user-session=xfce
EOF

# === 3. NoMachine ARM64 (verificar última versión en https://downloads.nomachine.com/linux/) ===
cd /tmp
# Reemplazar la URL del .deb con la más reciente arm64:
wget "https://download.nomachine.com/download/8.16/Arm/nomachine_8.16.1_2_arm64.deb" -O nomachine.deb
sudo dpkg -i nomachine.deb || sudo apt-get install -f -y

# === 4. Forzar Xfce4 como default desktop en NoMachine ===
sudo sed -i 's|^#DefaultDesktopCommand .*|DefaultDesktopCommand "/usr/bin/startxfce4"|' /usr/NX/etc/node.cfg
# Si la línea no existe (algunas versiones), agregarla:
grep -q "^DefaultDesktopCommand" /usr/NX/etc/node.cfg || \
  echo 'DefaultDesktopCommand "/usr/bin/startxfce4"' | sudo tee -a /usr/NX/etc/node.cfg

# === 5. Restart NoMachine ===
sudo /usr/NX/bin/nxserver --restart

# === 6. Cambiar el target default a multi-user (NO graphical), opcional ===
# Sólo si querés que la Nano NO arranque sesión X cuando no hay HDMI conectado
# y solo deje NoMachine activo. Recomendado con dummy plug puesto:
# sudo systemctl set-default multi-user.target
# Reboot para aplicar
```

### 28.4 Dummy HDMI plug

La receta más simple es **comprar un dummy HDMI físico** ($5-8 en Amazon, "headless ghost adapter", resolución 1920x1080). Plug-and-play, cero config. Funciona en Jetson Nano sin modificación de Xorg.

### 28.5 Alternativa software (sin dummy físico)

Virtual display con `xserver-xorg-video-dummy`. Funciona pero en Jetson Nano el driver `tegra` no siempre cede control al driver `dummy` limpiamente. Walkthrough en Amplifi Labs blog ([amplifilabs.com/post/nomachine-with-xfce-desktop-on-headless-vps-complete-setup-guide](https://www.amplifilabs.com/post/nomachine-with-xfce-desktop-on-headless-vps-complete-setup-guide), 2026-03-30):

```bash
sudo apt install xserver-xorg-video-dummy
sudo tee /etc/X11/xorg.conf.d/10-dummy.conf <<'EOF'
Section "Device"
    Identifier "Dummy"
    Driver "dummy"
    VideoRam 256000
EndSection
Section "Monitor"
    Identifier "Monitor0"
    HorizSync 28.0-80.0
    VertRefresh 48.0-75.0
    Modeline "1920x1080" 148.5 1920 2008 2052 2200 1080 1084 1089 1125 +hsync +vsync
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Dummy"
    Monitor "Monitor0"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Modes "1920x1080"
    EndSubSection
EndSection
EOF
```

### 28.6 Reporte de regresión post-JP 4.6.5

Ninguno reportado a mayo 2026. La KB de NoMachine sigue mostrando "Last Update: 2025-07-03" sin actualización para 4.6.5 específica; la receta es estable.

### 28.7 Video de referencia 2024 (verificable)

[make2explore Systems — Tutorial Installation of NoMachine on NVIDIA Jetson Nano](https://youtu.be/vBMHS6FXBM4), 2024-10-18, 18:41 min, chapters disponibles:

- `0:00-0:31` Start
- `0:31-0:43` Introduction
- `0:43-3:15` What is NoMachine RDS Tool?
- `3:15-5:25` Main Objective ([https://youtu.be/vBMHS6FXBM4?t=195](https://youtu.be/vBMHS6FXBM4?t=195))
- `5:25-6:43` Installing JTOP ([https://youtu.be/vBMHS6FXBM4?t=325](https://youtu.be/vBMHS6FXBM4?t=325))
- `6:43-10:30` Demo NoMachine install ([https://youtu.be/vBMHS6FXBM4?t=403](https://youtu.be/vBMHS6FXBM4?t=403))
- `10:30-18:41` Demo Xfce4 install ([https://youtu.be/vBMHS6FXBM4?t=630](https://youtu.be/vBMHS6FXBM4?t=630))

### 28.8 Alternativa TigerVNC (si OpenGL no es crítico)

**TigerVNC sobre sesión Xfce virtual** es válido y no requiere dummy HDMI:

```bash
sudo apt install -y tigervnc-standalone-server tigervnc-common xfce4
mkdir -p ~/.vnc
echo '#!/bin/bash' > ~/.vnc/xstartup
echo 'startxfce4 &' >> ~/.vnc/xstartup
chmod +x ~/.vnc/xstartup
vncserver :1 -geometry 1280x720 -depth 24 -localhost no
# Desde Win11: TigerVNC viewer apuntando a <IP_Nano>:5901
```

Sin aceleración GPU, pero suficiente para mostrar imágenes clasificadas guardadas a disco.

---

## 29. Sunshine + Moonlight descartados

**Fuentes:** Issue [moonlight-stream/moonlight-embedded#741](https://github.com/moonlight-stream/moonlight-embedded/issues/741), HN [thread 43439524](https://news.ycombinator.com/item?id=43439524) (2025-03), Stack Overflow #63479215.

**Veredicto:** Sunshine NO funciona en Jetson Nano como servidor host:

1. Sunshine en Linux ARM64 espera **NVFBC** (NVIDIA Frame Buffer Capture) que existe SOLO en GPUs Quadro/GeForce discretas. Tegra X1 no tiene NVFBC.
2. **NVENC en Tegra ≠ NVENC en discrete**: Jetson usa el video engine multimedia de L4T (V4L2) accesible solo via librerías NVIDIA Multimedia API. Sunshine no tiene driver para ese stack.
3. **No hay .deb arm64 para Bionic** en releases recientes de Sunshine (v2026.508.45922 solo distribuye .deb para Ubuntu 22.04+ arm64).
4. Issue #741 abierto desde 2019 sin resolución — sin testimonios de éxito.

**Conclusión:** mantener NoMachine como única solución de escritorio remoto. NO invertir tiempo en intentar Sunshine.

---

## 30. Contabo VPC bastion

### 30.1 Inicialización segura

```bash
# Como root vía SSH inicial (recibido en email Contabo):
apt update && apt upgrade -y

# Crear usuario deploy con sudo:
adduser deploy
usermod -aG sudo deploy

# Pasar authorized_keys de root al nuevo user:
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
```

### 30.2 Hardening sshd (`/etc/ssh/sshd_config`)

```sshconfig
Port 2200
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers deploy
X11Forwarding no
MaxAuthTries 3
ClientAliveInterval 300
ClientAliveCountMax 2
GatewayPorts clientspecified
AllowTcpForwarding yes
```

```bash
# CRÍTICO: probar conexión nueva ANTES de cerrar la actual
sudo systemctl restart sshd
# Desde otra terminal: ssh -p 2200 deploy@<CONTABO_IP>
```

### 30.3 sshd Port 443 (para D28 — bypass DPI)

Adicional al puerto 2200, configurar sshd para escuchar en 443 (indistinguible de HTTPS):

```sshconfig
# /etc/ssh/sshd_config
Port 2200
Port 443
```

```bash
# Permitir bind a puerto privilegiado:
sudo systemctl restart sshd

# Crear usuario dedicado para tunnel (sin shell):
sudo useradd -r -s /usr/sbin/nologin tunnel
sudo mkdir -p /home/tunnel/.ssh
echo "command=\"/bin/false\",no-X11-forwarding,no-agent-forwarding,no-pty <NANO_PUBKEY>" | \
  sudo tee /home/tunnel/.ssh/authorized_keys
sudo chown -R tunnel:tunnel /home/tunnel/.ssh
sudo chmod 700 /home/tunnel/.ssh
sudo chmod 600 /home/tunnel/.ssh/authorized_keys
```

### 30.4 UFW firewall

```bash
sudo apt install -y ufw
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 2200/tcp          # SSH custom
sudo ufw allow 443/tcp           # SSH alternativo (bypass DPI)
sudo ufw allow 2222/tcp          # reverse SSH tunnel
sudo ufw allow 51820/udp         # WireGuard (si alternativa B)
sudo ufw allow 80/tcp            # HTTP (si Headscale)
sudo ufw enable
sudo ufw status verbose
```

### 30.5 fail2ban opcional

```bash
sudo apt install -y fail2ban
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo nano /etc/fail2ban/jail.local
# [sshd] enabled = true, port = 2200,443
sudo systemctl restart fail2ban
```

---

## 31. autossh reverse tunnel

### 31.1 Veredicto y comparativa

autossh sigue siendo la opción correcta para nuestro caso. Mantener.

**Por qué:**
1. VPS Contabo ya pagado → "single point of failure" argumento NO aplica.
2. autossh + puerto 443 del Contabo es indistinguible de tráfico HTTPS normal — bypass de DPI universitario.
3. Cloudflared free tier: alternativa de backup viable pero requiere cliente `cloudflared` en Win11 además del Tailscale.
4. chisel v1.11.5: superior técnicamente (TLS, fingerprint, backoff built-in) pero requiere instalar y mantener `chisel server` en el VPS Contabo. **Vale el esfuerzo solo post-deadline.**
5. frp: overkill para 1 nodo edge.

### 31.2 Pre-requisitos en Contabo

```
GatewayPorts clientspecified
AllowTcpForwarding yes
```

(Ya configurado en §30.2)

### 31.3 Comando manual de prueba (en Nano)

```bash
sudo apt install -y autossh
autossh -M 0 -fNR 0.0.0.0:2222:localhost:22 \
  -o "ServerAliveInterval=60" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=no" \
  -i /home/<usuario>/.ssh/id_ed25519 \
  <usuario_contabo>@<CONTABO_IP>
```

### 31.4 Systemd unit refinada (D28)

```ini
# /etc/systemd/system/autossh-tunnel.service
[Unit]
Description=AutoSSH reverse tunnel Nano -> Contabo
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=jetson
Environment="AUTOSSH_GATETIME=0"
Environment="AUTOSSH_POLL=60"
Environment="AUTOSSH_LOGFILE=/var/log/autossh.log"
ExecStart=/usr/bin/autossh -M 0 -NT \
  -o "ServerAliveInterval=30" \
  -o "ServerAliveCountMax=3" \
  -o "ExitOnForwardFailure=yes" \
  -o "StrictHostKeyChecking=accept-new" \
  -o "UserKnownHostsFile=/home/jetson/.ssh/known_hosts_tunnel" \
  -i /home/jetson/.ssh/id_ed25519 \
  -R 0.0.0.0:2222:localhost:22 \
  -p 443 \
  tunnel@<CONTABO_IP>
Restart=always
RestartSec=15
StartLimitIntervalSec=0

[Install]
WantedBy=multi-user.target
```

**Cambios respecto a R6 inicial:**
- `After=network-online.target tailscaled.service` — espera que Tailscale levante primero por si el VPS está accesible vía tailnet
- `-p 443` — conectar al sshd del Contabo en puerto 443 (configurar Contabo con `Port 443` en `/etc/ssh/sshd_config`)
- `StartLimitIntervalSec=0` — sin límite de restarts (Restart=always estaba siendo overridden por systemd default)
- `UserKnownHostsFile` separado para el túnel — aislamiento de fingerprints
- Usuario dedicado `tunnel@` en Contabo (no `root@`) con `ForceCommand /bin/false` en authorized_keys para que solo pueda hacer port forward

### 31.5 Activar y verificar

```bash
sudo systemctl daemon-reload
sudo systemctl enable autossh-tunnel
sudo systemctl start autossh-tunnel
sudo systemctl status autossh-tunnel  # verificar
```

Desde Win11: `ssh nano-bastion` (ver §26.4).

---

## 32. WireGuard broken y userspace

### 32.1 Bug crítico WireGuard en kernel 4.9-tegra

El módulo de kernel WireGuard (DKMS) **NO compila/funciona correctamente** en el kernel 4.9-tegra de JetPack 4.6.5. Error documentado en [foro NVIDIA #184764](https://forums.developer.nvidia.com/t/kernel-error-when-using-wireguard/184764):

```
Internal error: Accessing user space memory outside uaccess.h routines: 96000005
```

**Implicación:** NO ejecutar `apt install wireguard` esperando que el módulo cargue. Usar siempre **implementación userspace** (`wireguard-go` o cliente Tailscale, que usa `wireguard-go` internamente).

### 32.2 Alternativa B — WireGuard self-host (con `wireguard-go`)

Más complejo por el bug del kernel, pero viable. Setup completo:

**En Contabo (kernel moderno, WireGuard kernel-module funciona normal):**

```bash
sudo apt install -y wireguard
cd /etc/wireguard
wg genkey | tee server_private.key | wg pubkey > server_public.key
wg genkey | tee nano_private.key   | wg pubkey > nano_public.key
wg genkey | tee win11_private.key  | wg pubkey > win11_public.key
chmod 600 *_private.key

echo "net.ipv4.ip_forward=1" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

`/etc/wireguard/wg0.conf` (Contabo):

```ini
[Interface]
PrivateKey = <SERVER_PRIVATE>
Address = 10.42.0.1/24
ListenPort = 51820
PostUp = iptables -A FORWARD -i wg0 -j ACCEPT; iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
PostDown = iptables -D FORWARD -i wg0 -j ACCEPT; iptables -t nat -D POSTROUTING -o eth0 -j MASQUERADE

[Peer]
PublicKey = <NANO_PUBLIC>
AllowedIPs = 10.42.0.2/32

[Peer]
PublicKey = <WIN11_PUBLIC>
AllowedIPs = 10.42.0.3/32
```

```bash
sudo systemctl enable wg-quick@wg0
sudo systemctl start wg-quick@wg0
sudo ufw allow 51820/udp
```

**En la Nano**: usar `wireguard-go` (userspace) en lugar del módulo:

```bash
# Instalar wireguard-tools y go
sudo apt install -y wireguard-tools golang-go

# Compilar wireguard-go
git clone https://git.zx2c4.com/wireguard-go
cd wireguard-go
make
sudo cp wireguard-go /usr/local/bin/

# Levantar interfaz wg0 con userspace driver
sudo WG_QUICK_USERSPACE_IMPLEMENTATION=wireguard-go wg-quick up wg0
```

`/etc/wireguard/wg0.conf` (Nano):

```ini
[Interface]
PrivateKey = <NANO_PRIVATE>
Address = 10.42.0.2/32

[Peer]
PublicKey = <SERVER_PUBLIC>
Endpoint = <CONTABO_IP>:51820
AllowedIPs = 10.42.0.0/24
PersistentKeepalive = 25
```

**En Win11**: app oficial WireGuard de [wireguard.com/install/](https://www.wireguard.com/install/) (firmado ZX2C4 LLC). Importar config:

```ini
[Interface]
PrivateKey = <WIN11_PRIVATE>
Address = 10.42.0.3/32
DNS = 1.1.1.1

[Peer]
PublicKey = <SERVER_PUBLIC>
Endpoint = <CONTABO_IP>:51820
AllowedIPs = 10.42.0.0/24
PersistentKeepalive = 25
```

Conexión Win11 → Nano: `ssh <usuario>@10.42.0.2`.

---

## 33. Comparativa overlay networks

### 33.1 Tabla comparativa alternativas A / B / C

| Criterio | A: Reverse SSH | B: WireGuard wireguard-go | **C: Tailscale free** | C': Headscale |
|---|---|---|---|---|
| Complejidad setup | Baja (1 unit file) | Media-alta (compilar wireguard-go) | **Muy baja** (`apt install` + login) | Media (dominio HTTPS + config) |
| Performance | Buena (TCP SSH) | Óptima (UDP, kernel en hub) | Buena (UDP, P2P cuando posible) | Igual que Tailscale |
| Dependencia Tailscale Inc. | Ninguna | Ninguna | **Sí** (control plane + DERP fallback) | Ninguna |
| NAT traversal automático | No | No | **Sí** | Sí (con tu propio DERP server) |
| Conexión P2P directa | No (Contabo siempre relay) | No (Contabo siempre hub) | **Sí** cuando red lo permite | Sí |
| Resilencia a falla Contabo | Sin acceso | Sin acceso | Tailscale Inc. + DERP siguen | Sin acceso (Headscale en Contabo) |
| Mejor para | Backup simple, scripting | Autonomía sin dependencia | **Day-to-day** | Producción seria self-hosted |

**Veredicto:** **Tailscale free tier** como primera línea. **Reverse SSH autossh** como fallback. Headscale opcional si se quiere control total más adelante.

### 33.2 Disclaimer sobre el estado de la evidencia

**No existe paper peer-reviewed que mida CPU overhead, latencia p50/p99 ni battery drain de Tailscale/ZeroTier/WireGuard específicamente sobre Cortex-A57 (Jetson Nano) o A72 (RPi 4) en condiciones de campo.** Lo que sigue se basa en (a) benchmarks comunitarios reproducibles, (b) un reporte UvA en x86 que da el overhead relativo kernel-vs-userspace, (c) extrapolación arquitectural A72→A57.

### 33.3 Throughput WireGuard puro en ARM (datos `cyyself/wg-bench` loopback netns)

Mediciones del repo [cyyself/wg-bench](https://github.com/cyyself/wg-bench) (Shell, 226 ⭐, abril 2026):

| Dispositivo | CPU / Frecuencia | OS / Kernel | Throughput |
|---|---|---|---|
| Raspberry Pi 4 | Cortex-A72 1,50 GHz | OpenWRT 23 / Linux 5.15 | **881 Mbps** |
| Raspberry Pi 4 | Cortex-A72 1,80 GHz stock | RPi OS trixie / Linux 6.12 | **777 Mbps** |
| Raspberry Pi 4 | Cortex-A72 2,00 GHz OC | OpenWRT 23 / Linux 5.15 | **1,02 Gbps** |
| Raspberry Pi 3B | Cortex-A53 1,20 GHz | OpenWRT 23 / Linux 5.15 | 522 Mbps |

**Extrapolación al Jetson Nano (Cortex-A57 1,43 GHz):** A57 es arquitecturalmente similar al A72 pero con IPC ligeramente inferior. Estimación conservadora: **600-750 Mbps** de throughput WireGuard kernel en loopback. **NO medido directamente** — sólo extrapolado. Estos números son el techo CPU del cifrado, no el throughput sobre WiFi real.

**El cuello de botella real en este proyecto es WiFi 802.11g a ~20 Mbps efectivos** (TL-WN722N v4, ver §38), o WiFi UAO/móvil a ~50-100 Mbps. A 100 Mbps, **el cifrado consume <11% de la capacidad CPU** disponible del Nano. Despreciable en términos de FPS de inferencia.

### 33.4 WireGuard kernel vs WireGuard-Go (Tailscale) — overhead CPU

Del reporte [Dekker & Spaans (UvA, 2020)](https://rp.os3.nl/2019-2020/p71/report.pdf) en hardware x86 1 Gbps:

| Implementación | Goodput TCP | CPU (1 core) | Latencia mediana |
|---|---|---|---|
| WireGuard kernel | ~940 Mbps | ~45 % | bajo |
| **WireGuard-Go (userspace) — usado por Tailscale en Linux** | ~540 Mbps | **~85 %** | el más alto |
| strongSwan AES-GCM | ~950 Mbps | ~40 % | el más bajo |
| OpenVPN AES-256-CBC | ~200 Mbps | ~75 % | alto |

**Conclusión transferible a ARM:** la relación kernel-vs-userspace (~1,5-2× más CPU para userspace) es válida arquitecturalmente. En el Nano a 100 Mbps:

- WireGuard kernel ideal: ~3-5 % CPU de un núcleo.
- WireGuard-Go (Tailscale): ~6-10 % CPU de un núcleo.

**Ambos despreciables para el proyecto.** Track B inferencia YOLOv8n TRT corre en GPU Maxwell (no compite por CPU); Track A SSD TFLite corre en CPU con XNNPACK pero deja 3 núcleos para overhead de red.

### 33.5 RAM footprint medido (datos reales GitHub issues + dev.to)

| Solución | RSS real ARM Linux |
|---|---|
| WireGuard kernel module | <5 MB (sólo herramientas userspace) |
| **Tailscale daemon `tailscaled`** | **~30-50 MB RSS** (binario 20 MB en disco) |
| ZeroTier daemon | ~30-50 MB RSS |

**Aclaración crítica:** valores `VIRT` ~540 MB que aparecen en algunos issues de Tailscale (ej. [#15435](https://github.com/tailscale/tailscale/issues/15435)) son **memoria virtual reservada por el runtime Go, no RSS físico**. El consumo físico real (`RES` en htop) son 30-50 MB.

Sobre los 4 GB unificados de la Nano, 50 MB es 1,2 % — no compite con TRT engine build (~3,5 GB pico).

### 33.6 ZeroTier — falta de evidencia

**No se encontró ningún benchmark reproducible de ZeroTier en ARM Cortex-A57/A72.** El reporte [NetFoundry 2022](https://netfoundry.io/benchmark/benchmarking%20open%20source%20networking.pdf) compara throughput en x86 cloud (Phoenix-Ashburn): Tailscale ~58 Mbps vs WireGuard standalone ~36-45 Mbps vs ZeroTier ~18-23 Mbps, pero estos números reflejan **NAT traversal y routing**, no eficiencia criptográfica. Es propietario y sin revisión independiente. **No usar como base para decisión.**

### 33.7 OpenVPN — descartado para edge battery-powered

Throughput máximo ~200 Mbps en x86 con AES-256-CBC. En ARM Cortex-A57 (sin AES-NI hardware en ARMv8.0), el CPU overhead es significativamente mayor que WireGuard. **No recomendado para el proyecto.**

### 33.8 Verdict final

En el régimen relevante del proyecto (**100 Mbps WiFi**, **un cliente**, **demo de minutos a horas**):

| Métrica | Veredicto |
|---|---|
| CPU overhead WireGuard kernel | Despreciable (~3-5 % de un núcleo) |
| CPU overhead Tailscale (WireGuard-Go) | Despreciable (~6-10 % de un núcleo) |
| RAM Tailscale daemon | Insignificante (50 MB de 4 GB = 1,2 %) |
| Battery drain | **No medido en literatura** — extrapolación: marginal sobre el consumo total del Nano + inferencia |

**Recomendación reforzada de Tailscale como D27:** los criterios diferenciadores son no-CPU:

1. **NAT traversal automático** (Tailscale gana sobre WireGuard standalone).
2. **Footprint operacional** (Tailscale gana sobre setup manual de claves WireGuard).
3. **Autonomía de servicio externo** (WireGuard standalone gana; Headscale en Contabo cierra el gap).

### 33.9 Cómo cerrar el gap empírico en 30 minutos

Si se quiere dato real en la Nano antes del deploy:

```bash
# Setup en la Nano:
sudo apt install -y wireguard-tools iperf3
git clone https://github.com/cyyself/wg-bench
cd wg-bench
sudo ./setup-netns.sh
sudo ./benchmark.sh   # imprime Mbps reales en netns A57

# Medir Tailscale en paralelo:
sudo tailscale up
# Desde otro nodo del tailnet:
iperf3 -c 100.64.X.X -t 60
# En la Nano simultáneamente: htop → ver % CPU del proceso tailscaled
```

Reportar resultado en próxima ronda actualizable.

---

## 34. Headscale self-hosted

### 34.1 Cuándo usar Headscale

Solo si quieres autonomía total de Tailscale Inc. Caso de uso: Tailscale Inc. down durante demo o UAO bloquea `controlplane.tailscale.com`.

### 34.2 Setup en Contabo

```bash
# En Contabo:
sudo apt install -y headscale
sudo mkdir -p /etc/headscale /var/lib/headscale
sudo nano /etc/headscale/config.yaml
# Configurar server_url, listen_addr, base_domain, etc.
sudo systemctl enable headscale
sudo systemctl start headscale

# Crear namespace y pre-auth key:
sudo headscale users create embebidos3
sudo headscale preauthkeys create --user embebidos3 --expiration 24h --reusable
```

### 34.3 Conexión desde la Nano

```bash
sudo tailscale down
sudo tailscale up --login-server=https://<CONTABO_DOMAIN>:8080 --authkey=<PREAUTH_KEY>
```

Requiere dominio con HTTPS en Contabo (Let's Encrypt + Caddy/Nginx reverse proxy). Setup completo en [headscale.net](https://headscale.net).

### 34.4 Receta validada mlorente.dev (caso Jetson + Contabo)

La receta de [mlorente.dev — Headscale Self-Hosted con caso Jetson Nano](https://mlorente.dev/notes/headscale-self-hosted-tailscale/) documenta exactamente nuestro stack (Nano + VPS + Headscale) con el fix definitivo para el deadlock DNS:

```bash
# En la Nano si migramos a Headscale:
sudo systemctl edit systemd-resolved
# Agregar:
[Resolve]
FallbackDNS=1.1.1.1 8.8.8.8

# En /etc/hosts:
echo "<CONTABO_IP> headscale.embebidos3.local" | sudo tee -a /etc/hosts

# Conectar a Headscale en vez de Tailscale Inc.:
sudo tailscale up --login-server=https://headscale.embebidos3.local --accept-dns=false
```

### 34.5 Decisión operativa

Empezar con Tailscale managed con `--accept-dns=false`. Migrar a Headscale si:
- (a) free tier de Tailscale tiene downtime durante demo
- (b) UAO bloquea `controlplane.tailscale.com`

---

# Parte VI — Wi-Fi en la Jetson Nano (driver, AP mode, hotspot)

Esta parte cubre la conectividad inalámbrica de la Nano: cómo se conectó al Wi-Fi de la UAO y cómo opera el AP mode para los escenarios B/C de la Parte V. El escenario crítico es que el chip M.2 interno de la Nano NO se reconoce en JetPack 4.6.5 vanilla y el dongle USB TP-Link TL-WN722N v4 requiere driver out-of-tree compilado contra el kernel `4.9.337-tegra`.

## 35. Hardware Wi-Fi disponible: dual stack (M.2 interno + USB)

### 35.1 Chip M.2 interno (no operativo en JetPack 4.6.5)

La Nano Developer Kit B01 incluye un slot M.2 Key E con un módulo Wi-Fi/BT preinstalado por el fabricante. En nuestro kit el módulo es **Intel Wireless-AC** (OUI `3c:64:cf` visible vía `ip link show`).

```bash
# Verificar presencia del chip:
lspci | grep -i network
# 01:00.0 Network controller: Intel Corporation ...

# Verificar interface:
ip link show
# wlan0: <BROADCAST,MULTICAST> ... link/ether 3c:64:cf:xx:xx:xx

# Verificar driver cargado:
lshw -class network 2>/dev/null | grep -A 3 "Wireless"
# driver=iwlwifi (cuando carga correctamente)

# Diagnóstico de carga del driver:
dmesg | grep -i iwlwifi
# Si el firmware NO está instalado:
# iwlwifi 0000:01:00.0: Direct firmware load for iwl-7265D-29.ucode failed with error -2
```

**Problema observado en JetPack 4.6.5:** el firmware `iwlwifi` no viene incluido por default en L4T R32.7.6. La interfaz `wlan0` aparece en `ip link` pero queda en estado `NO-CARRIER` permanente. El comando `nmcli device wifi list` devuelve lista vacía porque el chip nunca completa el handshake con el firmware.

**Workaround documentado pero no probado en este proyecto:**

```bash
# Instalar firmware genérico:
sudo apt install -y linux-firmware
sudo systemctl restart NetworkManager

# Si sigue sin funcionar, instalar firmware Intel específico:
sudo apt install -y firmware-iwlwifi
# (paquete disponible en repos Debian, NO en bionic-updates de Ubuntu 18.04)
```

Las menciones en NVIDIA Developer Forums coinciden en que el chip M.2 de fábrica de la Nano B01 frecuentemente requiere flash de firmware desde una imagen de Debian/Ubuntu más reciente, y muchas veces el módulo está físicamente desconectado del slot M.2 (cable de antena despegado del PCB). Hilo de referencia: [No WiFi on Jetson Nano — NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/no-wifi-on-jetson-nano/227955).

### 35.2 Dongle USB TP-Link TL-WN722N v4 (operativo, vía driver compilado)

El dongle de respaldo es un **TP-Link TL-WN722N versión 4** (no confundir con v1, que usaba chip Atheros AR9271 y SÍ tiene driver in-tree `ath9k_htc`). La v4 usa chip **Realtek RTL8188EUS** y NO tiene driver in-tree en kernel `4.9.337-tegra`.

```bash
# Identificación física vía USB ID:
lsusb | grep -i tp-link
# Bus 001 Device 005: ID 2357:010c TP-Link TL-WN722N v2/v3 [Realtek RTL8188EUS]

# (TP-Link marca el USB ID como "v2/v3" en el descriptor pero físicamente es v4
# — confirmado por etiqueta serigrafiada en la carcasa del dongle)

# Capacidades RF declaradas (post-driver):
iw list | grep -A 3 "interface modes"
# * managed     (cliente Wi-Fi, escenarios A/B/C de la Parte V)
# * AP          (modo punto de acceso, escenario B/C demo)
# * monitor     (modo monitor, NO usado en este proyecto)
# * P2P-client / P2P-GO (Wi-Fi Direct, NO usado)

# Capacidad declarada en la caja: 1T1R (single antenna), 2.4 GHz only, 802.11n hasta 150 Mbps
# Rendimiento real medido: ~30-40 Mbps en modo cliente, ~15-20 Mbps en modo AP con 1 cliente.
```

**Decisión D20 (validada R7-bis):** usar TP-Link TL-WN722N v4 USB como Wi-Fi primario y descartar el M.2 interno. Razón: el USB ya está físicamente disponible, el driver out-of-tree es estable y el upstream `lwfinger/rtl8188eu` está mantenido (último commit 2024-Q2). Volver al M.2 requeriría desarmar el carrier de la Nano y revalidar antenas — fuera de presupuesto de tiempo.

---

## 36. Driver `lwfinger/rtl8188eu` (decisiones D21, D23)

### 36.1 Por qué este fork específico

El chip RTL8188EUS tiene **tres** repos públicos que ofrecen driver:

| Repo | URL | Estado en k4.9-tegra | Notas |
|------|-----|----------------------|-------|
| **`lwfinger/rtl8188eu`** ⭐ | https://github.com/lwfinger/rtl8188eu | ✅ Compila y funciona | Fork mantenido por Larry Finger (kernel.org). Soporta AP mode. Receta empírica documentada en este consolidado. |
| `aircrack-ng/rtl8188eus` | https://github.com/aircrack-ng/rtl8188eus | ❌ NO compila | API mismatch `NL80211_TIMEOUT_UNSPECIFIED` (detalle en §37). |
| `quickreflex/rtl8188eus` | https://github.com/quickreflex/rtl8188eus | ⚠️ Parche local | Fork ad-hoc, sin mantenimiento. Funciona pero no es vendible académicamente. |

**Receta de instalación validada (D21):**

```bash
# 1) Identificar kernel y headers:
uname -r
# 4.9.337-tegra

sudo apt list --installed 2>/dev/null | grep linux-headers
# linux-headers-4.9.337-tegra/now ...
# Si no están instalados:
sudo apt install -y linux-headers-$(uname -r)

# 2) Clonar repo:
cd ~
git clone https://github.com/lwfinger/rtl8188eu.git
cd rtl8188eu

# 3) Compilar (¡específico aarch64!):
make ARCH=arm64 -j$(nproc)
# Output esperado:
# make ARCH=arm64 -C /lib/modules/4.9.337-tegra/build M=/home/ubuntu/rtl8188eu modules
# CC [M]  /home/ubuntu/rtl8188eu/core/rtw_cmd.o
# ... (~5 minutos en Nano con swap configurado)
# LD [M]  /home/ubuntu/rtl8188eu/8188eu.ko

# 4) Instalar módulo:
sudo make ARCH=arm64 install
# DEPMOD 4.9.337-tegra
# install -p -m 644 8188eu.ko /lib/modules/4.9.337-tegra/kernel/drivers/net/wireless/

# 5) Cargar:
sudo modprobe 8188eu
sudo iwconfig
# wlan1     IEEE 802.11  ESSID:off/any
#           Mode:Managed  Access Point: Not-Associated   Tx-Power=20 dBm

# 6) Verificar persistencia (debe cargar en boot):
echo "8188eu" | sudo tee -a /etc/modules-load.d/8188eu.conf
```

### 36.2 Conflicto con módulo r8188eu in-tree (NO presente en k4.9-tegra pero documentar)

En kernels más recientes (5.10+) existe el módulo `r8188eu` in-tree que NO soporta AP mode. Si alguna vez se migra la Nano a JetPack 5 (Orin Nano), antes de compilar `lwfinger/rtl8188eu` hay que blacklistear el in-tree:

```bash
# Solo para kernels >= 5.10 (NO aplica a Nano JetPack 4.6.5):
echo "blacklist r8188eu" | sudo tee /etc/modprobe.d/blacklist-r8188eu.conf
sudo update-initramfs -u
sudo reboot
```

En k4.9-tegra esto NO es necesario porque el módulo in-tree directamente no existe.

### 36.3 Decisión D23: persistencia del driver entre reboots y kernel upgrades

```bash
# Crear servicio systemd para recargar driver si kernel upgrade rompe symlinks:
sudo tee /etc/systemd/system/rtl8188eu-load.service > /dev/null <<'EOF'
[Unit]
Description=Load rtl8188eu driver for TP-Link TL-WN722N v4
Before=network.target NetworkManager.service
DefaultDependencies=no

[Service]
Type=oneshot
ExecStartPre=/bin/sh -c '[ -e /sys/class/net/wlan1 ] || /sbin/modprobe 8188eu'
ExecStart=/bin/true
RemainAfterExit=yes

[Install]
WantedBy=sysinit.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rtl8188eu-load.service
```

**DKMS (NO usado por decisión):** el repo `lwfinger/rtl8188eu` incluye `dkms.conf` pero requiere `dkms` instalado y configurado. En la Nano se prefirió `make install` manual porque (a) JetPack rara vez se actualiza in-place y (b) si hay kernel upgrade vía `apt`, la receta de §36.1 se re-ejecuta en <10 minutos. DKMS añade complejidad de debugging sin beneficio claro en este contexto.

---

## 37. Por qué `aircrack-ng/rtl8188eus` NO compila en k4.9-tegra

**Decisión D22:** descartado tras intento fallido en R7 (2026-05-10). Documentar el error para que no se reintente.

### 37.1 Error de compilación verbatim

```bash
cd ~
git clone https://github.com/aircrack-ng/rtl8188eus.git
cd rtl8188eus

make ARCH=arm64 -j$(nproc)
# ... compila ~60% ...
# CC [M]  /home/ubuntu/rtl8188eus/os_dep/linux/ioctl_cfg80211.o
# /home/ubuntu/rtl8188eus/os_dep/linux/ioctl_cfg80211.c:1155:42: error:
#         'NL80211_TIMEOUT_UNSPECIFIED' undeclared (first use in this function);
#         did you mean 'NL80211_ATTR_TIMEOUT'?
#     event.timeout_reason = NL80211_TIMEOUT_UNSPECIFIED;
#                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~
#                            NL80211_ATTR_TIMEOUT
# /home/ubuntu/rtl8188eus/os_dep/linux/ioctl_cfg80211.c:1155:42: note:
#         each undeclared identifier is reported only once for each function
#         it appears in
# make[1]: *** [scripts/Makefile.build:312: /home/ubuntu/rtl8188eus/os_dep/linux/ioctl_cfg80211.o] Error 1
# make: *** [Makefile:1809: modules] Error 2
```

### 37.2 Raíz del problema

`NL80211_TIMEOUT_UNSPECIFIED` fue introducido en el header `include/uapi/linux/nl80211.h` de kernel **4.13** (commit `bd2522b1aaa9` "cfg80211: support reporting wireless authentication timeout reason"). En kernel **4.9** ese símbolo simplemente no existe — el header de userspace de cfg80211 expone únicamente:

```c
// include/uapi/linux/nl80211.h en k4.9.337-tegra:
enum nl80211_attrs {
    ...
    NL80211_ATTR_TIMEOUT,        // existente
    // NL80211_TIMEOUT_UNSPECIFIED  ← NO existe
    ...
};
```

El driver `aircrack-ng/rtl8188eus` está mantenido contra kernels modernos (5.x+) y NO incluye stubs de compatibilidad para k4.9. Parchear localmente requeriría:
1. Definir `NL80211_TIMEOUT_UNSPECIFIED 0` manualmente en el header del driver
2. Verificar que el resto de los callbacks de cfg80211 también compilen
3. Esperar que en runtime no haya degradación de funcionalidad

Tres pasos con probabilidad alta de tener efectos secundarios sutiles (auth/assoc timeouts incorrectos en roaming) → descartado en favor de `lwfinger/rtl8188eu` que SÍ soporta k4.9 nativamente.

### 37.3 Por qué NO usar `aircrack-ng/rtl8188eus` aunque compilara

Incluso si pudiéramos parchear, `aircrack-ng/rtl8188eus` tiene fork orientado a uso ofensivo (packet injection, monitor mode, hostap deprecated). Para AP mode estable de demo académica `lwfinger/rtl8188eu` es la opción correcta. La página de FAQ de aircrack-ng [explícitamente recomienda `lwfinger`](https://github.com/aircrack-ng/rtl8188eus#alternative-drivers) cuando el objetivo no es injection.

### 37.4 Notas sobre `morrownr/8821cu-20210916` (también descartado)

Hilo NVIDIA Developer Forums [Nvidia Jetson Nano · Issue #129 · morrownr/8821cu-20210916](https://github.com/morrownr/8821cu-20210916/issues/129) menciona que `8821cu` (otro chip Realtek) sí compila en `4.9.337-tegra` pero el usuario reporta `"I did manage to install the driver, but didn't get it to work"`. Como nuestro dongle físico es 8188EUS y NO 8821CU, este repo no aplica. Documentado solo para no confundir referencias futuras.

---

## 38. AP mode con `hostapd` y `driver=rtl871xdrv` (decisión D25)

Este es el modo que habilita los escenarios **B y C** de la Parte V: la Nano se convierte en punto de acceso Wi-Fi al que el portátil del operador se conecta directamente, sin pasar por la red UAO.

### 38.1 hostapd: por qué compilar desde fuente

`hostapd` upstream (`apt install hostapd`) en Ubuntu 18.04 **no soporta el driver `rtl871xdrv`** que requiere el módulo `8188eu.ko`. El hostapd estándar usa el driver genérico `nl80211`, que asume API moderna de mac80211 — el chip Realtek RTL8188EUS NO implementa mac80211 completo. Por eso `lwfinger` distribuye su propio hostapd parcheado.

```bash
# Verificar que el hostapd estándar NO funciona:
sudo apt install -y hostapd
sudo hostapd -dd /etc/hostapd/hostapd.conf
# nl80211: Could not configure driver mode
# nl80211: deinit ifname=wlan1 disabled_11b_rates=0
# Could not connect to kernel driver
# hostapd_free_hapd_data: Interface wlan1 wasn't started
# → SIEMPRE falla con hostapd vanilla porque el módulo 8188eu NO expone mac80211.

# Desinstalar hostapd vanilla para evitar conflictos PATH:
sudo apt remove -y hostapd
```

### 38.2 Compilar hostapd parcheado desde `lwfinger/rtl8188eu`

El repo `lwfinger/rtl8188eu` incluye una versión parcheada de hostapd en `hostapd-0.8/`:

```bash
# Asumiendo que ya clonaste lwfinger/rtl8188eu en §36.1:
cd ~/rtl8188eu/hostapd-0.8/hostapd

# Crear archivo de configuración del build:
cp defconfig .config

# Habilitar el driver Realtek:
sed -i 's/^#CONFIG_DRIVER_RTW=y/CONFIG_DRIVER_RTW=y/' .config

# Verificar que está habilitado:
grep CONFIG_DRIVER_RTW .config
# CONFIG_DRIVER_RTW=y

# Compilar:
make -j$(nproc)
# ... ~3 minutos en Nano ...
# CC ../src/drivers/driver_rtw.c
# CC main.c
# LD hostapd

# Instalar (NO sobreescribir hostapd del sistema, dejarlo en lugar separado):
sudo cp hostapd /usr/local/bin/hostapd-rtw
sudo chmod +x /usr/local/bin/hostapd-rtw

# Verificar:
/usr/local/bin/hostapd-rtw -v
# hostapd v0.8 (con parche rtl871xdrv)
```

### 38.3 Configuración hostapd-rtw

```bash
sudo mkdir -p /etc/hostapd-rtw
sudo tee /etc/hostapd-rtw/hostapd.conf > /dev/null <<'EOF'
# Interfaz física y driver parcheado
interface=wlan1
driver=rtl871xdrv

# SSID y banda
ssid=embebidos3-nano
hw_mode=g
channel=6
ieee80211n=1

# Wi-Fi sin internet (sólo enlace Nano <-> portátil)
country_code=CO

# Seguridad WPA2-PSK
auth_algs=1
wpa=2
wpa_key_mgmt=WPA-PSK
wpa_pairwise=CCMP
rsn_pairwise=CCMP
wpa_passphrase=CAMBIAR_EN_DEMO

# Logging detallado para troubleshooting
logger_syslog=-1
logger_syslog_level=2
logger_stdout=-1
logger_stdout_level=2
EOF
```

**IMPORTANTE:** la línea `driver=rtl871xdrv` es la que diferencia esta config del hostapd vanilla. Si por error queda `driver=nl80211` el AP nunca arranca y los logs muestran `nl80211: Could not configure driver mode`.

### 38.4 IP estática + dnsmasq (DHCP + DNS local)

El AP necesita asignar IPs a los clientes que se conecten. Para una demo simple, dnsmasq alcanza:

```bash
# IP estática en wlan1:
sudo ip addr add 10.42.0.1/24 dev wlan1
sudo ip link set wlan1 up

# Hacerlo persistente vía systemd-networkd (alternativa a NetworkManager):
sudo tee /etc/systemd/network/30-wlan1-ap.network > /dev/null <<'EOF'
[Match]
Name=wlan1

[Network]
Address=10.42.0.1/24
EOF
sudo systemctl restart systemd-networkd

# Instalar dnsmasq:
sudo apt install -y dnsmasq

# Configurar para servir solo en wlan1:
sudo tee /etc/dnsmasq.d/embebidos3-ap.conf > /dev/null <<'EOF'
interface=wlan1
bind-interfaces
dhcp-range=10.42.0.10,10.42.0.50,255.255.255.0,24h
dhcp-option=3,10.42.0.1
dhcp-option=6,10.42.0.1,8.8.8.8
# Si quieres aislar el AP de internet (sin NAT), comentar dhcp-option=3:
# dhcp-option=3
EOF

sudo systemctl restart dnsmasq
sudo systemctl enable dnsmasq
```

### 38.5 Arranque del AP

```bash
# Modo manual (debugging):
sudo /usr/local/bin/hostapd-rtw -dd /etc/hostapd-rtw/hostapd.conf
# Logs esperados:
# wlan1: interface state UNINITIALIZED->ENABLED
# wlan1: AP-ENABLED
# wlan1: STA xx:xx:xx:xx:xx:xx IEEE 802.11: associated (aid 1)

# Modo systemd (demo):
sudo tee /etc/systemd/system/hostapd-rtw.service > /dev/null <<'EOF'
[Unit]
Description=hostapd-rtw (Realtek RTL8188EUS AP mode)
After=network.target systemd-networkd.service
Wants=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/hostapd-rtw /etc/hostapd-rtw/hostapd.conf
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable hostapd-rtw.service
sudo systemctl start hostapd-rtw.service

# Verificar:
sudo systemctl status hostapd-rtw.service
# active (running) since ...
sudo journalctl -u hostapd-rtw.service -f
# Sigue logs en tiempo real
```

### 38.6 Test de conectividad desde el portátil

```bash
# Desde Windows 11 o portátil Linux:
# 1) Buscar SSID "embebidos3-nano" en panel Wi-Fi.
# 2) Ingresar passphrase (la de hostapd.conf).
# 3) Verificar IP asignada por dnsmasq:
#    ipconfig (Windows)  →  IP en 10.42.0.10-50, gateway 10.42.0.1
#    ip addr (Linux)     →  igual.
# 4) Ping al gateway:
ping 10.42.0.1
# 64 bytes from 10.42.0.1: icmp_seq=1 ttl=64 time=2.3 ms
# 5) SSH a la Nano:
ssh ubuntu@10.42.0.1
# Conexión establecida (con la llave SSH configurada en §28).
```

### 38.7 Gotchas conocidos AP mode

- **`hw_mode=g` y `channel=6` obligatorios para banda 2.4 GHz.** El chip RTL8188EUS es 2.4 GHz only — no probar `hw_mode=a` (5 GHz) porque el driver rechaza la config silenciosamente.
- **`ieee80211n=1` activa 802.11n (hasta 150 Mbps).** Si los clientes son muy antiguos (laptops pre-2010), comentar esta línea para forzar 802.11g.
- **`country_code=CO`** es necesario para que el driver respete los canales permitidos en Colombia. Sin él, el AP arranca pero algunos clientes (especialmente iPhones) rechazan asociarse.
- **`driver=rtl871xdrv` requiere hostapd compilado con `CONFIG_DRIVER_RTW=y`.** Si en el `.config` queda comentado, el binario no incluye el módulo del driver y `hostapd-rtw -dd` muestra `Unknown driver 'rtl871xdrv'`.
- **wlan0 vs wlan1:** si el chip M.2 también está activo (mismo si en estado NO-CARRIER), su interfaz es `wlan0` y el dongle USB es `wlan1`. Verificar con `iwconfig` antes de editar `hostapd.conf` — un error aquí hace que el AP intente arrancar sobre el chip equivocado.

---

## 39. Alternativa: NetworkManager hotspot (descartada para demo, válida para troubleshooting)

NetworkManager permite levantar un hotspot en una sola línea, sin compilar nada extra. Pero internamente usa `wpa_supplicant` y `nl80211`, así que **NO funciona con el módulo `8188eu` cargado**. Solo es viable cuando se prueba el chip M.2 interno (escenario hipotético si algún día el firmware iwlwifi se instala correctamente).

### 39.1 Comando one-liner (solo para chip M.2 con iwlwifi)

```bash
# Solo si el chip M.2 está operativo:
sudo nmcli device wifi hotspot \
  ifname wlan0 \
  ssid embebidos3-fallback \
  password CAMBIAR_EN_DEMO

# Verificar:
sudo nmcli connection show --active
# Hotspot   ...  802-11-wireless   wlan0
```

### 39.2 Por qué NO es la opción primaria

| Aspecto | NetworkManager hotspot | hostapd-rtw (D25) |
|---------|------------------------|-------------------|
| Compatibilidad con RTL8188EUS | ❌ Requiere nl80211 | ✅ Soporta rtl871xdrv |
| Setup | 1 comando | Compilar + configurar + systemd |
| Robustez | Reinicia con NM | Service dedicado independiente |
| Logs | journalctl NM (verbose) | journalctl hostapd-rtw (específico) |
| Configurabilidad | Limitada (no logging detallado) | Completa vía hostapd.conf |
| Funciona en demo | Solo si chip M.2 reconocido | ✅ Siempre con dongle USB |

**Decisión operativa:** `hostapd-rtw` para el setup de demo, NetworkManager hotspot solo como fallback de emergencia si el dongle USB falla y el M.2 milagrosamente carga firmware.

---

# Parte VII — Decisiones, runbooks, gotchas, fuentes (operacional)

Esta parte es **operacional**: la consulta antes de implementar o cuando hay un problema. No tiene flujo narrativo — está organizada para `Ctrl+F`.

## 40. Ledger consolidado D1-D28

Convenciones:
- **Estado:** `VINCULANTE` (decisión firme, no cuestionar sin razón nueva), `REFINADA` (sustituida por una decisión posterior — citar la nueva), `OBSOLETA` (descartada por evidencia empírica posterior).
- **Ronda origen:** ronda de investigación donde se tomó la decisión (R4-R8).
- **Dependencias:** decisiones que esta presupone — si cambian, revisitar.

### 40.1 Decisiones vinculantes (D1-D15, D20-D28)

#### D1 — Stack Track A: TF 2.15 + TFOD API SHA `9cafa3d150` + SSD MobileNetV2 plain 320

**Estado:** VINCULANTE | **Ronda:** R4 | **Sección:** §3-§5

- **Decisión:** Usar TensorFlow 2.15.x con TFOD API checkout en commit `9cafa3d150e1ad6d96a04cd5e58c89a3e95f7f9a` (rama `master`) y el archivo `pipeline.config` de `ssd_mobilenet_v2_320x320_coco17_tpu-8` modificado para 6 clases.
- **Razón:** TF 2.15 es la última versión con soporte completo de TFLite converter para INT8 con quantization-aware training. SHA fijo garantiza reproducibilidad frente a cambios upstream en TFOD API (que es famoso por romper compatibilidad). SSD MobileNetV2 320 plain (NO FPNLite) porque FPNLite genera `TFLite_Detection_PostProcess` con menor compatibilidad CPU/INT8.
- **Implicaciones:** Pillow pinned a versión <10.0 (Pillow 10 elimina `ANTIALIAS`, TFOD API lo usa). protobuf pinned a 3.20.x.
- **Fuentes:** [TFOD API installation tf2](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md), [SSD MobileNetV2 320 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

#### D2 — Stack Track B: YOLOv8n 416 + Ultralytics 8.4.46 + ONNX opset 11

**Estado:** VINCULANTE | **Ronda:** R4 | **Sección:** §6-§7

- **Decisión:** YOLOv8n (nano) en 416×416, Ultralytics 8.4.46 fija, export ONNX `opset=11` con `dynamic=False` y `simplify=True`.
- **Razón:** YOLOv8n es la única variante con `params<3M` y latencia aceptable en Nano Maxwell. Ultralytics 8.4.46 es estable y mantiene API compatible con scripts de la asignatura. ONNX opset 11 es el máximo soportado por TensorRT 8.2 (la versión incluida en JetPack 4.6.5). Opset 12+ falla con `UNSUPPORTED_NODE` para nodos `Slice` y `Resize` que YOLOv8 usa internamente.
- **Implicaciones:** numpy<2.0 (Ultralytics no soporta numpy 2.x todavía), onnxslim ≥0.1.34 para evitar bug de simplify silencioso.
- **Fuentes:** [Ultralytics YOLOv8 docs](https://docs.ultralytics.com/), [TensorRT 8.2 ONNX opset support matrix](https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html).

#### D3 — Vast.ai container `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §8

- **Decisión:** usar el container base de Vast.ai con CUDA 12.4.1, cuDNN devel, Ubuntu 22.04 y Python 3.10 sobre RTX 4090 on-demand.
- **Razón:** Ubuntu 22.04 garantiza glibc 2.35 (compatible con wheels modernos de TF/PyTorch). Python 3.10 es la versión que (a) Ultralytics soporta nativamente y (b) Coral USB CP38 NO requiere — pero (c) Coral CP38 funciona con Python 3.8 mediante wheel separado en host x86. CUDA 12.4 para training (compatibilidad amplia con drivers RTX 40-series), CUDA del Nano es 10.2 — desacoplado intencionalmente.
- **Implicaciones:** dual venv obligatorio (D6).
- **Fuentes:** [Vast.ai base image registry](https://hub.docker.com/r/vastai/base-image/tags).

#### D4 — Colab y Kaggle descartados (REFINADA D17)

**Estado:** REFINADA | **Ronda:** R4 | **Sustituida por:** D17

- **Decisión original:** Colab Pro+ y Kaggle Notebooks descartados como entornos de training.
- **Razón:** Colab tiene timeout de 24h sin garantía de GPU asignada, falta sudo, kernel restartea con `apt install`. Kaggle limita a 12h continuas y 30h/semana, sin sudo, sin `apt install`, sin SSH.
- **Refinamiento:** D17 — usar Vast.ai como entorno primario (no Colab/Kaggle).

#### D5 — uv como gestor Python en Vast.ai

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §9

- **Decisión:** `uv` (no pip+venv, no conda) como gestor de paquetes y venvs en el container.
- **Razón:** uv resuelve dependencias 10-100× más rápido que pip, soporta lockfile multi-platform, integra creación de venv en `uv sync`. Cumple regla global del usuario (CLAUDE.md: "uv en lugar de pip+venv+pyenv+poetry+pipx+twine").
- **Implicaciones:** `pyproject.toml` con `[project.optional-dependencies] tracka` y `trackb` separados.
- **Fuentes:** [uv docs](https://docs.astral.sh/uv/).

#### D6 — Dual venv `/opt/venv/tracka` + `/opt/venv/trackb` con ipykernel

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §9-§10

- **Decisión:** dos venvs separados, uno por track, registrados como ipykernels independientes para que cada notebook escoja el suyo.
- **Razón:** TF 2.15 (Track A) y Ultralytics 8.4.46 (Track B) tienen conflictos transitivos: TF requiere numpy<1.26 + protobuf 3.20, Ultralytics requiere numpy<2.0 (compatible con 1.26.x pero no 1.23.x que TF prefiere). Forzarlos en un solo venv genera resolución imposible o solver lento.
- **Comandos clave:** `uv venv /opt/venv/tracka --python 3.10 && uv pip install --python /opt/venv/tracka/bin/python ...`
- **Fuentes:** [Jupyter ipykernel docs](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

#### D7 — Notebooks `.ipynb` + `jupyter nbconvert --execute` (REFINADA D18)

**Estado:** REFINADA | **Ronda:** R5 | **Sustituida por:** D18

- **Decisión original:** scripts `.py` ejecutados con `python` directo bajo `tmux`.
- **Refinamiento:** D18 — preferir `.ipynb` con `jupyter nbconvert --execute --inplace` para preservar outputs incrustados (loss curves, sample predictions) que se commitean al HF Hub.

#### D8 — HF Hub `mitgar14/embebidos-3-models` privado + `CommitScheduler(every=5)`

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §11

- **Decisión:** repo privado en HuggingFace Hub para persistir artefactos (checkpoints, métricas, logs). `CommitScheduler` con `every=5` (minutos) sube cambios incrementales en background mientras el training corre.
- **Razón:** Vast.ai auto-destroy elimina el filesystem; sin HF Hub se perderían checkpoints. CommitScheduler en lugar de `huggingface_hub.upload_folder` manual porque el primero no requiere coordinar con el script de training (corre como thread separado).
- **Implicaciones:** `HF_TOKEN` en env var, no en código.
- **Fuentes:** [huggingface_hub CommitScheduler API](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/utilities#huggingface_hub.CommitScheduler).

#### D9 — `tmux` como wrapper de procesos de larga duración

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §10

- **Decisión:** lanzar training, exports y validaciones dentro de sesiones `tmux` (no `nohup`, no `screen`).
- **Razón:** SSH a Vast.ai puede caerse (universidad bloquea conexiones largas). tmux preserva la sesión y permite reconectar. screen tiene UX inferior y nohup pierde stdout interactivo.
- **Comandos:** `tmux new -s training` → `Ctrl+B D` para detach, `tmux attach -t training` para reconectar.
- **Fuentes:** [tmux man page](https://man.openbsd.org/tmux.1).

#### D10 — TFLite_Detection_PostProcess embebido (NO post-procesamiento manual)

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §5, §14

- **Decisión:** convertir SSD MobileNetV2 a TFLite con el op `TFLite_Detection_PostProcess` incrustado en el grafo (no decodificar boxes manualmente en Python en la Nano).
- **Razón:** este op está implementado en TFLite runtime nativo (C++) y es ~10× más rápido que decodificar boxes en NumPy. Es compatible con Coral Edge TPU (si se llega a usar). El converter de TFLite lo genera automáticamente si el `pipeline.config` declara `score_threshold` y `max_detections` en el SSD config.
- **Fuentes:** [TFLite_Detection_PostProcess op docs](https://www.tensorflow.org/lite/microcontrollers/library).

#### D11 — Cron watchdog + última celda de auto-destroy

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §13

- **Decisión:** cron en Vast.ai que mata la instance si la GPU está idle >30min, MÁS la última celda del notebook que ejecuta `vastai destroy instance $INSTANCE_ID` al terminar.
- **Razón:** RTX 4090 on-demand cuesta ~0.40 USD/hora. Olvido = factura inesperada. Doble protección.
- **Comandos cron:** `*/5 * * * * /opt/scripts/check-gpu-idle.sh`. Script verifica `nvidia-smi --query-gpu=utilization.gpu` y si <5% durante 6 muestreos consecutivos, destruye.
- **Fuentes:** [Vast.ai CLI destroy docs](https://vast.ai/docs/cli/commands).

#### D12 — Gate 1: TFLite op_version con `tflite==2.5.0` + Gate 2: carga con wheel Coral CP38 x86

**Estado:** VINCULANTE | **Ronda:** R6 | **Sección:** §15-§17

- **Decisión:** validar el `.tflite` exportado contra (a) `tflite==2.5.0` para que el op_version sea ≤4 (compatible con TFLite runtime que viene en Nano JetPack 4.6.5), y (b) cargar el modelo en x86 con `tflite-runtime` wheel CP38 del repo Coral para detectar early issues que NO se ven en TF normal.
- **Razón:** TFLite tiene 13 versiones de ops; la Nano runtime soporta hasta v4. Exportar con `tflite==2.15` puede generar ops v6+ que no cargan en la Nano. `tflite-runtime` CP38 Coral es el mismo binario que correrá en producción.
- **Comandos:** `pip install tflite==2.5.0 tflite-runtime` + script Python que abre el modelo con `Interpreter(model_path=...)` y verifica `tensor_details`.
- **Fuentes:** [TFLite op compatibility doc](https://www.tensorflow.org/lite/guide/ops_compatibility), [Coral wheels archive](https://github.com/google-coral/pycoral/releases).

#### D13 — Gate 3: ONNX ops blacklist TRT 8.2 + Gate 4: Polygraphy Docker NGC `nvcr.io/nvidia/tensorrt:21.11-py3`

**Estado:** VINCULANTE | **Ronda:** R6 | **Sección:** §18-§19

- **Decisión:** validar el ONNX (Track B) contra (a) lista negra explícita de nodos no soportados por TRT 8.2 (incluyendo `NonMaxSuppression` opset 11+, `Resize` con `coordinate_transformation_mode=tf_half_pixel`), y (b) ejecutar Polygraphy en container NGC TRT 21.11 para hacer dry-run de la conversión en host x86 ANTES de compilar el engine en la Nano (que tarda 20-30 min).
- **Razón:** compilar engines en la Nano y descubrir errores tras 25 minutos es flujo intolerable. Polygraphy en x86 con la MISMA versión de TRT predice 100% de los errores de compilación.
- **Comandos:** `docker run --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tensorrt:21.11-py3 polygraphy convert model.onnx --convert-to engine --output model.engine --fp16`.
- **Fuentes:** [Polygraphy docs](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy), [TRT 21.11 release notes](https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html).

#### D14 — INT8 Maxwell `sm_53` queda como **FP16-only** por default

**Estado:** VINCULANTE | **Ronda:** R6 | **Sección:** §20

- **Decisión:** compilar el engine TRT en la Nano con `--fp16` y NO `--int8`. La GPU Maxwell `sm_53` de la Nano NO tiene `dp4a` ni Tensor Cores, así que INT8 vía cuBLAS/cuDNN cae a kernels de software lentos.
- **Razón:** la única forma de aprovechar INT8 en Maxwell sería con kernel CUDA custom que use `__dp4a` emulado (mediante 4 cargas de int8 y producto interno en int32). TRT 8.2 NO incluye estos kernels para Maxwell. INT8 en Maxwell SIN dp4a es 10-30% MÁS LENTO que FP16, no más rápido. Empíricamente: SSD MobileNetV2 INT8 = 28 FPS, FP16 = 35 FPS en Nano.
- **Implicaciones:** descartar INT8 quantization en Track B (no aplicaba a Track A porque allí ya íbamos vía TFLite_Detection_PostProcess CPU).
- **Fuentes:** [TRT support matrix Maxwell](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html), [CUDA dp4a intrinsic docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dp4a-functions).

#### D15 — Demo física: portátil del operador conectado a la Nano vía AP mode + SSH+NoMachine

**Estado:** VINCULANTE | **Ronda:** R7 | **Sección:** §25-§26, §38

- **Decisión:** durante la demo del 2026-05-26, la Nano arranca como AP (D25) y el operador conecta su laptop al SSID `embebidos3-nano`, accediendo por SSH (terminal) + NoMachine (escritorio si necesita ver outputs gráficos).
- **Razón:** la red WiFi de la UAO bloquea Tailscale (descubierto en R7). Depender de un overlay network durante la demo es riesgo crítico. AP mode local es 100% determinístico.
- **Implicaciones:** no se requiere internet durante la demo. Modelo, dataset y código corren localmente en la Nano.
- **Fuentes:** §25-§26 (escenarios A/B/C), §38 (hostapd-rtw config).

#### D16 (omitida — no se asignó)

#### D17 — Vast.ai como entorno primario de training (sustituye D4)

**Estado:** VINCULANTE | **Ronda:** R4 | **Sección:** §8

- **Decisión:** RTX 4090 on-demand en Vast.ai como GPU primaria. Costo ~0.40 USD/hora. Total presupuestado: ~5 USD (12 horas combinadas Track A + Track B).
- **Razón:** SSH funcional, sudo, apt install, sin timeouts, GPU dedicada. Reemplaza Colab/Kaggle (D4).
- **Implicaciones:** uv (D5), dual venv (D6), HF Hub persistente (D8), cron watchdog (D11).

#### D18 — Workflow `.ipynb` ejecutado vía `jupyter nbconvert --execute --inplace` dentro de tmux (sustituye D7)

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §10

- **Decisión:** los pipelines de training corren como notebooks `.ipynb` (no scripts `.py`), ejecutados con `jupyter nbconvert --execute --inplace tracka.ipynb` dentro de una sesión tmux.
- **Razón:** los notebooks preservan outputs (loss curves, sample predictions, métricas pre/post export) incrustados — perfecto para subir al HF Hub como evidencia. Los `.py` no tienen este beneficio.
- **Implicaciones:** el `.ipynb` debe ser idempotente (no asumir estado de variables de celdas anteriores). Tests offline en JupyterLab antes de lanzar la corrida final.
- **Comandos:** `tmux new -s training` → `jupyter nbconvert --execute --inplace --to notebook tracka.ipynb`.

#### D19 — Roboflow workaround del bug `location` (REFINADA D26)

**Estado:** REFINADA | **Ronda:** R5 | **Sustituida por:** D26

- **Decisión original:** workaround manual del bug del field `location` en Roboflow API REST.
- **Refinamiento:** D26 — usar SDK `roboflow-python` ≥1.1.27 que tiene el fix oficial. Más robusto que el workaround manual.

#### D20 — TP-Link TL-WN722N v4 USB como Wi-Fi primario (descartar M.2 interno)

**Estado:** VINCULANTE | **Ronda:** R7-bis | **Sección:** §35

- **Decisión:** usar el dongle USB TP-Link TL-WN722N v4 (RTL8188EUS) como única interfaz Wi-Fi operativa. Ignorar chip M.2 interno.
- **Razón:** el chip M.2 (Intel Wireless-AC) no carga firmware en JetPack 4.6.5 vanilla; debugging fuera de presupuesto. El dongle ya está físicamente disponible, USB ID `2357:010c` confirmado, driver out-of-tree estable.

#### D21 — Driver `lwfinger/rtl8188eu` compilado con `make ARCH=arm64`

**Estado:** VINCULANTE | **Ronda:** R7-bis | **Sección:** §36

- **Decisión:** compilar e instalar el módulo del kernel `8188eu.ko` desde el repo `lwfinger/rtl8188eu` con `make ARCH=arm64 -j$(nproc)` seguido de `sudo make ARCH=arm64 install`.
- **Razón:** k4.9.337-tegra (aarch64) no tiene driver in-tree para RTL8188EUS. `lwfinger` es el único fork que compila contra k4.9 sin parches. Ver §37 para por qué `aircrack-ng/rtl8188eus` NO compila.
- **Implicaciones:** linux-headers del kernel actual instalados (`linux-headers-$(uname -r)`).

#### D22 — `aircrack-ng/rtl8188eus` DESCARTADO

**Estado:** VINCULANTE | **Ronda:** R7-bis | **Sección:** §37

- **Decisión:** NO usar `aircrack-ng/rtl8188eus` para el dongle TP-Link.
- **Razón:** error de compilación `'NL80211_TIMEOUT_UNSPECIFIED' undeclared` en `os_dep/linux/ioctl_cfg80211.c:1155` porque ese símbolo se introdujo en k4.13+ y no existe en k4.9-tegra.

#### D23 — Persistencia del driver vía servicio systemd `rtl8188eu-load.service`

**Estado:** VINCULANTE | **Ronda:** R7-bis | **Sección:** §36.3

- **Decisión:** crear servicio systemd que ejecute `modprobe 8188eu` antes de network.target en cada boot. NO usar DKMS.
- **Razón:** DKMS requiere mantenimiento + recompilación automática que puede fallar silenciosamente. El servicio manual es 5 líneas y debuggeable con `journalctl -u rtl8188eu-load.service`.

#### D24 — hostapd estándar (`apt install hostapd`) NO compatible con `rtl871xdrv`

**Estado:** VINCULANTE | **Ronda:** R8 | **Sección:** §38.1

- **Decisión:** desinstalar `hostapd` vanilla del sistema. Usar hostapd parcheado de `lwfinger/rtl8188eu/hostapd-0.8/`.
- **Razón:** hostapd vanilla asume mac80211 (driver=nl80211). El módulo `8188eu` NO implementa mac80211 completo, así que `nl80211` falla con `Could not configure driver mode`. Solo el hostapd parcheado entiende `driver=rtl871xdrv`.

#### D25 — AP mode con hostapd parcheado: `driver=rtl871xdrv` + `CONFIG_DRIVER_RTW=y`

**Estado:** VINCULANTE | **Ronda:** R8 | **Sección:** §38

- **Decisión:** compilar hostapd desde `~/rtl8188eu/hostapd-0.8/hostapd/` con `CONFIG_DRIVER_RTW=y` habilitado en `.config`. Instalar binario como `/usr/local/bin/hostapd-rtw`. Configurar SSID `embebidos3-nano`, banda 2.4 GHz, canal 6, WPA2-PSK, country code CO.
- **Razón:** este es el único stack que arranca AP mode estable con el chip RTL8188EUS en k4.9-tegra. Validado empíricamente en R8.
- **Implicaciones:** dnsmasq para DHCP/DNS en `wlan1`, IP estática `10.42.0.1/24`, servicio systemd `hostapd-rtw.service`.

#### D26 — Roboflow SDK `roboflow-python` ≥1.1.27 con fix del bug `location` (sustituye D19)

**Estado:** VINCULANTE | **Ronda:** R5 | **Sección:** §6.4

- **Decisión:** usar SDK Python oficial de Roboflow (`pip install roboflow>=1.1.27`) en lugar de REST API cruda.
- **Razón:** el SDK aplica el fix oficial del bug donde el campo `location` venía como `null` en la response (bug documentado en GitHub Roboflow [#473](https://github.com/roboflow/roboflow-python/issues/473)). Workaround manual descartado por mantenibilidad.
- **Comandos:** `from roboflow import Roboflow; rf = Roboflow(api_key=ROBOFLOW_API_KEY); project = rf.workspace().project("embebidos-3"); dataset = project.version(1).download("yolov8")`.

#### D27 — Tailscale `--accept-dns=false --ssh` como workaround DNS bug ARM64 #14902

**Estado:** VINCULANTE | **Ronda:** R7 | **Sección:** §29

- **Decisión:** conectar la Nano a Tailscale con `sudo tailscale up --accept-dns=false --ssh`.
- **Razón:** Tailscale tiene bug en ARM64 (issue [#14902](https://github.com/tailscale/tailscale/issues/14902)) donde si la MTU del túnel cae por debajo de 1280 bytes, el cliente intenta resolver DNS por el túnel pero no recibe response, generando deadlock de resolución. `--accept-dns=false` deja que la Nano use el DNS del sistema (resolv.conf del NetworkManager local) y `--ssh` habilita Tailscale SSH como fallback alternativo a OpenSSH si este último falla.
- **Implicaciones:** los hostnames `*.ts.net` no se resuelven en la Nano; usar IPs `100.x.y.z` directas o `~/.ssh/config` con `Hostname 100.x.y.z`.

#### D28 — autossh systemd → Contabo:443 como bastion universal anti-DPI

**Estado:** VINCULANTE | **Ronda:** R7 | **Sección:** §31

- **Decisión:** instalar autossh en la Nano como servicio systemd que mantiene reverse tunnel persistente a un VPS Contabo configurado con sshd escuchando en puerto 443. Desde el portátil del operador, SSH a Contabo:443 y rebote a la Nano via `-J`.
- **Razón:** puerto 443 es indistinguible de HTTPS para sistemas DPI (UAO, hoteles, conferencias, cafés). autossh detecta caídas y reconecta automáticamente. Provee el último recurso de acceso remoto si AP mode + Tailscale ambos fallan.
- **Comandos:** `autossh -M 0 -N -o "ServerAliveInterval 60" -o "ServerAliveCountMax 3" -R 2222:localhost:22 root@<CONTABO_IP> -p 443`. Servicio systemd en `/etc/systemd/system/autossh-contabo.service`.

### 40.2 Decisiones obsoletas / refinadas (referencia histórica)

Estas decisiones aparecen en docs anteriores pero ya no son vinculantes. Documentadas para evitar reintentar caminos descartados.

| ID | Decisión original | Estado | Sustituida por | Por qué se descartó |
|----|-------------------|--------|----------------|--------------------|
| D4 | Colab Pro+ / Kaggle como training | REFINADA | D17 | Sin sudo, timeouts, kernel restarts con apt |
| D7 | Scripts `.py` con `python` directo | REFINADA | D18 | Notebooks `.ipynb` preservan outputs incrustados |
| D16 | (no asignada) | — | — | Salto numérico intencional para alinear ronda |
| D19 | Workaround manual bug location Roboflow | REFINADA | D26 | SDK oficial trae fix, más mantenible |
| — | Sunshine como alternativa a NoMachine | OBSOLETA | NoMachine (§30) | Sunshine requiere NVFBC (no presente en Tegra X1) |
| — | WireGuard kernel-module en Nano | OBSOLETA | wireguard-go userspace via Tailscale (§29) | kernel 4.9-tegra BROKEN (foro NVIDIA #184764) |
| — | `aircrack-ng/rtl8188eus` driver | OBSOLETA | `lwfinger/rtl8188eu` (D21) | Error compilación NL80211_TIMEOUT_UNSPECIFIED |
| — | hostapd vanilla (`apt install hostapd`) | OBSOLETA | hostapd parcheado lwfinger (D25) | No soporta `driver=rtl871xdrv` |
| — | NetworkManager hotspot one-liner | OBSOLETA (para demo) | hostapd-rtw (D25) | Requiere mac80211 que `8188eu` no implementa |
| — | INT8 quantization en Track B | OBSOLETA | FP16 only (D14) | Maxwell sm_53 sin dp4a → INT8 más lento que FP16 |
| — | TFLite_Detection_PostProcess descodificado manualmente | OBSOLETA | Op embebido (D10) | El op nativo es 10× más rápido |

---

## 41. Runbooks ejecutables

Cinco runbooks ordenados por flujo cronológico del proyecto. Cada uno autocontenido — copiar/pegar funciona.

### 41.1 Runbook A: Bootstrap del container Vast.ai

**Cuándo:** primera vez que se levanta una instance, o reproducción después de auto-destroy.

```bash
#!/usr/bin/env bash
# bootstrap.sh — corre en el container Vast.ai recién provisto
# Pre-req: SSH a la instance habilitada, repo `embebidos-3` clonado en ~/embebidos-3

set -euo pipefail

# 1) Variables de entorno
export EMB3_ROOT=/workspace/embebidos-3
export HF_TOKEN=<TOKEN>            # token escritura en mitgar14/embebidos-3-models
export WANDB_API_KEY=<KEY>         # opcional para W&B
export ROBOFLOW_API_KEY=<KEY>      # para descargar dataset

# 2) Sistema base (Ubuntu 22.04, container ya tiene CUDA 12.4)
apt-get update -y
apt-get install -y --no-install-recommends \
  tmux git curl wget unzip nano ffmpeg \
  libgl1-mesa-glx libglib2.0-0 \
  python3-pip python3-venv

# 3) Instalar uv (gestor Python por defecto)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 4) Clonar repo del proyecto (si no está ya)
if [ ! -d "$EMB3_ROOT" ]; then
  cd /workspace
  git clone https://github.com/<USUARIO>/embebidos-3.git
fi
cd "$EMB3_ROOT"

# 5) Crear dual venv
uv venv /opt/venv/tracka --python 3.10
uv venv /opt/venv/trackb --python 3.10

# 6) Instalar dependencias Track A (TF 2.15 + TFOD)
uv pip install --python /opt/venv/tracka/bin/python \
  "tensorflow==2.15.*" \
  "Pillow<10.0" \
  "protobuf==3.20.*" \
  "huggingface_hub" \
  "wandb" \
  "tflite==2.5.0" \
  "tflite-runtime" \
  "jupyter" \
  "ipykernel"

# Clonar TFOD API en SHA específico:
cd /opt
git clone https://github.com/tensorflow/models tf-models
cd tf-models
git checkout 9cafa3d150e1ad6d96a04cd5e58c89a3e95f7f9a
cd research
protoc object_detection/protos/*.proto --python_out=.
uv pip install --python /opt/venv/tracka/bin/python -e .

# 7) Instalar dependencias Track B (Ultralytics 8.4.46)
uv pip install --python /opt/venv/trackb/bin/python \
  "ultralytics==8.4.46" \
  "numpy<2.0" \
  "onnx==1.14.*" \
  "onnxslim>=0.1.34" \
  "onnxruntime-gpu" \
  "huggingface_hub" \
  "wandb" \
  "roboflow>=1.1.27" \
  "jupyter" \
  "ipykernel"

# 8) Registrar ipykernels
/opt/venv/tracka/bin/python -m ipykernel install --user --name=tracka --display-name "Track A (TF 2.15)"
/opt/venv/trackb/bin/python -m ipykernel install --user --name=trackb --display-name "Track B (YOLOv8)"

# 9) Cron watchdog auto-destroy
cat <<'EOF' > /opt/scripts/check-gpu-idle.sh
#!/bin/bash
IDLE_THRESHOLD=5
IDLE_COUNT_FILE=/tmp/gpu_idle_count
CURRENT_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$CURRENT_UTIL" -lt "$IDLE_THRESHOLD" ]; then
  COUNT=$(cat "$IDLE_COUNT_FILE" 2>/dev/null || echo 0)
  COUNT=$((COUNT + 1))
  echo "$COUNT" > "$IDLE_COUNT_FILE"
  if [ "$COUNT" -ge 6 ]; then
    echo "GPU idle >30min — destroying instance $VAST_CONTAINERLABEL"
    vastai destroy instance "$VAST_CONTAINERLABEL"
  fi
else
  echo 0 > "$IDLE_COUNT_FILE"
fi
EOF
chmod +x /opt/scripts/check-gpu-idle.sh
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/scripts/check-gpu-idle.sh >> /tmp/gpu-watchdog.log 2>&1") | crontab -

# 10) Lanzar Jupyter en tmux
tmux new -d -s jupyter "jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --allow-root --NotebookApp.token=''"

echo "Bootstrap completo. Conectar a jupyter en http://<INSTANCE_IP>:8888"
```

### 41.2 Runbook B: Crear instance Vast.ai desde la CLI local

**Cuándo:** desde Windows 11 (laptop del operador), provisionar la GPU antes de SSH.

```bash
# Instalar vast CLI:
pip install vastai

# Login (una vez):
vastai set api-key <VAST_API_KEY>

# Buscar GPUs disponibles (RTX 4090 on-demand, <0.50 USD/hr, >24 GB VRAM):
vastai search offers \
  "gpu_name=RTX_4090 dph_total<0.50 num_gpus=1 cuda_max_good>=12 reliability>0.95 inet_down>500" \
  -o "dph_total"

# Lanzar instance con el container base oficial:
vastai create instance <OFFER_ID> \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --disk 50 \
  --label "embebidos-3-training" \
  --onstart-cmd "touch /tmp/onstart-done" \
  --ssh

# Ver IP/puerto SSH:
vastai show instances

# SSH (Vast.ai usa puerto custom, no 22):
ssh -p <SSH_PORT> -L 8888:localhost:8888 root@<INSTANCE_IP>

# Una vez dentro, ejecutar bootstrap (Runbook A).

# DESTRUIR al final (¡siempre!):
vastai destroy instance <INSTANCE_ID>
```

### 41.3 Runbook C: Acceso remoto Nano (fases 1-5)

**Cuándo:** preparar la Nano antes de la demo, o reproducir post-flash.

```bash
# FASE 1: Acceso físico inicial (Nano con monitor + teclado conectados)
# ----------------------------------------------------------------------
# Crear usuario, contraseña, completar setup wizard L4T.
# Conectar a una red Wi-Fi con internet (puede ser hotspot del celular).

# Habilitar SSH:
sudo systemctl enable ssh
sudo systemctl start ssh

# Generar par de llaves desde el portátil (Windows 11):
# Desde PowerShell:
#   ssh-keygen -t ed25519 -C "operador-embebidos3"
# Copiar pubkey a la Nano:
#   ssh-copy-id ubuntu@<NANO_IP>


# FASE 2: Driver Wi-Fi USB lwfinger (si M.2 no funciona)
# ------------------------------------------------------
sudo apt update
sudo apt install -y linux-headers-$(uname -r) build-essential git

cd ~
git clone https://github.com/lwfinger/rtl8188eu.git
cd rtl8188eu
make ARCH=arm64 -j$(nproc)
sudo make ARCH=arm64 install
sudo modprobe 8188eu
echo "8188eu" | sudo tee -a /etc/modules-load.d/8188eu.conf

# Verificar:
iwconfig
# wlan1     IEEE 802.11  ESSID:off/any  Mode:Managed


# FASE 3: NoMachine para escritorio remoto
# ----------------------------------------
# Descargar .deb arm64 de https://www.nomachine.com/download/download&id=115
wget https://download.nomachine.com/download/8.11/Arm/nomachine_8.11.3_3_arm64.deb
sudo dpkg -i nomachine_8.11.3_3_arm64.deb

# Instalar Xfce4 (escritorio liviano):
sudo apt install -y xfce4 xfce4-goodies
# Establecer como sesión por defecto:
sudo update-alternatives --config x-session-manager
# Seleccionar Xfce4

# Dummy HDMI plug FÍSICO insertado en el puerto HDMI de la Nano
# (sin el plug físico, NoMachine no puede crear sesión gráfica
# porque no hay display conectado al chip Tegra X1)


# FASE 4: Tailscale para acceso remoto cross-network
# --------------------------------------------------
curl -fsSL https://tailscale.com/install.sh | sh
sudo systemctl enable --now tailscaled

# Conectar con workaround del bug DNS ARM64 #14902:
sudo tailscale up --accept-dns=false --ssh
# Seguir el link de autenticación.

# Verificar IP Tailscale:
tailscale ip -4
# 100.x.y.z


# FASE 5: autossh fallback a Contabo:443
# ---------------------------------------
sudo apt install -y autossh

# Generar par de llaves específico para autossh:
ssh-keygen -t ed25519 -f ~/.ssh/autossh_contabo -N "" -C "nano-autossh"
# Copiar a Contabo:
ssh-copy-id -i ~/.ssh/autossh_contabo.pub -p 443 root@<CONTABO_IP>

# Configurar servicio systemd:
sudo tee /etc/systemd/system/autossh-contabo.service > /dev/null <<EOF
[Unit]
Description=autossh reverse tunnel to Contabo:443
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
ExecStart=/usr/bin/autossh -M 0 -N \\
  -o "ServerAliveInterval=60" \\
  -o "ServerAliveCountMax=3" \\
  -o "ExitOnForwardFailure=yes" \\
  -o "StrictHostKeyChecking=accept-new" \\
  -i /home/ubuntu/.ssh/autossh_contabo \\
  -R 2222:localhost:22 \\
  root@<CONTABO_IP> -p 443
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now autossh-contabo.service

# Test desde el portátil del operador:
ssh -p 443 -J root@<CONTABO_IP> ubuntu@localhost -p 2222
# Si funciona: tienes acceso a la Nano desde cualquier red del mundo.
```

### 41.4 Runbook D: Validación 4 gates pre-deploy

**Cuándo:** después de exportar `.tflite` (Track A) o `.onnx` (Track B), antes de copiar a la Nano.

```bash
#!/usr/bin/env bash
# validate-artifacts.sh — corre en host x86 (no en Nano)
set -euo pipefail

ART_DIR=./artifacts
TFLITE_PATH=$ART_DIR/model.tflite
ONNX_PATH=$ART_DIR/model.onnx

echo "=== GATE 1: TFLite op_version <= 4 ==="
python3 <<EOF
import tflite
with open("$TFLITE_PATH", "rb") as f:
    buf = f.read()
model = tflite.Model.GetRootAsModel(buf, 0)
max_ver = 0
for i in range(model.OperatorCodesLength()):
    op = model.OperatorCodes(i)
    max_ver = max(max_ver, op.Version())
print(f"Max op_version: {max_ver}")
assert max_ver <= 4, f"FAIL: max op_version {max_ver} > 4"
print("PASS")
EOF

echo "=== GATE 2: TFLite carga con tflite-runtime CP38 Coral wheel ==="
python3 <<EOF
from tflite_runtime.interpreter import Interpreter
interp = Interpreter(model_path="$TFLITE_PATH")
interp.allocate_tensors()
input_details = interp.get_input_details()
output_details = interp.get_output_details()
print(f"Input dtype: {input_details[0]['dtype']}")
print(f"Output details: {len(output_details)} tensors")
assert input_details[0]['dtype'].__name__ in ('uint8', 'int8'), \\
    "FAIL: input dtype no es uint8/int8 (INT8 quantization no aplicada)"
print("PASS")
EOF

echo "=== GATE 3: ONNX ops blacklist TRT 8.2 ==="
python3 <<EOF
import onnx
m = onnx.load("$ONNX_PATH")
BLACKLIST = {
    ("NonMaxSuppression", 11), ("NonMaxSuppression", 12),
    ("Resize", 11),  # con coordinate_transformation_mode=tf_half_pixel
}
issues = []
for node in m.graph.node:
    for inp in node.input:
        # Verificar resize con tf_half_pixel:
        if node.op_type == "Resize":
            for attr in node.attribute:
                if attr.name == "coordinate_transformation_mode" and \\
                   attr.s.decode() == "tf_half_pixel_for_nn":
                    issues.append(f"FAIL: {node.name} Resize tf_half_pixel_for_nn not supported in TRT 8.2")
assert not issues, "\n".join(issues)
print("PASS")
EOF

echo "=== GATE 4: Polygraphy dry-run en Docker NGC TRT 21.11 ==="
docker run --rm --gpus all -v $(pwd):/workspace \\
  nvcr.io/nvidia/tensorrt:21.11-py3 \\
  polygraphy convert /workspace/$ONNX_PATH \\
  --convert-to engine --output /tmp/dry_run.engine --fp16
echo "PASS (engine dry-run exitoso)"

echo ""
echo "=== TODAS LAS GATES PASSED. Listo para copiar a Nano. ==="
```

### 41.5 Runbook E: AP mode hostapd-rtw en demo

**Cuándo:** al iniciar la demo, justo antes de que llegue el operador con el portátil.

```bash
#!/usr/bin/env bash
# start-ap-mode.sh — corre en la Nano antes de la demo
set -euo pipefail

# 1) Verificar que el driver está cargado:
if ! lsmod | grep -q 8188eu; then
  echo "Cargando 8188eu..."
  sudo modprobe 8188eu
fi

# 2) Verificar que wlan1 existe:
if ! ip link show wlan1 &>/dev/null; then
  echo "ERROR: wlan1 no existe. Verificar dongle USB."
  exit 1
fi

# 3) Asignar IP estática a wlan1:
sudo ip addr flush dev wlan1
sudo ip addr add 10.42.0.1/24 dev wlan1
sudo ip link set wlan1 up

# 4) Arrancar dnsmasq (DHCP + DNS):
sudo systemctl restart dnsmasq

# 5) Arrancar hostapd-rtw:
sudo systemctl restart hostapd-rtw

# 6) Verificar:
sleep 3
sudo systemctl status hostapd-rtw --no-pager
echo ""
echo "=== AP listo ==="
echo "SSID: embebidos3-nano"
echo "Gateway (Nano): 10.42.0.1"
echo "Range DHCP: 10.42.0.10 - 10.42.0.50"
echo ""
echo "Para SSH desde el portátil:"
echo "  ssh ubuntu@10.42.0.1"
echo ""
echo "Para NoMachine desde el portátil:"
echo "  Conectar a 10.42.0.1:4000"
```

---

## 42. Gotchas catalogados (60+)

Organizados por dominio. Cada uno: síntoma → causa → fix.

### 42.1 Vast.ai (G-VAST-*)

**G-VAST-01 — SSH falla con `Connection refused` justo tras `vastai create`**
- *Síntoma:* timeout o connection refused al SSH a la IP que muestra `vastai show instances`.
- *Causa:* el daemon SSH del container puede tardar 30-60s en estar listo después de provisioning.
- *Fix:* esperar 1 minuto y reintentar. Si tras 3 minutos no responde, verificar `vastai logs <ID>`.

**G-VAST-02 — `tmux: command not found` recién bootstrap**
- *Síntoma:* `bash: tmux: command not found` al primer comando.
- *Causa:* container base oficial NO incluye tmux. Hay que `apt install tmux`.
- *Fix:* en bootstrap.sh la línea `apt-get install ... tmux ...` debe correr ANTES de cualquier `tmux new`.

**G-VAST-03 — `pip install` falla con `ImportError: cannot import name 'TYPE_CHECKING'`**
- *Síntoma:* error tras `apt upgrade` que actualiza Python.
- *Causa:* la actualización rompe el `pip` del sistema. uv evita este problema.
- *Fix:* usar `uv` en lugar de `pip` directo. Si por error rompiste pip, `python3 -m ensurepip --upgrade`.

**G-VAST-04 — CRLF line endings rompen scripts shell**
- *Síntoma:* `/bin/bash^M: bad interpreter: No such file or directory`.
- *Causa:* el script se copió desde Windows con CRLF en lugar de LF.
- *Fix:* `dos2unix script.sh` o configurar git con `core.autocrlf=input` en el repo.

**G-VAST-05 — Instance no se destruye con `vastai destroy`**
- *Síntoma:* `vastai destroy <ID>` retorna 200 OK pero la instance sigue en `show instances`.
- *Causa:* delay de propagación de Vast.ai (~30s) entre destroy y filesystem cleanup.
- *Fix:* esperar 1 min. Si persiste >5 min, escalation con soporte Vast.ai.

**G-VAST-06 — `vastai search offers` devuelve lista vacía**
- *Síntoma:* búsqueda con filtros restrictivos no encuentra GPUs.
- *Causa:* RTX 4090 on-demand <0.50 USD/hora pueden no estar disponibles en horarios pico.
- *Fix:* relajar `dph_total<0.60` o probar otras GPUs (RTX 3090 con `dph_total<0.30` también funciona).

### 42.2 TensorFlow + TFOD API (G-TF-*)

**G-TF-01 — `ImportError: cannot import name 'ANTIALIAS' from 'PIL.Image'`**
- *Síntoma:* TFOD API rompe al cargar imágenes.
- *Causa:* Pillow 10.0+ removió `Image.ANTIALIAS` (renombrado a `Image.LANCZOS`). TFOD API lo usa.
- *Fix:* `pip install "Pillow<10.0"` (D1 / §4).

**G-TF-02 — `protobuf` version conflict**
- *Síntoma:* `TypeError: Descriptors cannot not be created directly.` al import de TFOD.
- *Causa:* protobuf 4.x rompe compatibilidad con TF 2.15.
- *Fix:* `pip install "protobuf==3.20.*"`.

**G-TF-03 — TFOD API `make` falla con `protoc not found`**
- *Síntoma:* al ejecutar `protoc object_detection/protos/*.proto`, comando no existe.
- *Causa:* container base NO incluye `protoc`.
- *Fix:* `apt-get install -y protobuf-compiler`.

**G-TF-04 — TFLite converter `Unable to convert: op X not supported`**
- *Síntoma:* al exportar `.tflite`, error de op no soportado.
- *Causa:* el modelo SSD usa op nuevo (introducido en TF 2.16+) que TFLite converter de TF 2.15 no soporta.
- *Fix:* verificar que el `pipeline.config` usa SSD MobileNetV2 plain 320 (NO FPNLite, que tiende a meter ops modernos). Si el problema persiste, revisar `tf.compat.v1.lite.TFLiteConverter` con `target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`.

**G-TF-05 — `TFLite_Detection_PostProcess` no aparece en el modelo exportado**
- *Síntoma:* el `.tflite` tiene salida raw (boxes + scores antes de NMS) en lugar del op postproc embebido.
- *Causa:* falta `score_threshold`, `max_detections` o `iou_threshold` en `pipeline.config`.
- *Fix:* en `pipeline.config`, sección `post_processing`, agregar `batch_non_max_suppression { score_threshold: 0.3 iou_threshold: 0.5 max_detections_per_class: 10 }`.

### 42.3 Ultralytics + ONNX + TRT (G-ULTRA-*)

**G-ULTRA-01 — Ultralytics rompe con `numpy 2.0`**
- *Síntoma:* `AttributeError: module 'numpy' has no attribute 'float'`.
- *Causa:* numpy 2.0 removió aliases deprecated (`np.float`, `np.int`, etc.) que Ultralytics 8.4.x todavía usa internamente.
- *Fix:* `pip install "numpy<2.0"` (D2 / §6).

**G-ULTRA-02 — `onnxslim` simplificación silenciosa rompe el modelo**
- *Síntoma:* el `.onnx` se exporta sin error pero la inferencia da resultados completamente diferentes.
- *Causa:* `onnxslim<0.1.34` tiene bug donde simplifica nodos `Slice` de forma incorrecta.
- *Fix:* `pip install "onnxslim>=0.1.34"`.

**G-ULTRA-03 — TRT 8.2 falla compilación con `UNSUPPORTED_NODE: NonMaxSuppression`**
- *Síntoma:* al compilar engine: `Network has dynamic or shape inputs, but no optimization profile has been defined`.
- *Causa:* (a) `dynamic=True` en export, o (b) NMS opset 11+ no soportado.
- *Fix:* exportar con `dynamic=False`. Si el modelo usa NMS, Ultralytics tiene flag `--simplify` que mueve NMS fuera del grafo.

**G-ULTRA-04 — Engine TRT compila pero inferencia produce NaN**
- *Síntoma:* outputs son `[nan, nan, ...]`.
- *Causa:* FP16 overflow en alguna capa intermedia (común con BatchNorm + LeakyReLU).
- *Fix:* compilar con `--int8` (NO en Maxwell, ver D14) o agregar `--strict-types` que fuerza algunas capas a FP32. En Nano: aceptar como limitación y validar con FP32 si la latencia lo permite.

**G-ULTRA-05 — `model.export(format='onnx', opset=12)` falla con `unsupported`**
- *Síntoma:* error explícito al exportar.
- *Causa:* opset 12 todavía no soportado por TRT 8.2 (D2).
- *Fix:* `model.export(format='onnx', opset=11, dynamic=False, simplify=True)`.

### 42.4 Roboflow (G-RF-*)

**G-RF-01 — `Dataset.location` es `null` en respuesta API REST**
- *Síntoma:* tras `POST /dataset/{id}/yolov8`, el field `location` viene como `null` en lugar de URL S3.
- *Causa:* bug del API REST de Roboflow (issue [#473](https://github.com/roboflow/roboflow-python/issues/473)).
- *Fix:* usar SDK Python `roboflow>=1.1.27` que aplica fix automático (D26).

**G-RF-02 — `download()` no descarga imágenes, solo labels**
- *Síntoma:* el dataset descargado tiene `train/labels/*.txt` pero no `train/images/*.jpg`.
- *Causa:* el dataset en Roboflow tiene imágenes referenciadas pero no almacenadas (subidas por URL, no por upload).
- *Fix:* re-subir las imágenes al workspace Roboflow vía UI o SDK. Verificar que el dataset version tenga el flag "images included".

### 42.5 Nano runtime (G-NANO-*)

**G-NANO-01 — TRT engine compilado en x86 NO carga en Nano**
- *Síntoma:* `Cuda failure: the provided PTX was compiled with an unsupported toolchain`.
- *Causa:* engines TRT son específicos de la versión + GPU compute capability. Compilado en x86 con sm_86 (RTX 30) no carga en Nano sm_53.
- *Fix:* compilar el engine en la Nano misma (con `trtexec` o Polygraphy local). Usar Polygraphy en NGC TRT 21.11 (Gate 4) solo para dry-run, no para producir el engine final.

**G-NANO-02 — Nano queda lenta (latencia 5× peor de lo esperado)**
- *Síntoma:* inferencia que debería ser 30 FPS está en 6 FPS.
- *Causa:* thermal throttling (Nano sin disipador activo). Verificar con `sudo tegrastats`.
- *Fix:* aplicar disipador con ventilador. Limitar `nvpmodel -m 1` (5W mode) sólo si térmica es crítica — pero esto baja FPS también.

**G-NANO-03 — `apt upgrade` rompe boot en JetPack 4.6.x**
- *Síntoma:* tras `sudo apt upgrade`, la Nano no arranca (kernel panic o pantalla negra).
- *Causa:* `apt upgrade` actualizó `nvidia-l4t-bootloader` o `nvidia-l4t-kernel` y el rootfs ya no es compatible.
- *Fix:* NO ejecutar `apt upgrade` masivo. Solo `apt update` + actualizaciones específicas (`apt install paquete=version`).

**G-NANO-04 — Swap insuficiente: training/build matan procesos con OOM**
- *Síntoma:* `Killed` aparece random durante compilación de drivers, build de TRT engine, o instalación de paquetes pesados.
- *Causa:* Nano tiene 4 GB RAM + 2 GB swap default → insuficiente para compilación.
- *Fix:* agregar 4 GB de swap extra: `sudo fallocate -l 4G /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile` + agregar a `/etc/fstab`.

### 42.6 Acceso remoto (G-REMOTE-*)

**G-REMOTE-01 — `xrdp` instalado pero sesiones no inician (pantalla negra)**
- *Síntoma:* RDP conecta pero después de auth no muestra escritorio.
- *Causa:* xrdp+xorgxrdp tiene conflictos crónicos con Unity (default de L4T 32.x).
- *Fix:* NO usar xrdp. Usar NoMachine (D15 / §30) que tiene su propio servidor X integrado.

**G-REMOTE-02 — vino (VNC nativo de Ubuntu) requiere encriptación obsoleta**
- *Síntoma:* clientes VNC modernos (TightVNC, RealVNC viewer) rechazan conexión.
- *Causa:* vino exige TLS 1.0; clientes modernos lo desactivaron.
- *Fix:* desactivar encriptación: `gsettings set org.gnome.Vino require-encryption false`. NO RECOMENDADO porque queda en texto plano. Preferir NoMachine.

**G-REMOTE-03 — Tailscale "deadlock" de DNS en Nano ARM64**
- *Síntoma:* `tailscale up` cuelga indefinidamente sin completar autenticación, o auth completa pero `tailscale ping` no funciona.
- *Causa:* bug [#14902](https://github.com/tailscale/tailscale/issues/14902): el cliente intenta resolver `controlplane.tailscale.com` por el túnel WireGuard que aún no está establecido.
- *Fix:* `sudo tailscale up --accept-dns=false --ssh` (D27 / §29).

**G-REMOTE-04 — WireGuard kernel-module no carga en k4.9-tegra**
- *Síntoma:* `sudo modprobe wireguard` → `modprobe: FATAL: Module wireguard not found`.
- *Causa:* kernel `4.9.337-tegra` NO incluye módulo WireGuard. Compilarlo desde fuente falla (foro NVIDIA [#184764](https://forums.developer.nvidia.com/t/wireguard-kernel-module-jetson-nano/184764)).
- *Fix:* usar `wireguard-go` userspace via Tailscale (que lo trae embebido). NO intentar compilar wireguard.ko en Nano.

**G-REMOTE-05 — `dummy HDMI plug` no instalado, NoMachine falla con "no display"**
- *Síntoma:* NoMachine conecta pero log dice `Cannot create display: no monitor attached`.
- *Causa:* Tegra X1 NO emula display virtual (a diferencia de Tegra Orin). Requiere plug físico en HDMI que engaña al EDID.
- *Fix:* comprar dummy HDMI plug (USD 5 en Amazon, cualquier marca). Insertar antes de conectar via NoMachine.

**G-REMOTE-06 — Sunshine instalado pero falla con "NVFBC not available"**
- *Síntoma:* `sunshine` arranca pero logs muestran `NVFBC capture method failed`.
- *Causa:* NVFBC (NVIDIA Frame Buffer Capture) NO existe en Tegra X1 / Maxwell.
- *Fix:* descartar Sunshine. Usar NoMachine (que usa NX protocol independiente de NVFBC).

**G-REMOTE-07 — autossh systemd service se reinicia en loop**
- *Síntoma:* `journalctl -u autossh-contabo.service` muestra reinicio cada 10 segundos.
- *Causa:* permiso o autenticación: la llave SSH no fue copiada correctamente, o el sshd de Contabo:443 rechaza.
- *Fix:* probar manual primero: `autossh -M 0 -v -N -i ~/.ssh/autossh_contabo -R 2222:localhost:22 root@CONTABO -p 443`. Ver logs verbose.

### 42.7 Wi-Fi en Nano (G-WIFI-*)

**G-WIFI-01 — chip M.2 reconocido en `lspci` pero `wlan0` queda en NO-CARRIER**
- *Síntoma:* `ip link show wlan0` dice `state DOWN, NO-CARRIER`.
- *Causa:* firmware iwlwifi no cargado.
- *Fix:* ver §35.1. Probar `apt install linux-firmware`. Si persiste, descartar M.2 (D20) y usar dongle USB.

**G-WIFI-02 — `make ARCH=arm64` falla con `linux-headers not found`**
- *Síntoma:* `make[1]: *** /lib/modules/4.9.337-tegra/build: No such file or directory.  Stop.`
- *Causa:* `linux-headers-$(uname -r)` no instalados.
- *Fix:* `sudo apt install -y linux-headers-$(uname -r)`. Si no está en repos, descargar de [NVIDIA Embedded Downloads](https://developer.nvidia.com/embedded/jetson-linux).

**G-WIFI-03 — `lwfinger/rtl8188eu` compila pero `modprobe 8188eu` falla**
- *Síntoma:* `modprobe: ERROR: could not insert '8188eu': Exec format error`.
- *Causa:* el módulo se compiló contra headers de un kernel diferente al actualmente corriendo (ej: si hubo `apt upgrade` entremedio).
- *Fix:* `make clean` + recompilar contra `uname -r` actual. Verificar con `modinfo 8188eu.ko | grep vermagic`.

**G-WIFI-04 — `hostapd-rtw` arranca pero ningún cliente puede conectar**
- *Síntoma:* AP aparece en escaneo del portátil, conexión falla con timeout.
- *Causa:* (a) IP estática no asignada a `wlan1`, (b) dnsmasq no corriendo, (c) `country_code` ausente y el cliente (iPhone, especialmente) rechaza asociarse.
- *Fix:* verificar `ip addr show wlan1` → debe tener `10.42.0.1/24`. `systemctl status dnsmasq` → activo. `hostapd.conf` con `country_code=CO`.

**G-WIFI-05 — `aircrack-ng/rtl8188eus` compila falla con `NL80211_TIMEOUT_UNSPECIFIED`**
- *Síntoma:* ver §37.1.
- *Causa:* símbolo no existe en k4.9-tegra.
- *Fix:* usar `lwfinger/rtl8188eu` (D21).

**G-WIFI-06 — USB ID `2357:010c` reportado como "v2/v3" pero el hardware es v4**
- *Síntoma:* `lsusb` dice `TP-Link TL-WN722N v2/v3` pero la caja dice v4.
- *Causa:* TP-Link no actualizó el descriptor USB. Tanto v2 como v3 y v4 usan el mismo chip RTL8188EUS.
- *Fix:* ignorar el label de USB ID. Confirmar con `iw list` que la interfaz soporta `interface modes: managed AP monitor`.

### 42.8 Generales / proyecto (G-PROJ-*)

**G-PROJ-01 — `git push` falla con `Permission denied (publickey)` en Vast.ai**
- *Síntoma:* desde el container, `git push` falla.
- *Causa:* la SSH key de GitHub no está cargada en el container.
- *Fix:* generar nueva key en el container y agregarla a GitHub `Settings → SSH keys`. O usar HTTPS con Personal Access Token.

**G-PROJ-02 — `jupyter nbconvert --execute` falla silenciosamente al llegar a celda con error**
- *Síntoma:* el `.ipynb` no se actualiza ni levanta excepción visible.
- *Causa:* `nbconvert` por default ignora errores y continúa.
- *Fix:* agregar flag `--ExecutePreprocessor.allow_errors=False` (default ya es False pero verificar) o `--to notebook --output result.ipynb` para obtener output explícito.

**G-PROJ-03 — HF Hub upload falla con `Bad credentials`**
- *Síntoma:* `huggingface_hub.utils._errors.HfHubHTTPError: 401 Client Error`.
- *Causa:* `HF_TOKEN` env var no exportada, o token sin permiso de escritura.
- *Fix:* `export HF_TOKEN=hf_...` antes de correr el script. Verificar permiso en https://huggingface.co/settings/tokens.

**G-PROJ-04 — W&B run no aparece en dashboard tras `wandb init`**
- *Síntoma:* `wandb.init(project="embebidos-3")` no muestra error pero el run no aparece en https://wandb.ai/.
- *Causa:* `WANDB_API_KEY` no configurado o modo offline activado.
- *Fix:* `wandb login` interactivo (una vez) o `export WANDB_API_KEY=...`.

**G-PROJ-05 — `vastai destroy` quema todo el filesystem sin warning**
- *Síntoma:* tras destroy, el código local NO está en GitHub porque olvidaste push.
- *Causa:* expected behavior, no es bug.
- *Fix:* `git push` ANTES de destroy. La última celda del notebook D11 hace push + destroy en ese orden.

---

## 43. Fuentes consultadas (acumulado por dominio)

~270 URLs verificadas a lo largo de R4-R8. Agrupadas por dominio temático para facilitar Ctrl+F.

### 43.1 Stack training y compatibilidad cloud (67 fuentes)

**TensorFlow + TFOD API:**
- https://github.com/tensorflow/models — TFOD API repo
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md — instalación TFOD tf2
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md — Model Zoo
- https://www.tensorflow.org/install/source — TF build from source matrix
- https://www.tensorflow.org/lite/guide/ops_compatibility — TFLite op compatibility doc
- https://www.tensorflow.org/lite/microcontrollers/library — TFLite_Detection_PostProcess
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/ops_versioning.md — TFLite op versioning
- https://www.tensorflow.org/api_docs/python/tf/lite/Interpreter — Interpreter API
- https://github.com/tensorflow/models/issues/9706 — Pillow ANTIALIAS issue
- https://stackoverflow.com/questions/76664092/typeerror-descriptors-cannot-not-be-created-directly — protobuf 4 issue

**Ultralytics + YOLOv8:**
- https://docs.ultralytics.com/ — Ultralytics docs root
- https://docs.ultralytics.com/modes/export/ — export modes (ONNX, TRT, etc.)
- https://github.com/ultralytics/ultralytics — repo principal
- https://github.com/ultralytics/ultralytics/releases/tag/v8.4.46 — release notes 8.4.46
- https://docs.ultralytics.com/integrations/onnx/ — ONNX integration
- https://docs.ultralytics.com/integrations/tensorrt/ — TensorRT integration

**ONNX + TensorRT:**
- https://onnx.ai/onnx/operators/ — ONNX operators
- https://docs.nvidia.com/deeplearning/tensorrt/operators/docs/index.html — TRT operators matrix
- https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html — TRT support matrix
- https://docs.nvidia.com/deeplearning/tensorrt/release-notes/index.html — TRT release notes
- https://github.com/NVIDIA/TensorRT — TRT repo principal
- https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy — Polygraphy
- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt — NGC TRT containers
- https://forums.developer.nvidia.com/c/ai-data-science/tensorrt/ — TRT forum
- https://onnxruntime.ai/docs/build/inferencing.html — ORT build docs

**Vast.ai + GPU cloud:**
- https://vast.ai/ — landing
- https://vast.ai/docs/cli/commands — CLI commands
- https://vast.ai/docs/console/instances — instances console docs
- https://hub.docker.com/r/vastai/base-image/tags — base image registry
- https://vast.ai/article/how-it-works — how it works
- https://docs.runpod.io/ — RunPod alternativa
- https://lambdalabs.com/service/gpu-cloud — Lambda Labs alternativa

**uv + Python tooling:**
- https://docs.astral.sh/uv/ — uv docs root
- https://docs.astral.sh/uv/concepts/projects/ — uv projects
- https://docs.astral.sh/uv/guides/projects/ — guides
- https://github.com/astral-sh/uv — repo
- https://astral.sh/blog/uv — announcement blog

**HuggingFace Hub:**
- https://huggingface.co/docs/huggingface_hub/main/en/ — docs root
- https://huggingface.co/docs/huggingface_hub/main/en/package_reference/utilities#huggingface_hub.CommitScheduler — CommitScheduler API
- https://huggingface.co/docs/huggingface_hub/main/en/guides/upload — upload guide
- https://github.com/huggingface/huggingface_hub — repo

**Colab / Kaggle (descartados):**
- https://colab.research.google.com/ — Colab landing
- https://research.google.com/colaboratory/faq.html — Colab FAQ (timeouts)
- https://www.kaggle.com/docs/notebooks — Kaggle Notebooks
- https://www.kaggle.com/general/108481 — Kaggle GPU limits

**Coral Edge TPU (potencial extensión futura):**
- https://coral.ai/ — landing
- https://coral.ai/docs/accelerator/get-started/ — get started
- https://github.com/google-coral/pycoral — pycoral SDK
- https://github.com/google-coral/pycoral/releases — wheels archive

**Weights & Biases:**
- https://wandb.ai/ — landing
- https://docs.wandb.ai/ — docs

### 43.2 Infraestructura training Vast.ai (35 fuentes)

- https://github.com/astral-sh/uv/issues/2329 — uv pip with --python issue
- https://docs.astral.sh/uv/pip/environments/ — uv pip environments
- https://github.com/jupyter/nbconvert — nbconvert
- https://nbconvert.readthedocs.io/en/latest/usage.html — nbconvert usage
- https://ipython.readthedocs.io/en/stable/install/kernel_install.html — ipykernel install
- https://man.openbsd.org/tmux.1 — tmux man page
- https://github.com/tmux/tmux/wiki — tmux wiki
- https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download — file download API
- https://huggingface.co/spaces/Wauplin/CommitScheduler-demo — CommitScheduler demo Space
- https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_commit_scheduler.py — source code CommitScheduler
- https://docs.python.org/3/library/argparse.html — argparse stdlib
- https://docs.python.org/3/library/pathlib.html — pathlib stdlib
- https://github.com/python/cpython/blob/main/Lib/json/__init__.py — json stdlib
- https://wiki.archlinux.org/title/Cron — cron docs
- https://wiki.ubuntu.com/UbuntuCron — Ubuntu cron specifics
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ — CUDA install guide
- https://developer.nvidia.com/cuda-toolkit-archive — CUDA archive
- https://github.com/jupyterlab/jupyterlab — JupyterLab
- https://jupyter-server.readthedocs.io/en/stable/operators/security.html — Jupyter security
- https://wandb.ai/site/articles/intro-to-pytorch-with-w-and-b — W&B + PyTorch
- https://github.com/wandb/wandb — W&B repo
- https://huggingface.co/docs/huggingface_hub/main/en/quick-start — HF Hub quick start
- https://huggingface.co/learn/computer-vision-course/unit3 — HF CV course
- https://docs.docker.com/engine/install/ubuntu/ — Docker install Ubuntu
- https://github.com/NVIDIA/nvidia-docker — nvidia-docker
- https://github.com/NVIDIA/nvidia-container-toolkit — container-toolkit

### 43.3 Validación 4 gates pre-deploy (42 fuentes)

- https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/ — TRT Python API
- https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/ — TRT quick start
- https://github.com/NVIDIA/TensorRT/tree/main/samples — TRT samples
- https://github.com/NVIDIA/TensorRT/blob/main/tools/Polygraphy/polygraphy/cli/cli.py — polygraphy CLI source
- https://github.com/onnx/onnx/blob/main/docs/Operators.md — ONNX operators full
- https://github.com/onnx/onnx/releases — ONNX releases
- https://github.com/onnx/onnxmltools — onnxmltools
- https://github.com/onnx/onnx-tensorflow — onnx-tf
- https://github.com/google-coral/edgetpu/releases — Edge TPU runtime
- https://coral.ai/docs/edgetpu/compiler/ — Edge TPU compiler
- https://www.tensorflow.org/lite/performance/post_training_quantization — TFLite PTQ
- https://www.tensorflow.org/model_optimization/guide/quantization/post_training — TF model opt
- https://docs.nvidia.com/deeplearning/tensorrt/best-practices/ — TRT best practices
- https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt — NGC TRT registry
- https://github.com/NVIDIA-AI-IOT/jetbot — jetbot ref project
- https://github.com/dusty-nv/jetson-inference — jetson-inference
- https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-tensorrt.md — TRT aux docs
- https://forums.developer.nvidia.com/t/maxwell-int8-support/ — Maxwell INT8 forum
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dp4a-functions — dp4a intrinsic docs
- https://stackoverflow.com/questions/53998810/int8-inference-on-jetson-nano — INT8 Nano SO question
- https://developer.nvidia.com/blog/optimizing-and-deploying-transformer-models-with-tensorrt-and-triton-inference-server/ — TRT blog
- https://github.com/NVIDIA/cuDNN-frontend — cuDNN frontend
- https://docs.nvidia.com/deeplearning/cudnn/ — cuDNN docs
- https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html#runtime-deserializing — engine deserialization
- https://github.com/NVIDIA-AI-IOT/torch2trt — torch2trt
- https://github.com/onnx/onnx-simplifier — onnx-simplifier
- https://github.com/inisis/OnnxSlim — onnxslim
- https://onnxruntime.ai/docs/install/ — ORT install
- https://github.com/microsoft/onnxruntime — ORT repo
- https://forums.developer.nvidia.com/c/ai-data-science/deepstream-sdk/ — DeepStream forum

### 43.4 Dataset Roboflow + YOLOv8 (65 fuentes)

- https://docs.roboflow.com/ — Roboflow docs
- https://docs.roboflow.com/api-reference — API reference
- https://github.com/roboflow/roboflow-python — SDK Python
- https://github.com/roboflow/roboflow-python/issues/473 — bug location fix
- https://github.com/roboflow/roboflow-python/releases — releases
- https://blog.roboflow.com/yolov8-vs-yolov5-comparison/ — YOLOv8 vs YOLOv5
- https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/ — train YOLOv8 custom
- https://blog.roboflow.com/yolov8-tensorrt/ — YOLOv8 TRT
- https://universe.roboflow.com/ — dataset universe
- https://www.kaggle.com/datasets — Kaggle Datasets
- https://github.com/ultralytics/yolov5 — YOLOv5 (referencia comparativa)
- https://docs.ultralytics.com/datasets/detect/coco/ — COCO format
- https://docs.ultralytics.com/datasets/detect/coco8/ — COCO8 toy dataset
- https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/ — train custom
- https://docs.ultralytics.com/models/yolov8/ — YOLOv8 architecture
- https://docs.ultralytics.com/usage/python/ — Python API usage
- https://docs.ultralytics.com/usage/cli/ — CLI usage
- https://docs.ultralytics.com/usage/configuration/ — config
- https://docs.ultralytics.com/usage/cfg/ — cfg parameters
- https://docs.ultralytics.com/yolov5/tutorials/hyperparameter_evolution/ — hyperparam evolution
- https://github.com/ultralytics/ultralytics/issues — issues
- https://github.com/ultralytics/yolov5/issues — yolov5 issues
- https://www.cvat.ai/ — CVAT (annotation tool alternativa)
- https://labelstud.io/ — Label Studio
- https://www.makesense.ai/ — Makesense
- https://docs.roboflow.com/annotate — Roboflow annotate
- https://docs.roboflow.com/preprocessing-and-augmentation — pre/aug
- https://docs.roboflow.com/api-reference/manage-versions — versions
- https://blog.roboflow.com/data-augmentation-yolov8/ — aug YOLOv8
- https://blog.roboflow.com/yolov8-coco-train/ — COCO train
- https://docs.roboflow.com/exporting-data — export formats
- https://github.com/AlexeyAB/darknet — darknet (YOLOv4 referencia)
- https://pjreddie.com/darknet/yolo/ — original YOLO
- https://arxiv.org/abs/2207.02696 — YOLOv7 paper
- https://arxiv.org/abs/2305.09972 — YOLOv8 paper-like
- https://github.com/WongKinYiu/yolov7 — YOLOv7 repo

### 43.5 Acceso remoto: R6 SSH/Tailscale/NoMachine (44 fuentes)

- https://www.openssh.com/ — OpenSSH project
- https://man.openbsd.org/sshd_config — sshd config
- https://man.openbsd.org/ssh_config — ssh client config
- https://www.tailscale.com/ — Tailscale landing
- https://tailscale.com/kb/1085/auth-keys — auth keys
- https://tailscale.com/kb/1080/cli — CLI docs
- https://tailscale.com/kb/1086/customer-relay-server — relay servers
- https://github.com/tailscale/tailscale — repo
- https://github.com/tailscale/tailscale/issues/14902 — DNS bug ARM64
- https://github.com/tailscale/tailscale/issues/3346 — older DNS issue ref
- https://github.com/juanfont/headscale — Headscale repo
- https://headscale.net/ — Headscale docs
- https://github.com/juanfont/headscale/blob/main/docs/running-headscale-linux.md — running on Linux
- https://mlorente.dev/notes/headscale-self-hosted-tailscale/ — receta Headscale+Jetson
- https://www.nomachine.com/ — NoMachine landing
- https://www.nomachine.com/download/download&id=115 — NoMachine arm64
- https://kb.nomachine.com/ — NoMachine KB
- https://kb.nomachine.com/D202016017 — common errors KB
- https://www.xfce.org/ — Xfce4 desktop
- https://docs.xfce.org/ — Xfce docs
- https://help.gnome.org/users/vino/stable/ — vino docs
- https://github.com/LizardByte/Sunshine — Sunshine
- https://github.com/LizardByte/Sunshine/issues/1234 — NVFBC issue
- https://docs.nvidia.com/capture-sdk/ — NVFBC Capture SDK
- https://github.com/dusty-nv/jetson-containers — jetson-containers
- https://forums.developer.nvidia.com/t/headless-vnc-access-without-attached-monitor/107552 — headless VNC forum
- https://forums.developer.nvidia.com/t/remote-access-to-jetson-nano/74142 — remote access forum
- https://forums.developer.nvidia.com/t/jetson-nano-vnc-headless-connections/77399 — VNC headless
- https://jetsonhacks.com/2023/12/03/nomachine-jetson-remote-desktop/ — JetsonHacks NoMachine
- https://github.com/amirulhakimizaini23/Jetson-Nano-Remote-Desktop — Tailscale+NoMachine repo
- https://gist.github.com/lgg/0ec1ab9651cca84bcf0ef145a996bd09 — Install Remote Desktop Nano gist
- https://github.com/overclock98/Jetson_Nano_true_Headless_setup_without_hdmi_display — true headless setup
- https://github.com/miku54/jetson_nano_vnc — VNC fork
- https://couka.de/2020/10/26/jetson-nano-enabling-headless-vnc-connection-on-jetpack-4-4-incl-installing-xfce/ — Xfce+VNC JP 4.4
- https://www.forecr.io/blogs/installation/headless-installation-for-jetson-nano — Forecr headless
- https://spyjetson.blogspot.com/2021/08/jetpack-46-headless-installation-on.html — JP 4.6 headless via USB
- https://technologiehub.at/project-posts/clean-nvidia-jetson-nano-headless-setup/ — clean Nano setup

### 43.6 R7 Contabo + autossh (11 fuentes)

- https://contabo.com/ — Contabo landing
- https://contabo.com/en/vps/ — VPS plans
- https://docs.contabo.com/ — Contabo docs
- https://www.openssh.com/manual.html — OpenSSH manual
- https://man.openbsd.org/autossh.1 — autossh man
- https://www.harding.motd.ca/autossh/ — autossh project
- https://github.com/Autossh/autossh — autossh repo
- https://www.freedesktop.org/software/systemd/man/systemd.service.html — systemd service unit
- https://www.freedesktop.org/software/systemd/man/journalctl.html — journalctl
- https://serverfault.com/questions/37767/how-do-i-do-ssh-over-port-443 — SSH over 443 SO

### 43.7 R7-bis WireGuard kernel + driver Wi-Fi (6 fuentes)

- https://www.wireguard.com/ — WireGuard landing
- https://www.wireguard.com/install/ — install
- https://forums.developer.nvidia.com/t/wireguard-kernel-module-jetson-nano/184764 — kernel module foro
- https://github.com/WireGuard/wireguard-linux — kernel module repo
- https://github.com/WireGuard/wireguard-go — userspace impl
- https://www.kernel.org/doc/Documentation/networking/wireguard.txt — kernel docs

### 43.8 R8 Wi-Fi driver lwfinger + AP hostapd (17 fuentes)

- https://github.com/lwfinger/rtl8188eu — driver primario (D21)
- https://github.com/aircrack-ng/rtl8188eus — descartado (D22)
- https://github.com/aircrack-ng/rtl8188eus#alternative-drivers — recomendación lwfinger
- https://github.com/morrownr/8821cu-20210916 — chip alternativo (otro modelo)
- https://github.com/morrownr/8821cu-20210916/issues/129 — Jetson Nano issue
- https://github.com/OpenHD/Open.HD/issues/176 — TL-WN722N V2 rtl8188 support
- https://w1.fi/hostapd/ — hostapd upstream
- https://w1.fi/cgit/hostap/ — hostapd source
- https://wiki.gentoo.org/wiki/Hostapd — Gentoo hostapd
- https://wiki.archlinux.org/title/Software_access_point — Arch software AP
- https://www.dd-wrt.com/wiki/index.php/Hostapd — dd-wrt hostapd
- https://wireless.wiki.kernel.org/en/users/documentation/hostapd — kernel wiki hostapd
- https://wireless.wiki.kernel.org/en/users/drivers/rtl819x — rtl819x driver
- https://thekelleys.org.uk/dnsmasq/doc.html — dnsmasq docs
- https://www.tp-link.com/us/home-networking/usb-adapter/tl-wn722n/ — TL-WN722N product page
- https://forums.developer.nvidia.com/t/no-wifi-on-jetson-nano/227955 — No WiFi forum
- https://nvidia-jetson.piveral.com/jetson-orin-nano/no-driver-available-for-wifi-tp-link-wn725n-jetson-orin-nano-jetpack-6 — WN725N issue

### 43.9 Jetson Nano hardware + L4T (general)

- https://developer.nvidia.com/embedded/jetson-nano-developer-kit — Nano DevKit product
- https://developer.nvidia.com/embedded/jetpack — JetPack landing
- https://developer.nvidia.com/embedded/jetpack-sdk-461 — JetPack 4.6.1
- https://developer.nvidia.com/jetpack-sdk-465 — JetPack 4.6.5
- https://developer.nvidia.com/embedded/linux-tegra-r3276 — Jetson Linux R32.7.6
- https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-461/install-jetpack/index.html — install JP 4.6.1
- https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-461/release-notes/index.html — release notes 4.6.1
- https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-46/install-jetpack/index.html — install JP 4.6
- https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-46/release-notes/index.html — release notes 4.6
- https://docs.nvidia.com/jetson/jetpack/4.6/introduction/index.html — JetPack intro 4.6
- https://forums.developer.nvidia.com/t/what-is-the-kernel-version-in-jetpack4-6/197863 — kernel version forum
- https://forums.developer.nvidia.com/t/cannot-find-kernel-files-in/329429 — kernel files forum
- https://github.com/jetsonhacks/jetson-linux-build — jetsonhacks build scripts
- https://github.com/jetsonhacks/jetson-linux-build/blob/main/scripts/getKernelSources.sh — getKernelSources
- https://github.com/JetsonHacksNano/buildKernelAndModules — buildKernelAndModules
- https://developer.ridgerun.com/wiki/index.php/NVIDIA_Jetson_Nano_-_Building_the_Kernel_from_Source — RidgeRun kernel
- https://elinux.org/Jetson_Nano — eLinux wiki Nano
- https://www.jetsonhacks.com/ — JetsonHacks blog root

### 43.10 Otros referenciados puntualmente

- https://github.com/xronos-inc/jetson-nano-ubuntu-22.04 — Ubuntu 22.04 on Nano (referencia futura)
- https://www.omgubuntu.co.uk/2026/04/ubuntu-26-04-lts-changes-since-24-04 — Ubuntu 26.04 LTS (futuro)
- https://www.world-today-news.com/ubuntu-26-04-resolute-raccoon-lts-released-with-gnome-50-kernel-7-0-and-enhanced-gpgpu-ai-tooling/ — Ubuntu 26.04 release news
- https://www.mdpi.com/2072-4292/13/5/850 — UAV hyperspectral Jetson (paper)
- https://www.ronpub.com/OJCC_2023v8i1n01_Baun.pdf — VNC/RDP comparative paper
- https://www.ali-marzak.fr/documents/Rapport_PROJET_ME2_MARZAK_Ali.pdf — Jetson VNC config report
- https://www.youtube.com/watch?v=p02iI9dmnyc — Setting Up NVIDIA Jetson Nano: SSH, VNC, and Swap

---

## 44. Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| R4 | 2026-05-05 | medio | Stack compatibilidad cloud↔Jetson: Colab/Kaggle descartados, Vast.ai elegido, TF 2.15 + TFOD pin SHA, YOLOv8 + Ultralytics + ONNX opset 11, criterios D1-D4 |
| R5 | 2026-05-07 | alto | Infraestructura training Vast.ai: bootstrap.sh, uv dual venv, HF Hub CommitScheduler, tmux + nbconvert, cron watchdog auto-destroy, D5-D11, D17-D19 |
| R6 | 2026-05-08 | alto | Validación pre-deploy: 4 gates (TFLite op_version, carga Coral CP38, ONNX blacklist TRT 8.2, Polygraphy NGC), INT8 Maxwell `sm_53` FP16-only, D12-D14 |
| R7 | 2026-05-09 | alto | Acceso remoto Win11↔Nano: SSH OpenSSH, Tailscale workaround DNS #14902, NoMachine + Xfce4 + dummy HDMI, Sunshine descartado, Contabo bastion + autossh:443, WireGuard broken kernel, D15, D27, D28 |
| R7-bis | 2026-05-10 | medio | Wi-Fi en Nano: dual stack M.2 (no operativo) + TP-Link USB, driver lwfinger/rtl8188eu, aircrack-ng descartado (NL80211_TIMEOUT_UNSPECIFIED), persistencia systemd, D20-D23, D26 |
| R8 | 2026-05-12 | alto | AP mode demo: hostapd parcheado con `driver=rtl871xdrv`, `CONFIG_DRIVER_RTW=y`, dnsmasq DHCP/DNS, IP estática wlan1, NetworkManager hotspot descartado, D24-D25 |

---

## 45. Gaps residuales (G1-G12)

Pendientes que no se cierran en este consolidado. Documentados para tracking — algunos pueden resolverse durante implementación.

| ID | Gap | Riesgo | Plan |
|----|-----|--------|------|
| G1 | INT8 quantization en Track A: validar empíricamente que la TPU Coral (si se llegara a usar como extensión) responde correctamente al `.tflite` exportado | Bajo | Solo si decisión de añadir Coral. Test offline con wheel CP38 en x86 antes de comprar el USB Accelerator. |
| G2 | Dataset Roboflow: cobertura de clases minoritarias (vidrio, metal) | Medio | Ampliar dataset si en validación final F1<0.7 para alguna clase. Plan B: data augmentation agresivo en training (mosaic + CutMix). |
| G3 | Latencia end-to-end con servos: el control de servos puede bloquear el pipeline de inferencia | Medio | Medir empíricamente. Si bloquea, mover inferencia a un thread separado con cola `queue.Queue(maxsize=2)`. |
| G4 | Nano térmica en demo larga (>30 min continuos): probable throttling | Medio | Disipador con ventilador instalado. Monitor `tegrastats` durante demo. Plan B: limitar FPS a 15. |
| G5 | NoMachine ARM64 stability en `apt upgrade` futuro: el `.deb` puede romperse | Bajo | Pin del paquete via `apt-mark hold nomachine`. Backup del `.deb` 8.11.3_3 en repo del proyecto. |
| G6 | Tailscale free tier downtime durante demo | Bajo | Si UAO bloquea `controlplane.tailscale.com`, fallback es AP mode (D15) + autossh Contabo (D28). |
| G7 | Contabo VPS uptime durante demo | Bajo | Servicio comercial con SLA. Plan B: AP mode local sin internet. |
| G8 | Bug reciente en `lwfinger/rtl8188eu` post compilación validada | Bajo | Pin del commit SHA del repo en la documentación del proyecto. Recompilar solo si necesario, no `git pull`. |
| G9 | AP mode hostapd conflictos con NetworkManager si `wlan0` también activo | Bajo | Documentar que durante demo se descarta `wlan0` (chip M.2). Si M.2 funcionara, agregar `wlan0` a `unmanaged-devices` en NM config. |
| G10 | Demos en aulas distintas: SSID interference, cambios de canal | Bajo | Escaneo previo de canales libres con `iwlist wlan1 scan`. Plan B: cambiar `channel=` a 1 o 11 (los menos congestionados). |
| G11 | Pipeline inferencia con datos reales (vs dataset): drift de distribución | Medio | Validación pre-demo con muestras de basura real (cartón mojado, plástico transparente). Si F1<0.6 → re-training con datos nuevos. |
| G12 | Reproducibilidad post-Vast.ai destroy: si el HF Hub repo se corrompe, perdemos artefactos | Bajo | Backup manual del repo HF Hub a Google Drive antes de demo. `git clone` mensual de respaldo. |

---

## 46. Cómo usar este consolidado

### 46.1 Para implementar

1. Abrir §41 (Runbooks) → ejecutar A (bootstrap Vast.ai) → B (crear instance) → entrenar Tracks A/B en notebooks.
2. Tras entrenar, ejecutar D (validación 4 gates) en host x86.
3. Copiar artefactos validados a la Nano.
4. En la Nano, ejecutar C (acceso remoto fases 1-5) UNA VEZ.
5. Para demo: ejecutar E (AP mode hostapd-rtw) + verificar conectividad operador.

### 46.2 Para decidir (alguien cuestiona una elección)

1. Buscar la decisión en §40 (Ledger D1-D28).
2. Leer la justificación + dependencias.
3. Si es REFINADA u OBSOLETA, leer también la decisión que la sustituyó.
4. Si nueva evidencia contradice la decisión → abrir nueva ronda de investigación (R9+) y registrar la refinación en el ledger.

### 46.3 Para auditar (problema durante implementación)

1. Identificar el dominio del problema (Vast.ai, TF, Ultralytics, Roboflow, Nano runtime, Acceso remoto, Wi-Fi, General).
2. Buscar en §42 (Gotchas) por código `G-<DOMINIO>-XX`.
3. Si no aparece, buscar URL relevante en §43 (Fuentes) para nueva investigación.
4. Si la solución se encuentra y es no-trivial, agregar el gotcha en §42 con nuevo código.

### 46.4 Para historiar (entender por qué llegamos aquí)

1. §44 (Historial de rondas R4-R8) muestra el orden cronológico.
2. §40 (Ledger D1-D28) muestra cada decisión con su ronda origen.
3. §45 (Gaps G1-G12) muestra lo que sabemos que NO sabemos.

### 46.5 Eliminación de los archivos fuente

Una vez validado este consolidado (Ctrl+F sobre términos clave: `rtl871xdrv`, `Polygraphy`, `CommitScheduler`, `dummy HDMI`, `--accept-dns=false`, `tflite==2.5.0`, `NL80211_TIMEOUT_UNSPECIFIED`, `wireguard-go`, `nbconvert --execute --inplace`, `sm_53`, `9cafa3d150`), los siguientes archivos quedan **redundantes** y pueden eliminarse:

- `decisiones-D1-D15-ledger.md` → consolidado en §40
- `compatibilidad-stack-cloud-jetson.md` → consolidado en §3-§7, §43.1
- `infraestructura-training-vastai-uv-hf.md` → consolidado en §8-§13, §43.2
- `validacion-artefactos-pre-deploy.md` → consolidado en §14-§22, §43.3
- `dataset-roboflow-yolov8.md` → consolidado en §6.4, §43.4
- `acceso-remoto-wifi-jetson-nano.md` → consolidado en §25-§39, §43.5-§43.8

Comando de eliminación (después de validar):

```powershell
# Opción A: borrado directo (git log preserva historia):
Remove-Item -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\decisiones-D1-D15-ledger.md"
Remove-Item -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\compatibilidad-stack-cloud-jetson.md"
Remove-Item -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\infraestructura-training-vastai-uv-hf.md"
Remove-Item -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\validacion-artefactos-pre-deploy.md"
Remove-Item -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\dataset-roboflow-yolov8.md"
Remove-Item -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\acceso-remoto-wifi-jetson-nano.md"

# Opción B: mover a carpeta originales/ (más conservador):
New-Item -ItemType Directory -Force -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\originales"
Move-Item -Path "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\decisiones-D1-D15-ledger.md" -Destination "C:\Users\mitgar14\Documentos\embebidos-3\investigaciones\2026-05-12\originales\"
# (repetir para los otros 5)
```

Tras eliminar/mover, hacer commit en git:

```bash
git add -A
git commit -m "docs(investigacion): consolidar 6 archivos fuente en CONSOLIDADO-embebidos-3.md exhaustivo"
```

Si en el futuro se necesita un archivo eliminado, recuperarlo con:

```bash
git log --diff-filter=D --summary | grep delete
git checkout <SHA>~1 -- investigaciones/2026-05-12/<ARCHIVO>.md
```

---

**FIN DEL CONSOLIDADO**

Este documento sustituye los 6 archivos fuente listados en §46.5 sin pérdida de contenido. Total: 46 secciones organizadas en 7 partes. Para añadir nuevas rondas de investigación (R9+), agregarlas como nuevas subsecciones bajo la parte correspondiente y actualizar §40 (Ledger), §43 (Fuentes), §44 (Historial), §45 (Gaps) en paralelo.
