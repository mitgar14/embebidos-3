# Compatibilidad stack Cloud ↔ Jetson Nano B01 (JetPack 4.6.1)

**Proyecto:** `embebidos-3` (clasificador de residuos para Jetson Nano B01, entrega 2026-05-26).
**Dominio:** matriz de versiones de plataformas de training (Vast.ai elegida, Colab y Kaggle descartados) frente al runtime fijo del Jetson Nano B01 con JetPack 4.6.1. Justificación de por qué cada decisión de stack se tomó.
**Documentos hermanos:** [`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md) (D1, D2 dependen de este doc) · [`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md) · [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) · [`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md).
**Fecha de cierre:** 2026-05-12.

---

## 1. Resumen ejecutivo

El runtime del **Jetson Nano B01 con JetPack 4.6.1 es inmutable** (Python 3.6.9, TF 2.5.0+nv21.8, TFLite runtime 2.5, TensorRT 8.2.1.8, CUDA 10.2, Maxwell `sm_53` sin Tensor Cores INT8 ni instrucción `dp4a`). El stack de training en la nube debe ser **forward-compatible**: producir `.tflite` (Track A) o `.onnx` (Track B) que carguen sin error en runtime 2.5 / TRT 8.2.1.

Tras dos fallos reproducibles en Colab (Track A → `condacolab.install` no baja Python a 3.10 porque Miniforge 23.11.0-0 trae Python 3.12; `mamba install python=3.10` falla por pin `google-colab` → 3.12) y constraints de Kaggle (PyTorch 2.9–2.10 con NumPy 2.4 default + sin soporte para TF OD API legacy), la decisión vinculante (D1, D2) es **migrar ambos tracks a Vast.ai** con container `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` (Python 3.10) y dos `uv venv` aislados.

Pin de stack final:

- **Track A:** TF 2.15 + `tf-models-official` 2.15 + `tensorflow/models@9cafa3d150` + Pillow 10.4 + protobuf 3.20.3 + grpcio-tools 1.64.1.
- **Track B:** torch 2.1+cu121 + Ultralytics 8.4.46 + `numpy<2.0` + onnxslim ≥ 0.1.82 + W&B nativo.

---

## 2. Hardware target — Jetson Nano Developer Kit B01

### 2.1 Specs hardware

| Componente | Especificación |
|------------|----------------|
| SoC | NVIDIA Tegra X1 (TX1) |
| CPU | ARM Cortex-A57 quad-core @ 1,43 GHz (aarch64) |
| GPU | NVIDIA Maxwell, **128 CUDA cores**, `sm_53` |
| RAM | 4 GB LPDDR4 64-bit @ 1600 MHz, compartida CPU/GPU |
| Tensor Cores | **❌ Ninguno** (Maxwell predates Volta) |
| Instrucción `dp4a` (dot product 4 × INT8) | **❌ NO disponible** (introducida en Pascal `sm_61` en 2016) |
| Storage | microSD (recomendado U3/V30, 64 GB+) |
| Power modes | 5 W (`MAXN-5W`) y 10 W (`MAXN`) |
| Cámara | USB UVC v4l2 (proyecto usa Logitech C920 OG Rev 1, PID `046d:082d`) |

### 2.2 Software inmutable (JetPack 4.6.1)

| Componente | Versión exacta | Notas |
|------------|----------------|-------|
| L4T | R32.7.1 | kernel Linux 4.9.337 |
| OS | Ubuntu 18.04 LTS | bionic, EOL upstream pero Nano sigue soportada |
| **Python** | **3.6.9** | system, no actualizable sin riesgo de romper L4T |
| **TensorFlow** | **2.5.0+nv21.8** | wheel oficial NVIDIA `tensorflow==2.5.0+nv21.8` |
| **TFLite runtime** | **2.5** | bundled con TF; fallback wheel Coral CP36 aarch64 si falla custom op (D15) |
| **TensorRT** | **8.2.1.8** | bundled JetPack; APIs Python + C++ |
| **CUDA** | **10.2** | runtime + dev |
| **cuDNN** | **8.2.1** | bundled CUDA |
| **OpenCV** | **4.1.1** | sin GStreamer support compilado por default; suficiente para v4l2 → numpy |
| **TFLite schema** | **v3** | estable desde TF 2.x (`TFLITE_SCHEMA_VERSION = 3` en `tensorflow/lite/version.h` HEAD = TF 2.21 master) |

### 2.3 Implicación crítica: ausencia de `dp4a` → INT8 sin speedup

Maxwell `sm_53` carece de la instrucción `dp4a` (dot product de 4 × INT8 acumulando a INT32) introducida en Pascal `sm_61` en 2016. Sin `dp4a`, TensorRT tiene tres opciones para INT8:

1. **Kernels CUDA INT8 SIMD vía `dp4a`** → no disponible en `sm_53`.
2. **Emular INT8 vía FP16/FP32** → elimina cualquier beneficio de velocidad y añade overhead de conversiones.
3. **Mixed precision fallback** → TensorRT revierte la capa a FP16, generando grafo mixto con conversiones adicionales.

Confirmación NVIDIA (issue [`NVIDIA/TensorRT#3762`](https://github.com/NVIDIA/TensorRT/issues/3762)): *"`--int8` means Enable int8 precision, in addition to fp32."* INT8 nunca reemplaza FP32, lo complementa; las capas no cuantizables revierten.

**Consecuencia operativa:** Track B **FP16-only por default** (D14 del ledger). Detalle en [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md).

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

## 3. Por qué NO Colab (decisión cerrada, informativo)

Track A se intentó en Google Colab con `condacolab` para bajar Python a 3.10. La cadena de fallos reproducibles fue:

### 3.1 Fallo 1 — Colab 2026.04 default Python 3.12

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

### 3.2 Fallo 2 — `condacolab.check()` lanza `AssertionError`, no devuelve `False`

Source verbatim ([`condacolab.py` línea 320](https://github.com/conda-incubator/condacolab/blob/main/condacolab.py)):

```python
assert find_executable("conda"), "Conda not found!"
```

El patrón `if not condacolab.check(): condacolab.install()` rompe con `AssertionError: Conda not found!` en el primer run. Solución: llamar `install_from_url(URL)` directamente — ya hace `try: check(); except AssertionError: pass` internamente (`condacolab.py` líneas 132–136).

### 3.3 Fallo 3 — Miniforge 23.11.0-0 trae Python 3.12

Observado empíricamente 2026-05-12: tras `condacolab.install_from_url("23.11.0-0/Miniforge3-23.11.0-0-Linux-x86_64.sh")` y kernel restart, `sys.version_info` sigue en 3.12.13.

- Release notes 23.11.0-0 **no especifican** versión de Python explícitamente.
- Hipótesis 1: el installer fue rebuild con Python 3.12 retroactivamente.
- Hipótesis 2: el wrapper de `condacolab` no exec el conda Python correctamente en Colab moderno (`/usr/bin/python3` es ahora un wrapper shell que hace `exec /opt/conda/bin/python`).
- Intento de fix: `mamba install python=3.10` post-restart → **falla** porque `google-colab` está pinneado a 3.12 en el env conda → solver de mamba no encuentra resolución.

### 3.4 Fallos auxiliares confirmados (gotchas informativos)

| # | Gotcha | Fuente |
|---|--------|--------|
| C1 | `condacolab.install_miniforge` NO acepta `python_version`. Firma real: `install_miniforge(prefix, env, run_checks, restart_kernel)` | `condacolab.py` línea 233, rama `main` |
| C2 | `do_shutdown(True)` es **asíncrono** → necesita `sys.exit(0)` después | `install_from_url` líneas 132–136 |
| C3 | Tras kernel restart, NO se mantienen las definiciones de celdas previas | Comportamiento estándar de Jupyter |
| C4 | `list(NEW_SRC)` produce caracteres individuales al asignar `nb['cells'][idx]['source']`. Usar `splitlines(keepends=True)` | Memoria mnemon `a8c6ef5b` |
| C5 | TF 2.16+ rompe TF OD API por removal de `tf.estimator` | Issues [tensorflow/models#13575](https://github.com/tensorflow/models/issues/13575), [#13599](https://github.com/tensorflow/models/issues/13599) |
| C6 | `graph_rewriter` para QAT silently broken en TF2 desde 2021 | Issue [tensorflow/models#9835](https://github.com/tensorflow/models/issues/9835) |

**Decisión vinculante:** Track A abandona Colab. Decisión cerrada. No re-investigar.

---

## 4. Por qué NO Kaggle (decisión cerrada, informativo)

### 4.1 Stack Kaggle GPU v168 (marzo 2026)

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

### 4.2 Bloqueadores para Track A

- **Sin soporte TF OD API legacy:** TF 2.15 no tiene wheel cp312 (idem Colab).
- **Mismo problema de Python:** bajar a 3.10 vía conda en Kaggle es aún más fricción que en Colab (no tiene `condacolab` equivalente).
- **Quota GPU 30 h/sem** (vs Colab Pro adaptativa). Aceptable para Track B pero restrictivo para múltiples experimentos.

### 4.3 Bloqueadores para Track B (no fatales, pero frágiles)

- **NumPy 2.x rompe `ultralytics` en runtime** aunque `pyproject.toml` declare `numpy<2.0`. Deps transitivas pueden dejar 2.x. Issue [`ultralytics/ultralytics#22346`](https://github.com/ultralytics/ultralytics/issues/22346) "NumPy 2.2.6 import errors when running on Kaggle T4x2".
- **Conflicts de pip** con preinstalled Kaggle packages: issue [`#22336`](https://github.com/ultralytics/ultralytics/issues/22336) "dependency conflicts when install ultralytics" en Python 3.11.13.
- **PyTorch 2.10 con CUDA 12.8** no afecta el `.onnx` exportado (export con `device='cpu'`), pero introduce variables no controladas en el grafo si Ultralytics regresiona.

### 4.4 Quotas comparativas (informativo)

| Plataforma | Sesión máx | Cuota semanal | RAM | GPU típica |
|------------|-----------|---------------|-----|-----------|
| Colab Free | 12 h | adaptativa | ~12 GB | T4 / K80 compartida |
| Colab Pro | 12 h | mejor disponibilidad | High-RAM | T4 / V100 prioritaria |
| Kaggle GPU | 12 h | 30 h/sem | 29 GB | T4×2 o P100 |
| **Vast.ai (elegida)** | sin cap | sin cuota | 32–128 GB | **RTX 4090 24 GB** |

**Decisión vinculante:** ni Colab ni Kaggle. Vast.ai resuelve los dos tracks con un solo container.

---

## 5. Por qué SÍ Vast.ai (D1, D2)

### 5.1 Container elegido — `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`

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

**Alternativa robusta** si la build custom da problemas: imagen oficial `tensorflow/tensorflow:2.15.0-gpu` preinstalada para Track A.

### 5.2 Quirks confirmados de Vast.ai

#### Naming de imágenes Docker Hub (memoria mnemon `fba73ac3`, 2026-04-16)

- El repo `vastai/pytorch` es **OBSOLETO** (PyTorch 1.0 + CUDA 10.0). No usar.
- Usar `vastai/base-image` para construir imágenes custom.
- Tags compuestos para apps específicas: `vastai/openwebui:v0.5.7-cuda-12.1-pytorch-2.5.1-py311`, `vastai/vllm:v0.8.1-cuda-12.1-pytorch-2.5.1-py312`.
- Patrón de nombres de tags `vastai/base-image`: `cuda-X.Y.Z-cudnn-devel-ubuntu22.04-py310`.

#### Subcomandos del CLI

- Subcomando `vastai search templates` **NO existe** en el CLI. Los `template_hash` se extraen de la UI `cloud.vast.ai/templates/` y se pasan con `--template_hash`.
- Verificación reciente: `gh api repos/vast-ai/vast-python/releases?per_page=5` confirma el alcance del CLI.

#### SSH y `--ssh` flag (memoria mnemon `d5f717eb`, 2026-04-16)

- `--ssh` **reemplaza ENTRYPOINT** del container con proceso propio que INYECTA `sshd` desde fuera.
- Si `/usr/sbin/sshd` NO está en la imagen (caso `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`), Vast.ai intenta `apt-get install openssh-server` EN RUNTIME dentro del container.
- Falla con "Connection refused" permanente cuando el host bloquea `archive.ubuntu.com:80` (IP `185.125.190.82`).
- Confirmado por `rolandtannous` (colaborador Vast.ai) en [`unslothai/unsloth#4682`](https://github.com/unslothai/unsloth/issues/4682) (marzo 2026), [`vast-ai/base-image#141`](https://github.com/vast-ai/base-image/issues/141) (abril 2026), [`vast-ai/vast-cli#336`](https://github.com/vast-ai/vast-cli/issues/336) (feb 2026, reproducido en Corea y China).
- **Mitigación para nuestro caso:** la imagen `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` ya incluye `sshd`. No aplica.

#### Conversión CRLF en Windows (memoria mnemon `27b66a6b`)

- Git en Windows convierte LF → CRLF al checkout por default (`core.autocrlf=true`).
- Los `.sh` y `.py` shipped a Vast.ai Linux tienen `\r\n`, y bash falla con `set: pipefail: invalid option name` porque ve `pipefail\r`.
- **FIX obligatorio:** `.gitattributes` con `*.sh text eol=lf` y `*.py text eol=lf` antes de pushear bootstrap.
- Aplicar también a `.ipynb` y `.json` si se editan en Windows.

### 5.3 GPU recomendada (D2): RTX 4090 on-demand

| GPU | Arquitectura | VRAM | TFLOPS FP16 | Precio Vast.ai mayo 2026 |
|-----|--------------|------|-------------|---------------------------|
| **RTX 4090** | **Ada Lovelace `sm_89`** | **24 GB** | **1008** | **0,35–0,50 USD/h on-demand · 0,14–0,31 USD/h spot** |
| A100 40 GB | Ampere `sm_80` | 40 GB | 312 | 0,80–1,40 USD/h on-demand |
| RTX 3090 | Ampere `sm_86` | 24 GB | 285 | 0,25–0,40 USD/h on-demand |
| H100 80 GB | Hopper `sm_90` | 80 GB | 1979 | 2,00+ USD/h on-demand |

**Justificación:** SSD MV2 320 + YOLOv8n 416 son modelos pequeños (≤ 5 M parámetros, batch 32–64 entra en 4 GB VRAM); A100/H100 son desperdicio. Con 1,72 USD de saldo del usuario: 4–12 h de RTX 4090, holgado para 1–3 h de training por track.

**Quote del usuario (R4 verbatim):** *"Usar la mejor GPU. No me importan costos."* Se elige 4090 sobre 5090 por madurez de drivers CUDA 12.4 y disponibilidad consistente.

### 5.4 Alternativas si Vast.ai falla (memoria mnemon `1d6d237e`)

1. **RunPod** — limitado a PyTorch 2.6–2.9 (eliminó 2.1 del catálogo en 2025). Útil para Track B si Vast.ai cae temporalmente; **no apto para Track A** (TF 2.15 con torch 2.6+ tiene conflictos de protobuf).
2. **Lambda Labs** — buena GPU disponibility pero menos flexibilidad en imágenes Docker.
3. **Paperspace** — opciones gratuitas con tier limitado.
4. **Cluster UAO `uaodeepia11306`** — si está disponible para uso académico.

---

## 6. Stack Track A — TF 2.15 + TFOD API + PTQ (CPU TFLite + XNNPACK)

### 6.1 Decisión de modelo y precisión

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

### 6.2 SSD MV2 plain vs FPNLite (referencia)

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

### 6.3 Pin de dependencias Track A (kernel `tracka`)

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

### 6.4 Clone TF Models en pin SHA `9cafa3d150`

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

### 6.5 Compilación de protos (evitar `runtime_version` bug)

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

### 6.6 Tabla `op_version` TFLite 2.5 max vs TF 2.15 export

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

**Detalles completos del gate de validación TFLite:** ver [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §"Gate Track A".

### 6.7 MediaPipe Model Maker (Plan B, informativo)

Restricciones del wheel `mediapipe-model-maker`:

- `requirements.txt` verbatim: `tensorflow>=2.10,<2.16`, `tf-models-official>=2.13.2,<2.16.0`, `tensorflow-model-optimization<0.8.0`.
- `setup.py` Python classifiers: **3.8, 3.9, 3.10** (no 3.11/3.12).
- `MOBILENET_V2_I320` usa checkpoint `gs://tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i320_ckpt.tar.gz` — QAT preintegrado.
- **NMS omitido en TFLite export:** `tflite_post_processing=configs.common.TFLitePostProcessingConfig(omit_nms=True)` → el `.tflite` no contiene `TFLite_Detection_PostProcess`. Requiere decoder custom (decoding de anchors + NMS) en Nano.
- Dataset solo COCO / PASCAL VOC. Roboflow exporta ambos formatos → conversión trivial desde TFRecord.
- Caveat de accuracy: [`discuss.ai.google.dev/t/mediapipe-massive-accuracy-loss-with-quantization-aware-training/23177`](https://discuss.ai.google.dev/t/mediapipe-massive-accuracy-loss-with-quantization-aware-training/23177).

**No activar como Plan B sin nueva ronda `/investiga`.**

---

## 7. Stack Track B — PyTorch + Ultralytics + ONNX + TRT (GPU Maxwell)

### 7.1 Decisión de export

**Veredicto (R1 cerrado):** mantener YOLOv8n 416×416 + Ultralytics ≥ 8.4.46 + ONNX opset 11 explícito + onnxslim ≥ 0.1.82 + FP16 TRT engine construido en Nano + NMS en CPU NumPy.

Razones:

- **Kaggle/Colab traen PyTorch 2.9–2.10 con CUDA 12 host.** No afecta el `.onnx` exportado con `device='cpu'` (PyTorch exporta por trazado en CPU; el ONNX es portátil).
- **ONNX opset 11 es soportado por TRT 8.2.1.** [`onnx-tensorrt/docs/operators.md?ref=release/8.2-GA`](https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md) verbatim: *"TensorRT 8.2 supports operators up to Opset 13."*
- **Ultralytics 8.4.x default opset es 20 con torch 2.9+.** Hay que forzar `opset=11` explícitamente.
- **`onnxsim==0.4.36` no compila en Python 3.12** (issue [`#334`](https://github.com/daquexian/onnx-simplifier/issues/334) `daquexian/onnx-simplifier`). Pin a `>=0.6.2,<0.7`.
- **Ultralytics 8.3+ migró de `onnxsim` a `onnxslim`** (verificado en source de `ultralytics/engine/exporter.py`). El flag `simplify=True` llama `onnxslim.slim(model_onnx)`. Pin `onnxsim` es irrelevante para el exporter.
- **`EfficientNMS_TRT` plugin no funciona estable en Maxwell** (issue [`NVIDIA/TensorRT#1538`](https://github.com/NVIDIA/TensorRT/issues/1538)) → NMS en CPU NumPy con `cv2.dnn.NMSBoxes` o `torchvision.ops.nms` en el Nano.

### 7.2 Pin de dependencias Track B (kernel `trackb`)

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

### 7.3 Export ONNX canónico

```python
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt')

model.export(
    format='onnx',
    opset=11,           # CRÍTICO: default 20, TRT 8.2 no garantiza ops opset 14+
    simplify=True,      # llama onnxslim.slim() internamente
    dynamic=False,      # shapes fijas para TRT engine determinístico
    imgsz=416,          # PR #24028 fix INT8 calib non-square
    device='cpu',       # CUDA host irrelevante para .onnx
    half=False,         # FP16 se aplica en trtexec, no en ONNX
    int8=False,         # FP16-only por D14
    nms=False,          # NMS en CPU NumPy en Nano (EfficientNMS_TRT roto Maxwell)
)
# Output: runs/detect/train/weights/best.onnx
```

**Verificación post-export:**

```python
import onnx
m = onnx.load('best.onnx')
assert m.opset_import[0].version == 11, f"Opset {m.opset_import[0].version} ≠ 11"
assert m.ir_version <= 10, f"IR version {m.ir_version} > 10 puede romper onnxslim 0.6.x"
ops = {n.op_type for n in m.graph.node}
print(f"Opset: {m.opset_import[0].version}, IR: {m.ir_version}")
print(f"Ops únicos: {sorted(ops)}")
```

**Detalles del gate de validación ONNX (D13) + ops blacklist TRT 8.2:** ver [`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md) §"Gate Track B".

---

## 8. Gotchas acumulados (tabla)

| # | Plataforma / Componente | Gotcha | Mitigación / Referencia |
|---|-------------------------|--------|-------------------------|
| 1 | Colab Py 3.12 + TF 2.13/2.14/2.15 | No hay wheels cp312 | Migrar a Vast.ai (D1) |
| 2 | TF OD API + TF 2.16+ | `tf.estimator` removido (#13575, #13599) | Pin `tensorflow==2.15.0` |
| 3 | TF OD API + `graph_rewriter` | QAT silently broken (#9835) | PTQ post-train; o Plan B MediaPipe |
| 4 | TFMOT + SSD FPN | No soporta arquitecturas compuestas | No usar para Track A |
| 5 | MediaPipe Model Maker | NMS omitido en export TFLite | Solo Plan B, decoder custom Nano |
| 6 | Roboflow `version.download(location=X)` | Path relativo a CWD si `location=None`; bug residual aún en 1.3.9 | `os.chdir(WORK_DIR)` + `DATASET_DIRECTORY` env + cascada (ver [`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md)) |
| 7 | Roboflow `data.yaml` clases fantasma (#88) | Backend bug sin fix | Validar `nc:` y `names:` manualmente |
| 8 | `EfficientNMS_TRT` en Maxwell | Plugin roto (#1538) | NMS en CPU NumPy |
| 9 | `onnxsim 0.4.36` en Py 3.12 | No compila (#334) | Pin `>=0.6.2`; o usar `onnxslim` (Ultralytics 8.3+) |
| 10 | Ultralytics `best_onnx_opset` | Default 20 con torch 2.9+ | Forzar `opset=11` explícito |
| 11 | `onnxsim` breaking changes 0.5.0+ | `--dynamic-input-shape`, `--input-shape` deprecados | Usar nueva sintaxis o `onnxslim` |
| 12 | Kaggle GPU quota 30 h/sem | Sesión cap 12 h | (No aplica, usamos Vast.ai) |
| 13 | Colab idle disconnect | 30–90 min sin actividad | (No aplica) |
| 14 | `DATASET_DIRECTORY` env var | Debe estar set ANTES de `Roboflow()` | Set en bootstrap (#4') |
| 15 | Pillow 12 + OD API legacy | `Image.fromarray` stricter type checking | Pin `Pillow==10.4.0` + clone TF Models SHA `9cafa3d150` |
| 16 | `protobuf` con OD API | conflicto si versión > 3.20 | Pin `protobuf==3.20.3` + env var |
| 17 | TFLite forward-compat TF 2.15 → 2.5 | `op_version` puede ser nuevo | Validar con `flatbuffer_utils.py` pre-deploy (D12) |
| 18 | Track B Roboflow Kaggle secret | (No aplica, usamos Vast.ai) | — |
| 19 | YOLOv8n CPU TFLite en Nano | < 10 FPS, no viable | Track B solo TRT FP16 GPU |
| 20 | TRT engine OOM Jetson Nano (#14751) | TRT build necesita workspace | `trtexec --workspace=1024` en Nano |
| 21 | Pin SHA TF Models `v2.15.0` | NO contiene `research/` | Usar SHA `9cafa3d150` de master |
| 22 | `condacolab.check()` lanza `AssertionError` | No devuelve `False` | (No aplica, sin Colab) |
| 23 | Miniforge 23.11.0-0 | Trae Python 3.12 no 3.10 | (No aplica) |
| 24 | NumPy 2.x rompe ultralytics en runtime | Even con `pyproject.toml numpy<2.0` heredado | Pin explícito `numpy<2.0` ANTES de ultralytics |
| 25 | Vast.ai SSH (`d5f717eb`) | Falla si `/usr/sbin/sshd` no en imagen + host bloquea archive.ubuntu.com | `vastai/base-image:cuda-...` ya incluye sshd |
| 26 | Vast.ai SSH key del proyecto | Path: `.ssh/id_ed25519`; CLI en `C:\Users\mitgar14\AppData\Roaming\Python\Python312\Scripts\vastai.exe` | Memoria mnemon `b13050ac` |
| 27 | Vast.ai usar `python3` no `python` | `python` alias puede no existir | Memoria mnemon `67358204` |
| 28 | Git Windows CRLF → CRLF en `.sh`/`.py` | `set: pipefail: invalid option name` con `\r\n` | `.gitattributes` con `*.sh text eol=lf` y `*.py text eol=lf` |
| 29 | Repos `vastai/pytorch` y `vastai/tensorflow` | Tags algunos obsoletos | Usar `vastai/base-image` para custom |

---

## 9. Fuentes consultadas (acumuladas R1–R5)

| # | Título | URL | Tipo | Relevancia |
|---|--------|-----|------|------------|
| 1 | Colab Runtime Version FAQ | https://research.google.com/colaboratory/runtime-version-faq.html | Doc oficial | Colab Py 3.12 |
| 2 | Kaggle Notebook Specs | https://www.kaggle.com/docs/notebooks | Doc oficial | Kaggle stack 2026 |
| 3 | TensorRT 8.2 ops support | https://github.com/onnx/onnx-tensorrt/blob/release/8.2-GA/docs/operators.md | Doc oficial | Ops opset 11 TRT 8.2 |
| 4 | PyTorch Version Compatibility | https://github.com/pytorch/pytorch/wiki/PyTorch-Versions | Doc oficial | torch ↔ CUDA |
| 5 | Ultralytics Jetson Guide | https://docs.ultralytics.com/guides/nvidia-jetson/ | Doc oficial | YOLOv8 + Nano |
| 6 | MediaPipe Model Maker (Object Detector) | https://ai.google.dev/edge/mediapipe/solutions/customization/object_detector | Doc oficial | Plan B Track A |
| 7 | TF blog 2024-03 Python 3.12 support | https://blog.tensorflow.org/2024/03/whats-new-in-tensorflow-216.html | Blog oficial | TF 2.16+ Py 3.12 |
| 8 | TF blog 2022-06 QAT Model Garden | https://blog.tensorflow.org/2022/06/Adding-Quantization-aware-Training-and-Pruning-to-the-TensorFlow-Model-Garden.html | Blog oficial | QAT receta |
| 9 | DeepWiki tensorflow/models | https://deepwiki.com/tensorflow/models | Doc generada | TFOD API state |
| 10 | DeepWiki google-ai-edge/mediapipe | https://deepwiki.com/google-ai-edge/mediapipe | Doc generada | Plan B |
| 11 | DeepWiki roboflow/roboflow-python | https://deepwiki.com/roboflow/roboflow-python/4.2-dataset-download | Doc generada | Bug location |
| 12 | googlecolab/colabtools #5483 Py 3.12 | https://github.com/googlecolab/colabtools/issues/5483 | Issue | Colab Py 3.12 |
| 13 | tensorflow/models #13575 estimator | https://github.com/tensorflow/models/issues/13575 | Issue | TF 2.16+ rompe OD API |
| 14 | tensorflow/models #13599 PR guard TF 2.16 | https://github.com/tensorflow/models/issues/13599 | Issue/PR | Idem |
| 15 | tensorflow/models #9835 graph_rewriter | https://github.com/tensorflow/models/issues/9835 | Issue | QAT placebo TF2 |
| 16 | tensorflow/models #11168 eval_pb2 | https://github.com/tensorflow/models/issues/11168 | Issue | Protos no compilados |
| 17 | tensorflow/tensorflow #62003 Py3.12 | https://github.com/tensorflow/tensorflow/issues/62003 | Issue | TF 2.15 sin wheel cp312 |
| 18 | NVIDIA/TensorRT #1538 EfficientNMS Maxwell | https://github.com/NVIDIA/TensorRT/issues/1538 | Issue | NMS Maxwell roto |
| 19 | ultralytics/ultralytics #14751 Nano OOM | https://github.com/ultralytics/ultralytics/issues/14751 | Issue | TRT engine OOM |
| 20 | ultralytics/ultralytics #10298 postprocess Nano | https://github.com/ultralytics/ultralytics/issues/10298 | Issue | NMS Nano |
| 21 | ultralytics/ultralytics #7222 FP16 TRT | https://github.com/ultralytics/ultralytics/issues/7222 | Issue | FP16 export |
| 22 | ultralytics/ultralytics #19498 upsample opset8 | https://github.com/ultralytics/ultralytics/issues/19498 | Issue | IR version |
| 23 | daquexian/onnx-simplifier #334 Py 3.12 wheel | https://github.com/daquexian/onnx-simplifier/issues/334 | Issue | onnxsim Py 3.12 |
| 24 | daquexian/onnx-simplifier #367 ir_version | https://github.com/daquexian/onnx-simplifier/issues/367 | Issue | IR version mismatch |
| 25 | discuss.ai.google.dev MediaPipe QAT accuracy | https://discuss.ai.google.dev/t/mediapipe-massive-accuracy-loss-with-quantization-aware-training/23177 | Foro | Plan B caveat |
| 26 | Qengineering YoloV8 TensorRT Jetson | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | Repo | FP16 reference |
| 27 | the0807 YOLOv8 ONNX TensorRT | https://github.com/the0807/YOLOv8-ONNX-TensorRT | Repo | INT8 Orin reference |
| 28 | Qengineering TensorFlow-JetsonNano (wheels) | https://github.com/Qengineering/TensorFlow-JetsonNano | Repo | TF wheel Nano |
| 29 | google-coral.github.io tflite_runtime wheels | https://google-coral.github.io/py-repo/tflite-runtime/ | Doc oficial | Plan B Coral CP36 |
| 30 | NobuoTsukamoto/benchmarks (Jetson Nano) | https://github.com/NobuoTsukamoto/benchmarks | Repo benchmarks | FPS reference |
| 31 | Jacob et al. 2018 — QAT integer-only | https://arxiv.org/abs/1712.05877 | Paper CVPR | QAT theory |
| 32 | Karimov et al. 2025 — Quantization robustness | https://arxiv.org/abs/2508.19600 | Paper arXiv | PTQ drop 3-7 pp |
| 33 | Zagitov et al. 2024 — Edge object detection | https://doi.org/10.18287/2412-6179-CO-1343 | Paper Computer Optics | FPS threshold |
| 34 | Trisuwita et al. 2024 — SSD MV2 helmet | https://doi.org/10.34010/komputika.v13i1.10333 | Paper Komputika | Plain > FPNLite dominio |
| 35 | TF2 Detection Model Zoo | https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md | Doc oficial | Checkpoints |
| 36 | tf-models-official 2.15.0 PyPI JSON | https://pypi.org/pypi/tf-models-official/2.15.0/json | PyPI metadata | Pin Pillow sin constraint |
| 37 | tensorflow 2.15.0 PyPI JSON | https://pypi.org/pypi/tensorflow/2.15.0/json | PyPI metadata | protobuf constraint |
| 38 | condacolab.py (main, v0.1.4) | https://github.com/conda-incubator/condacolab/blob/main/condacolab.py | Código fuente | install_from_url |
| 39 | Miniforge 24.5.0 release notes (Py 3.12 base) | https://github.com/conda-forge/miniforge/releases/tag/24.5.0-0 | Release notes | Pin Py |
| 40 | Miniforge 23.11.0-0 release | https://github.com/conda-forge/miniforge/releases/tag/23.11.0-0 | Release notes | Hipótesis Py 3.12 |
| 41 | commit tf/models 971ded9e16 (Pillow 12 patch) | https://github.com/tensorflow/models/commit/971ded9e16 | Commit | Pin SHA upper bound |
| 42 | commit tf/models 9cafa3d150 (pre-Pillow12) | https://github.com/tensorflow/models/commit/9cafa3d150 | Commit | **Pin elegido** |
| 43 | commit protobuf 554a00c (runtime_version) | https://github.com/protocolbuffers/protobuf/commit/554a00c | Commit | protoc 5.x bug |
| 44 | SO #78671850 (grpcio-tools 1.64.1 fix) | https://stackoverflow.com/questions/78671850 | SO answer | Fix proto compile |
| 45 | SO #19548957 (pip --force-reinstall --no-deps) | https://stackoverflow.com/questions/19548957 | SO answer | Re-pin defensivo |
| 46 | Ultralytics PR #24028 (INT8 calib no-square) | https://github.com/ultralytics/ultralytics/pull/24028 | PR | Pin >= 8.4.31 |
| 47 | Ultralytics PR #23807 (Docker pytorch 2.10) | https://github.com/ultralytics/ultralytics/pull/23807 | PR | Stack moderno |
| 48 | Ultralytics PR #23808 (safer opset cap torch 2.9+) | https://github.com/ultralytics/ultralytics/pull/23808 | PR | opset 11 forzado |
| 49 | Ultralytics #22346 (NumPy 2.2.6 Kaggle) | https://github.com/ultralytics/ultralytics/issues/22346 | Issue | numpy<2.0 pin |
| 50 | Ultralytics #22336 (Kaggle dep conflicts) | https://github.com/ultralytics/ultralytics/issues/22336 | Issue | Kaggle frágil |
| 51 | Ultralytics source `engine/exporter.py` main | https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py | Código fuente | onnxslim no onnxsim |
| 52 | Kaggle/docker-python Dockerfile.tmpl | https://github.com/Kaggle/docker-python/blob/main/Dockerfile.tmpl | Repo oficial | Stack Kaggle v168 |
| 53 | vastai/base-image Dockerfile | https://github.com/vast-ai/base-image/blob/main/Dockerfile | Código fuente | Container base |
| 54 | vast-ai/vast-python (vastai CLI) | https://github.com/vast-ai/vast-python | Código fuente | Sin `--auto-stop` |
| 55 | docs.vast.ai create-instance | https://docs.vast.ai/cli/reference/create-instance | Doc oficial | Flags válidos |
| 56 | docs.vast.ai Jupyter & SSH FAQ | https://docs.vast.ai/documentation/reference/faq/jupyter-ssh | Doc oficial | Jupyter mode |
| 57 | hub.docker.com/r/vastai/pytorch tags | https://hub.docker.com/r/vastai/pytorch/tags | Doc oficial | Repo obsoleto |
| 58 | hub.docker.com/r/vastai/tensorflow tags | https://hub.docker.com/r/vastai/tensorflow/tags | Doc oficial | Imágenes TF |
| 59 | TensorFlow For Jetson Platform Release Notes | https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html | Doc oficial | TF 2.5.0+nv21.8 |
| 60 | docs.nvidia.com TensorRT 8.2.2 Support Matrix | https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-822/support-matrix/index.html | Doc oficial | TRT 8.2 features |
| 61 | JetPack 4.6.1 release page | https://developer.nvidia.com/embedded/jetpack-sdk-461 | Doc oficial | Stack inmutable |
| 62 | JetPack 4.6.1 jetsonhacks blog | https://jetsonhacks.com/2022/03/10/jetpack-4-6-1-production-release/ | Blog | Resumen JetPack |
| 63 | foro NVIDIA "Polygraphy on Jetson Nano (TRT 8.2.1.8)" | https://forums.developer.nvidia.com/t/how-to-generate-and-verify-an-int8-calibration-cache-cache-for-trtexec-on-on-jetson-nano-tensorrt-8-2-1-8-polygraphy-failing-on-device/349598 | Foro | Polygraphy ⊥ Nano |
| 64 | foro NVIDIA "TRT INT8 conversion fails assertion Orin" | https://forums.developer.nvidia.com/t/tensorrt-int8-conversion-fails-with-assertion-error-using-ultralytics/331356 | Foro | INT8 Orin frágil |
| 65 | foro Google AI Developers — runtime_version | https://discuss.ai.google.dev/t/importerror-cannot-import-name-runtime-version-from-google-protobuf/22770 | Foro | Workaround |
| 66 | tensorflow/hub commit a01d02f9 (ANTIALIAS) | https://github.com/tensorflow/hub/commit/a01d02f9 | Commit | Pillow 10 shim |
| 67 | pip docs v26.1.1 — constraint files | https://pip.pypa.io/en/stable/topics/dependency-resolution/ | Doc oficial | Pin strategy |

---

## 10. Cross-references

- **[`decisiones-D1-D15-ledger.md`](decisiones-D1-D15-ledger.md)** — D1 (container Vast.ai) y D2 (RTX 4090) salen de este doc.
- **[`infraestructura-training-vastai-uv-hf.md`](infraestructura-training-vastai-uv-hf.md)** — Implementación de `bootstrap.sh` con los pins de §6 y §7 + dos `uv venv` (D10).
- **[`validacion-artefactos-pre-deploy.md`](validacion-artefactos-pre-deploy.md)** — Gates basados en la tabla `op_version` de §6.6 (D12) y ops blacklist TRT 8.2 de §7.3 (D13).
- **[`dataset-roboflow-yolov8.md`](dataset-roboflow-yolov8.md)** — Bug Roboflow `location` mencionado en gotchas #6 y #7.
- **[`HANDOFF-implementacion-vastai-hf.md`](HANDOFF-implementacion-vastai-hf.md)** — Plan operativo §5 (#2'-#5').

---

**Fin del documento.** Cualquier cambio al stack (D1, D2 o pins de §6, §7) requiere nueva ronda `/investiga`.
