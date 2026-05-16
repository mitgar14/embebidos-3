# embebidos-3 — clasificador de residuos en Jetson Nano

> **MVP académico** para curso de IA en sistemas embebidos (UAO).
> **Entrega:** 2026-05-26.
> **Tarea:** detección de objetos en tiempo real sobre banda transportadora — 3 clases: `paper`, `glass`, `plastic`.
> **Hardware target:** Jetson Nano Developer Kit 4 GB rev. B01, **JetPack 4.6.1** (L4T R32.7.1, Ubuntu 18.04, Python 3.6.9, CUDA 10.2.300, TensorRT 8.2.1.8, GPU Maxwell `sm_53` 128 CUDA cores **sin tensor cores INT8**, sin instrucción `dp4a`).
> **Actuación:** 3 servomotores SG90 vía PCA9685 (I²C) — rampas deflectoras por clase.
> **Cámara:** Logitech C920 OG (USB UVC 720p) montada en diagonal sobre la banda.

---

## 1. Decisión arquitectónica — Track B exclusivo

El proyecto se enfoca exclusivamente en **YOLOv8n → ONNX → TensorRT FP16** (Track B). Track A (SSD MobileNet v2 + TFLite INT8 sobre CPU) fue evaluado y **descartado el 2026-05-13** por dos razones:

1. **Mejor rendimiento esperado en este hardware.** La GPU Maxwell de la Tegra X1 solo brilla con FP16 (no tiene tensor cores INT8 ni `dp4a`). Track A correría en CPU como en una Raspberry Pi 4 — desaprovecha el hardware.
2. **Ahorro de tiempo de cara a la entrega 2026-05-26.** Track A depende de QAT obligatorio, calibración con representative dataset, `TFLite_Detection_PostProcess` embebido y cuatro gates pre-deploy. Track B también tiene gates pero son más predecibles, y el dataset 1-B en Roboflow ya está listo desde el 2026-05-11.

**Validación empírica (SSH 2026-05-13):** YOLOv8n FP16 416×416 corre a **~40 FPS / 25 ms** end-to-end en este Jetson Nano específico (JP 4.6.1), **superando la predicción de Nature 2024 Tabla 4 (30 FPS)**. El margen sobre el threshold MVP (≥10 FPS) es 4×, dejando holgura para concurrencia con los servos.

**Stack training vs runtime**

| | Training (Vast.ai) | Runtime (Jetson Nano JP 4.6.1) |
|---|---|---|
| Hardware | RTX 4090 (24 GB) | Maxwell `sm_53`, 128 CUDA cores, 4 GB unified |
| Container/OS | `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` | Ubuntu 18.04 (L4T R32.7.1) |
| Python | 3.10 | 3.6.9 |
| Gestor paquetes | **uv** (`/opt/venv/trackb`) | apt + pip (sistema) |
| Framework | PyTorch 2.x + Ultralytics 8.4.x | TensorRT 8.2.1.8 runtime |
| Quantización | FP32 (training) | **FP16** (engine en Nano vía `trtexec --fp16`) |
| NMS | desacoplado (`nms=False` en ONNX) | tri-path: V0 `cv2.dnn.NMSBoxes` CPU (default), V1 `EfficientNMS_TRT`, V2 `BatchedNMSDynamic_TRT` |
| Persistencia | HuggingFace Hub vía `CommitScheduler` cada 10 min | scp manual desde Vast.ai |

Decisiones completas y razones: [`investigaciones/HANDOFF-track-b-2026-05-13.md`](investigaciones/HANDOFF-track-b-2026-05-13.md) (registro maestro de decisiones D2-D30).

---

## 2. Estado actual (2026-05-14)

| Componente | Estado | Notas |
|---|---|---|
| Dataset Roboflow `embebidos3/waste-3class-lwld8` v1-B (`yolov8`, Fit-black 416×416, 3 clases) | Hecho | 17 910 train / 1 739 valid / 844 test, validado en dry-run local. |
| Investigaciones consolidadas | Hecho | HANDOFF + ronda 2026-05-14 (uv-en-notebook, NMS Maxwell, headless training). |
| `bootstrap.sh` Track B v1 | Hecho | Idempotente: apt + uv + venv `/opt/venv/trackb` + kernel `trackb` + cron watchdog auto-destroy + JupyterLab en tmux. Aún sin validar en Vast.ai. |
| Notebook training Track B | Hecho | 29 celdas, headless vía `nbconvert + tmux`, `CommitScheduler` cada 10 min con signal handlers, heartbeat TRAINCHECK-style, ONNX opset=11 + Gates 3/4, auto-destroy Vast.ai. Dry-run parcial (§1-9) ejecutado OK 2026-05-14. |
| Engine TRT FP16 en Nano | Validado empíricamente | ~40 FPS / 25 ms confirmado vía SSH 2026-05-13 con un `.engine` previo. Repetir con el `best.onnx` fine-tuned del primer entrenamiento Vast.ai. |
| Sprint 1 phase C — provisioning Vast.ai end-to-end | Pendiente | Subir bootstrap.sh + notebook, ejecutar nbconvert en tmux, producir `best.onnx` fine-tuned. |
| Pipeline runtime (captura + infer + I²C 3 hilos) | Pendiente | Spec en investigaciones; implementación post-engine. |

---

## 3. Estructura del repositorio

```
embebidos-3/
├── README.md                                          ← este archivo
├── CLAUDE.md                                          ← memo de contexto para Claude (acceso SSH al Nano)
├── pyproject.toml                                     ← uv project (Python ≥3.10, host deps)
├── uv.lock
├── .gitattributes                                     ← LF para .sh / .py / .ipynb / .yml / .md
├── main.py                                            ← stub
│
├── investigaciones/                                   ← input directo para el informe IEEE
│   ├── HANDOFF-track-b-2026-05-13.md                  ← registro maestro de decisiones D2-D30
│   └── 2026-05-14-training-headless-uv-nms-maxwell.md ← ronda /investiga: uv-en-notebook, NMS Maxwell, headless
│
├── notebooks/
│   └── train_track_b_yolov8.ipynb                     ← 29 celdas, kernel `trackb`, headless Vast.ai
│
└── scripts/
    ├── server/                                       ← FastAPI + WS, recovery, constantes
    ├── builder/                                      ← pipeline TRT (download, trtexec, validate, swap)
    ├── hub/                                          ← HF REST (download/upload, raw + LFS)
    ├── install/                                      ← installers (systemd units, inference deps)
    ├── training/
    │   └── bootstrap.sh                              ← provisiona host Vast.ai (idempotente)
    └── dashboard/                                    ← UI estática (HTML/JS) servida en local
```

---

## 4. Investigaciones (input directo para el informe IEEE)

| Documento | Foco | Decisiones que ancla |
|---|---|---|
| [`HANDOFF-track-b-2026-05-13.md`](investigaciones/HANDOFF-track-b-2026-05-13.md) | Ledger maestro D2-D30: stack training (Ultralytics 8.4.x, uv, Vast.ai), pipeline ONNX opset 11, runtime Nano JP 4.6.1, NMS tri-path V0/V1/V2, persistencia HF Hub, auto-destroy. | Todo Track B. |
| [`2026-05-14-training-headless-uv-nms-maxwell.md`](investigaciones/2026-05-14-training-headless-uv-nms-maxwell.md) | uv-en-notebook bajo `nbconvert`, `EfficientNMS_TRT` confirmado en binary JP 4.6.1 (fix #1538), `CommitScheduler` + heartbeat patrón TRAINCHECK, headless via `nbconvert + tmux`. | 8 decisiones aplicadas al notebook (uv invocation, NMS tri-path, signal handlers, manifest). |

La carpeta `investigaciones/` reemplaza al CONSOLIDADO previo (su contenido fusionado quedó en el HANDOFF como sección histórica).

---

## 5. Quick start

### 5.1 Local (host x86, Windows/Linux) — dry-run y desarrollo

```powershell
# Python 3.10+ con uv (gestor por defecto)
uv sync                          # instala deps de pyproject.toml
# Variables de entorno requeridas:
#   HF_TOKEN              → HuggingFace, repo mitgar14/embebidos-3-models
#   ROBOFLOW_API_KEY      → workspace embebidos3
# Conviene tenerlas en .env (gitignored).
```

Sin GPU NVIDIA local, el notebook aborta intencionalmente en la celda 8 (assert CUDA). El flow §1-9 (env, config, stack, pre-flight, Roboflow cascada, validación, scheduler init, heartbeat init) se valida en local en ~2 min sin tocar Vast.ai ni HF Hub.

### 5.2 Training en Vast.ai (Sprint 1 phase C)

```bash
# 1. Provisionar instancia (template cuda-12.4.1-cudnn-devel-ubuntu22.04-py310, RTX 4090, 50 GB disk)
vastai create instance <OFFER_ID> --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 ...

# 2. Subir scripts + notebook
rsync -avz scripts/training/bootstrap.sh notebooks/train_track_b_yolov8.ipynb root@<INSTANCE_IP>:/workspace/

# 3. En la instancia: ejecutar bootstrap + lanzar training en tmux
ssh root@<INSTANCE_IP>
bash /workspace/bootstrap.sh
tmux new -s training
export HF_TOKEN=... ROBOFLOW_API_KEY=...
jupyter nbconvert --execute --inplace \
    --ExecutePreprocessor.kernel_name=trackb \
    /workspace/train_track_b_yolov8.ipynb
# Ctrl+B D para desconectar tmux; la sesión sobrevive SSH dropout.
```

Outputs producidos por el notebook (subidos a HF Hub cada 10 min via `CommitScheduler`):
- `runs/detect/train/weights/best.pt`
- `exports/best.onnx` (opset 11, `nms=False`)
- `manifests/{eval_summary,gate3_onnx,gate4_polygraphy,manifest}.json`
- `runs/heartbeat.jsonl`

La instancia Vast.ai se auto-destruye al final del notebook (cell 28) o por idle del cron watchdog (GPU <5% durante 30 min).

### 5.3 Compilar engine TensorRT FP16 en el Nano

```bash
# scp del best.onnx al Nano
scp mitgar14/embebidos-3-models/exports/best.onnx jetson@nano:/home/jetson/models/

# En el Jetson Nano:
sudo systemctl stop lightdm                              # libera RAM de X11
sudo sh -c "sync && echo 3 > /proc/sys/vm/drop_caches"
export PATH=$PATH:/usr/src/tensorrt/bin
trtexec --onnx=/home/jetson/models/best.onnx \
        --saveEngine=/home/jetson/models/yolov8n_waste_fp16.engine \
        --fp16 --workspace=1024 --verbose
```

`--workspace=1024` (1 GiB) seguro: ultralytics issue #14751 reporta `Killed` por OOM con workspace mayor en Nano 4 GB. Tiempo de compilación: 15-45 min.

### 5.4 NMS tri-path en el Nano

| Variante | Cuándo usarla | Cómo |
|---|---|---|
| **V0** (default) | Compatibilidad garantizada, sin riesgo. | Decodificar la salida del engine + `cv2.dnn.NMSBoxes` en CPU. Overhead 3-5 ms/frame. |
| **V1** (smoke test) | Validar si `EfficientNMS_TRT` funciona en este Nano específico. | Re-export ONNX con `nms=True, format=onnx` (Ultralytics inyecta nodo `EfficientNMS_TRT`), recompilar engine. El plugin está presente en el binary JP 4.6.1 con el fix del issue #1538 (commit `3235cc2`, jul-2021). |
| **V2** (fallback) | Si V1 falla. | `BatchedNMSDynamic_TRT` — plugin estable del mismo binary. |

### 5.5 Pipeline dashboard end-to-end (compilación + tracking + UI)

A partir del 2026-05-16 todo el ciclo de vida del engine se opera desde el dashboard web. Reemplaza el flujo manual `scp + trtexec` de §5.3.

**Componentes en el Nano** (instalados por `scripts/install/nano_install_systemd.sh`):

- `embebidos3-server.service` — FastAPI/WS (uvicorn foreground, Type=simple, Restart=on-failure). Expone WS de inferencia + endpoints de gestión.
- `embebidos3-builder@<jobid>.service` — templated oneshot que ejecuta `nano_build_engine.sh` (12 fases: lock → download manifest → download ONNX → stop server → prep Nano (swap 8GB, lightdm off) → trtexec → validate → swap atómico → restore → cleanup). Tarda 15-40 min en Maxwell sm_53.
- `/etc/sudoers.d/embebidos3` — 14 reglas granulares NOPASSWD con paths absolutos (cuidado: `fallocate` está en `/usr/bin/`, no `/sbin/`).
- `/run/embebidos3/job.json` — estado del job activo con heartbeat + PID + cmdline check (recovery automático al startup).
- `engines/best_fp16.engine` + `engines/best_fp16.engine.meta.json` — el `.meta.json` registra `hf_revision`, `onnx_sha256`, `engine_sha256`, `trtexec_args`, `validation`. Sin meta el sistema reporta `state=no_model`.

**Endpoints HTTP** (consumidos por `scripts/dashboard/modelo.js`):

| Endpoint | Función |
|---|---|
| `GET /model/state` | `no_model \| ready \| degraded \| building` + `engine_binary_present` |
| `POST /model/build {force, workspace_mb?}` | Lanza job (202 + `job_id`) |
| `POST /model/check-updates` | Compara `hf_revision` Y `onnx_sha256` (LFS oid de HF, sin descargar) |
| `POST /model/rollback` | Swap inverso con `.previous` |
| `POST /model/adopt` | Registra meta retroactivo para engine huérfano (binario sin meta) |
| `GET /jobs/<id>/logs` | SSE stream de logs en vivo |
| `DELETE /jobs/<id>` | Cancela vía SIGTERM al builder |

**Dashboard local** (sin tocar el Nano):

```powershell
# Sirve scripts/dashboard/ en localhost:8001 + abre browser, apunta al Nano por Tailscale
uv run --with requests python scripts/launch_demo.py
# o sin abrir browser:
uv run --with requests python scripts/launch_demo.py --no-browser
```

Tab `modelo` muestra: hero card con CTA "descargar y compilar engine" si no hay modelo; banner "adoptar engine existente" si hay binario sin meta; tarjeta "Modelo activo" con metadata + badge `ADOPTADO` cuando aplica; logs SSE en vivo durante el build; cross-tab banner en pestaña `live` cuando hay build en curso.

**Skill para Claude Code**: `.claude/skills/embebidos3-nano/SKILL.md` documenta endpoints, paths del filesystem, sudoers, restricciones Py3.6, y motiva verificación del estado del Nano antes de cualquier cambio (evita incompatibilidades por suposiciones obsoletas).

---

## 6. Aportes para el informe IEEE

1. **YOLOv8n FP16 416×416 a 40 FPS en Nano B01 JP 4.6.1, superando la predicción de Nature 2024 Tabla 4 (30 FPS).** Resultado empírico reproducible.
2. **Validación empírica del fix `EfficientNMS_TRT` en el binary JP 4.6.1** (issue NVIDIA/TensorRT#1538, commit `3235cc2` jul-2021). La literatura dual-track previa asume el plugin roto en Maxwell — confirmamos lo contrario y dejamos los tres paths (V0/V1/V2) documentados.
3. **Comparativa imgsz 416 vs 640 en dataset waste custom.** Nature 2024 lo midió en COCO; nosotros sobre 20 493 imágenes propias con 3 clases desbalanceadas.
4. **Pipeline headless de training reproducible sobre cloud spot (Vast.ai)** con `CommitScheduler` + signal handlers + heartbeat TRAINCHECK-style + auto-destroy. Patrón replicable para otros proyectos académicos con presupuesto limitado.
5. **Ablación letterbox-vs-stretch en waste detection con aspect ratios mixtos.** Brecha documentada — opcional: generar Version 1-B-alt-stretch en Roboflow y comparar mAP@50.

---

## 7. Gotchas conocidos

- **`EfficientNMS_TRT` en Maxwell**: el issue NVIDIA/TensorRT#1538 está fixed en TRT 8.2.1.8 (commit `3235cc2`, jul-2021), incluido en JP 4.6.1. V0 (CPU NumPy) es default por compatibilidad; V1 y V2 quedan disponibles.
- **Roboflow SDK `location` bug** (verificado en ≥1.3.x): el argumento `location=` falla intermitentemente. Notebook implementa cascada de 3 estrategias (`location` directo → `download()` + `shutil.move` → 3 retries con backoff). NO setear env var `DATASET_DIRECTORY`.
- **`numpy<2.0` obligatorio**: ultralytics + onnx no funcionan con NumPy 2.x en este entorno. Notebook hace hard assert en cell 8.
- **`KillUserProcesses=no` en Vast.ai**: sin esta línea en `/etc/systemd/logind.conf`, `tmux` muere al cerrar la sesión SSH. `bootstrap.sh` la configura.
- **Padding=114 (Ultralytics LetterBox) vs Fit-black=0 (Roboflow)**: mismatch teórico no medido en literatura. Decisión documentada en HANDOFF §6.
- **Roboflow Versions inmutables y consumen créditos**: no regenerar Version 1-B sin necesidad.
- **Tailscale ARM64 + kernel 4.9-tegra**: bug DNS #14902. Workaround `--accept-dns=false --ssh` (HANDOFF D27).

---

## 8. Referencias citables principales

- Crasto, K. (2024). *Class Imbalance in Object Detection: An Experimental Diagnosis and Study of Mitigation Strategies*. arXiv:2403.07113.
- Bochkovskiy, A. et al. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*. arXiv:2004.10934.
- Chakraborty et al. (2025). *Half-core utilization rule on Jetson edge devices*. arXiv:2508.08430.
- Nature Scientific Reports (oct 2024). *Real-time waste detection on Jetson Nano* (DOI: 10.1038/s41598-024-74798-3).
- Yan, P. et al. (2025). *TRAINCHECK: practical training-time invariant checking for ML pipelines*. arXiv:2506.14813.
- Wang, A. et al. (2024). *YOLOv10: Real-Time End-to-End Object Detection*. arXiv:2405.14458.

Bibliografía completa con +30 fuentes adicionales en las tablas acumulativas del HANDOFF y de `2026-05-14-training-headless-uv-nms-maxwell.md`.
