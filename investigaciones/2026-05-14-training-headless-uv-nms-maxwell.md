# Investigación — Training headless + `uv` en notebook + NMS Maxwell `sm_53`

**Dominio:** notebook de training Track B en Vast.ai + post-processing en Jetson Nano B01
**Proyecto:** `embebidos-3` (clasificador residuos plastic/glass/paper, demo 2026-05-26)
**Ronda 1 inaugural** — sin rondas previas en este dominio. El HANDOFF principal vive en `investigaciones/HANDOFF-track-b-2026-05-13.md`.

---

## Resumen ejecutivo

Cuatro recomendaciones accionables que cierran esta ronda:

1. **`uv` desde notebook (R2):** invocar con ruta absoluta del intérprete del venv —
   `subprocess.run(["uv","pip","install","--python","/opt/venv/trackb/bin/python", "<paquete>"], check=True)`. Independiente del estado de `VIRTUAL_ENV`, del CWD y del modo de ejecución (interactivo vs `nbconvert`). Confirmado por docs oficiales Astral + comportamiento de `--python` aceptando rutas absolutas desde PR #3064 (2024-04).

2. **NMS en Jetson Nano (R3 + T1):** ruta de tres tiers.
   - **V0 producción:** `cv2.dnn.NMSBoxes` CPU — ya validado, 1-3 ms estimado, cero riesgo.
   - **V1 optimización (smoke test antes de producir):** `EfficientNMS_TRT` plugin embebido en ONNX vía `onnx-graphsurgeon`. **Verificación empírica SSH confirma que el plugin está registrado en el `libnvinfer_plugin.so.8.2.1` del JetPack 4.6.1**; el fix del bug Maxwell (`regsPerBlock < 65536`) commit `3235cc2` es de jul-2021 y el binario es de nov-2021, alta probabilidad de incluirlo. Probar con `trtexec --iterations=100` antes de adoptar.
   - **V2 si V1 falla:** `BatchedNMSDynamic_TRT` (confirmado funciona en `sm_53` por usuario `mokpo4550` en issue #1538).
   - **No tomar:** `torchvision.ops.nms` CUDA (torch/torchvision no instalados en system Python del Nano — requiere build manual aarch64, no vale la complejidad para 1-3 ms).

3. **Training headless persistente (R4):** mantener D18 — `jupyter nbconvert --execute --inplace --ExecutePreprocessor.kernel_name=trackb` dentro de `tmux new -s training`. Decoupla del SSH, preserva outputs incrustados (necesario para informe IEEE), permite re-attach. Verificar antes de provisionar que `KillUserProcesses=no` en `/etc/systemd/logind.conf` de la instance Vast.ai (default suele ser `no`, pero `=yes` mataría tmux también).

4. **`CommitScheduler` clean shutdown (R4 + T2):** `signal.SIGTERM/SIGINT` handler que llame a `sys.exit(0)` para disparar `atexit`; en `atexit` hacer `scheduler.trigger().result(timeout=120)` + `scheduler.stop()` síncronos. Argumentar a favor de `every=10` (no `every=5`) + `squash_history=True` por recomendación oficial HF Hub para repos con muchos commits. Tres papers (TRAINCHECK arXiv:2506.14813, TrainMover arXiv:2412.12636, CRIUgpu arXiv:2502.16631) refuerzan el patrón "checkpoint local → upload separado".

---

## Hallazgos críticos verificados empíricamente (SSH a Nano, 2026-05-14)

Antes de cualquier síntesis bibliográfica, dos verificaciones empíricas modifican el cuadro vs el HANDOFF:

### Discrepancia 1 — Versión JetPack real ≠ HANDOFF

| Fuente | JetPack | L4T | Fecha build |
|---|---|---|---|
| HANDOFF §1.2 | 4.6.5 | — | — |
| SSH (`/etc/nv_tegra_release`) | **4.6.1** | **R32.7.1** | **2022-02-19** |
| Notebook actual (cell-0) | 4.6.1 | — | — |

El notebook viejo tenía la versión correcta; el HANDOFF la rotuló mal. Implicación: ningún parche acumulado de 4.6.2-4.6.5 (CVE fixes, mejoras `nvidia-l4t-bootloader`, etc.) está en esta unidad. Para el plan de deploy esto es **neutral** porque TRT 8.2.1 y CUDA 10.2 son idénticos en toda la rama 4.6.x, pero conviene actualizar HANDOFF + memoria mnemon.

### Discrepancia 2 — Plugins NMS disponibles en el binario

Inspección runtime vía `tensorrt.get_plugin_registry()` sobre la Nano real:

```
TRT version: 8.2.1.8
Total plugins registrados: 46
NMS-related plugins (6):
  - NMS_TRT                v1
  - BatchedNMS_TRT         v1
  - BatchedNMSDynamic_TRT  v1
  - EfficientNMS_TRT       v1
  - EfficientNMS_ONNX_TRT  v1
  - EfficientNMS_TFTRT_TRT v1
```

`EfficientNMS_TRT` **está presente y registrado**. El bug del issue NVIDIA/TensorRT#1538 era runtime (`kSTATUS_SUCCESS` assertion en kernel CUDA por exceder `regsPerBlock`), no de registro del plugin. Que el plugin se cargue no garantiza que ejecute sin crash en `sm_53`, pero el `libnvinfer_plugin.so.8.2.1` (timestamp 2021-11-17) es posterior al commit del fix (`3235cc2`, 2021-07-01). Probabilidad alta de incluirlo. **Decisión:** se prueba en smoke test antes de adoptar, no se asume.

### Otros datos confirmados via SSH

- TensorRT `8.2.1.8` ✓, CUDA `10.2.300` ✓, cuDNN `8.2.1` ✓, OpenCV `4.1.1` ✓, Python `3.6.9` ✓
- `uv 0.11.14 (aarch64-unknown-linux-musl)` en `/home/jetson/.local/bin/uv` ✓
- `nvcc` en `/usr/local/cuda/bin/nvcc` (no en PATH default)
- **`torch` y `torchvision` no están instalados** en system Python 3.6.9 → ruta `torchvision.ops.nms` CUDA requiere build manual aarch64 (wheel oficial NVIDIA `torch==1.10.0a0` + compilar `torchvision` con CUDA backend desde fuente — proceso documentado en Qengineering pero no trivial). **Descartada por costo/beneficio.**

---

## Ronda 1 — 2026-05-14 (profundidad media)

### Track A — Agentes de research

#### R2 (research-web) — Patrón canónico `uv` dentro de notebook Jupyter

**Tabla comparativa de invocaciones (resumen del agente):**

| Invocación | ¿Toca el venv correcto? | Setup extra |
|---|---|---|
| `subprocess.run(["uv","pip","install","--python","/opt/venv/trackb/bin/python", X])` | **Sí, siempre.** Ruta absoluta independiente de `VIRTUAL_ENV` | Ninguno |
| `!uv pip install --python /opt/venv/trackb/bin/python X` | Sí (bang shell, mismo principio) | Ninguno |
| `!uv pip install X` (sin `--python`) | **Frágil.** Depende de `VIRTUAL_ENV` o `.venv` en CWD | Requiere `VIRTUAL_ENV` exportado antes del kernel |
| `!uv add X` (con `pyproject.toml`) | Solo si CWD tiene `pyproject.toml` y `UV_PROJECT_ENVIRONMENT=/opt/venv/trackb` | Requiere ambos |
| `%pip install --uv X` (magic) | **NO EXISTE.** Verificado por ausencia total en issues `ipython/ipykernel` | N/A |
| `%pip install X` (magic estándar) | Sí si `sys.executable` es el python del kernel; pero usa `pip` no `uv` | Requiere `pip` instalado en el venv (uv no lo instala por default) |

**Recomendación ganadora:** opción 1 (`subprocess.run` + `--python <ruta absoluta>`). Capture de stderr, control de excepciones, ruta determinística.

**Caveats clave:**
- Cuando `nbconvert --execute --ExecutePreprocessor.kernel_name=trackb` arranca el kernel, el `kernel.json` registrado vía `python -m ipykernel install --user --name trackb` apunta a `argv[0] = /opt/venv/trackb/bin/python`. En ese caso `sys.executable` resuelve correctamente. Pero si el kernel se registra vía `uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project` (variante recomendada por Astral docs), el `kernel.json` puede pasar `VIRTUAL_ENV` como env var en lugar de ruta absoluta, generando ambigüedad. **Para eliminar la duda, usar siempre la ruta absoluta explícita en el `--python`.**
- Issue `astral-sh/uv#15219` (nbconvert con `uv run --with` templates missing) **no aplica** a nuestro caso (invocamos nbconvert directamente con el kernel registrado por nombre, no como wrapper de uv run).

**Fuentes citadas por el agente:**
- [docs.astral.sh/uv/pip/environments](https://docs.astral.sh/uv/pip/environments/)
- [github.com/astral-sh/uv/issues/3060](https://github.com/astral-sh/uv/issues/3060) (cerrado, PR #3064, 2024-04)
- [github.com/astral-sh/uv/issues/17873](https://github.com/astral-sh/uv/issues/17873) (2026-02, jupyter no ve project venv en `uv run --with`)

---

#### R3 (research-code) — `EfficientNMS_TRT` y alternativas NMS en Maxwell `sm_53`

**Status del issue NVIDIA/TensorRT#1538:**

- Abierto 2021-10-07, cerrado por inactividad 2022-06-15 (sin reapertura).
- Causa raíz confirmada por @wraveane (NVIDIA contributor): bug en `tileSize = 1024` que excedía `regsPerBlock` en sm_53 / sm_61 (TX1/TX2/Nano).
- **Fix:** commit `3235cc2` (2021-07-01), `plugin/efficientNMSPlugin/efficientNMSInference.cu`. Reemplazó `tileSize = 1024` por `tileSize = param.numSelectedBoxes / 4` y añadió detección de `regsPerBlock < 65536` → fuerza `numSelectedBoxes = 2000` (vs 5000 para devices modernos).
- Fix está en rama `release/8.2` del repo TRT OSS.
- **Verificación binaria JetPack 4.6.1:** `libnvinfer_plugin.so.8.2.1` timestamp `2021-11-17`, posterior al fix de julio. **Plugin se carga** (verificado via `get_plugin_registry()`). Que ejecute sin crash en runtime no está confirmado públicamente — solo `franferraz98` reportó éxito en JetPack 4.6.1 bajo Docker.

**Tabla de alternativas NMS (compacta):**

| Opción | Decisión | Razón resumida |
|---|---|---|
| `cv2.dnn.NMSBoxes` CPU | **ADOPT V0** | Ya validado en el HANDOFF. Latencia ~1-3 ms estimada (no documentada con metodología pública). |
| `EfficientNMS_TRT` plugin embebido | **EXPERIMENTAL — high probability** | Plugin presente en binario JetPack 4.6.1. Probar con smoke test 100 iteraciones antes de adoptar. |
| `BatchedNMSDynamic_TRT` | **EXPERIMENTAL** fallback | Confirmado funciona en sm_53 (usuario `mokpo4550`, JP 4.4 + JP 4.6). Deprecated en TRT 9 pero plenamente soportado en 8.2. |
| `BatchedNMS_TRT` (static shapes) | SKIP | Sin ventaja sobre la variante dynamic para nuestro caso. |
| `torchvision.ops.nms` CUDA | **SKIP** | torch/torchvision no instalados; build aarch64 con CUDA backend desde fuente no documentado para JP 4.6.1 con Python 3.6 sin riesgo. |
| Numba `@cuda.jit` NMS kernel | SKIP | No hay implementación de referencia validada para sm_53. |
| Plugin custom community (jkjung-avt, marcoslucianops) | SKIP | jkjung-avt: YOLOv3/v4 only, repo sin commits desde 2021-10. DeepStream-Yolo: requiere DeepStream 6.3+ que necesita JetPack 5.x. |

**Patrón de adopción experimental para `EfficientNMS_TRT`:**

1. Export ONNX con `nms=False` (como ya hace el flujo actual).
2. Pre-build: injectar nodo `EfficientNMS_TRT` con `onnx-graphsurgeon` (referencia: `WongKinYiu/yolov7/utils/add_nms.py`).
3. `trtexec --onnx=yolov8n_with_nms.onnx --saveEngine=test.engine --fp16 --workspace=1024` en la Nano.
4. `trtexec --loadEngine=test.engine --iterations=100`. Si crashea → fallback a V0/V2.

**Plan B si V1 falla** (recompilar `nvinfer_plugin` OSS): 1-2 h en Nano, instrucciones documentadas por @wraveane. No es prioritario; primero probamos el binario stock.

---

#### T1 (research-academic) — Papers acceleration NMS en edge devices

Cinco papers relevantes 2023-2025:

| # | Autor / año | Identificador | Aporte |
|---|---|---|---|
| 1 | Si, Sun, Zhang et al., 2024 | **arXiv:2407.00618** | QSI-NMS (6,2× speedup, 0,1 % drop mAP), eQSI-NMS (10,7×), **BOE-NMS (5,1× sin pérdida mAP)** — algorítmicos puros, no requieren hardware especializado. Primer benchmark NMS especializado (NMS-Bench) sobre YOLOv8-N / MS COCO. |
| 2 | Shao et al., 2025 | Semantic Scholar (sin arXiv) | FPGA + NPU dedicated NMS module, 149× vs CPU Intel E5. No aplica a Maxwell pero contextual. |
| 3 | Chen et al., 2023 | O3NMS (FPGA) | 2,51× speedup vs SoA NMS hardware. Contextual. |
| 4 | Alqahtani, Cheema, Toosi, 2024 | **arXiv:2409.16808** | Benchmark edge devices YOLOv8/EfficientDet/SSD sobre Raspberry Pi 3-5, Coral TPU, **Jetson Orin Nano** (Ampere sm_87, **no Maxwell**). YOLOv8n en Orin Nano = 16 ms total. Jetson = menor energía por request. |
| 5 | Wang et al., 2024 | **arXiv:2405.14458** YOLOv10 | **NMS-free** vía dual assignment consistente. YOLOv10-S = 1,8× más rápido que RT-DETR-R18 a igual AP. |

**Lectura para nuestro caso:**

- Ningún paper benchmarkea `EfficientNMS_TRT` en `sm_53` específicamente. Gap real de literatura — toda la evidencia es de issues / foros.
- **BOE-NMS (5,1× speedup sin pérdida)** es la propuesta algorítmica más atractiva si decidimos no usar plugin TRT. Implementación O(n) con localidad espacial; podría replicarse en numpy/numba sobre ARM Cortex-A57.
- **YOLOv10 NMS-free** es interesante pero implica cambiar D2 del HANDOFF (YOLOv8n → YOLOv10n). Fuera de scope para esta ronda; sugerir como tema R2 si se hace una segunda ronda.

---

#### T2 (research-academic) — Reliability long-running training jobs

Cuatro papers que arman el patrón canónico actual:

| # | Autor / año | Identificador | Aporte aplicable |
|---|---|---|---|
| 1 | Jiang et al. (U. Michigan), 2025 | **arXiv:2506.14813** **TRAINCHECK** | Infiere invariantes (norma gradientes, parámetros, activaciones) automáticamente para detectar silent errors dentro de una iteración. **Detectó 18/20 errores reales + 6 bugs desconocidos**. Aplicable directo a nuestro watchdog. |
| 2 | Lao et al. (Harvard/UT/Alibaba), 2024 | **arXiv:2412.12636** TrainMover | Resilient runtime con shadow iterations. Downtime de segundos durante migración, 99 % eficiencia. Pattern de "checkpoint local primero, upload separado". |
| 3 | Kokolis et al. (Meta), 2024 | **arXiv:2410.21680** | Análisis 11 meses, 4 M jobs, 150 M GPU-hours A100. Taxonomía fallos + métrica Effective Training Time Ratio. Contextual, no directly accionable. |
| 4 | Stoyanov et al. (Google/Oxford/NVIDIA), 2025 | **arXiv:2502.16631** CRIUgpu | Transparent GPU checkpointing vía CUDA driver. Complemento al patrón graceful shutdown. |

**Patrón TRAINCHECK-style aplicado a Vast.ai (mínimo viable):**

```python
# Heartbeat-watchdog interno (corre como thread daemon en el notebook)
import threading, time, os

HEARTBEAT_FILE = "/workspace/embebidos-3/.heartbeat"
HEARTBEAT_INTERVAL = 60     # log cada 60 s
WATCHDOG_TIMEOUT = 600      # si no avanza en 10 min, alarma

def heartbeat_writer(get_metrics_fn):
    """Thread daemon que escribe métricas a archivo."""
    while True:
        m = get_metrics_fn()  # {"epoch": N, "gradient_norm": x, "loss": y}
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(f"{time.time()}\t{m}\n")
        time.sleep(HEARTBEAT_INTERVAL)

def watchdog_checker():
    """Otro thread que lee el heartbeat y alarma si no avanza."""
    last_mtime = 0
    while True:
        time.sleep(WATCHDOG_TIMEOUT // 2)
        try:
            mtime = os.path.getmtime(HEARTBEAT_FILE)
            if mtime == last_mtime:
                # No avanzó: log + posible kill
                with open("/workspace/embebidos-3/.stall_alert", "a") as f:
                    f.write(f"{time.time()}\tno_progress_for_{WATCHDOG_TIMEOUT}s\n")
            last_mtime = mtime
        except FileNotFoundError:
            pass
```

Esto se acopla al callback `on_train_epoch_end` de Ultralytics que ya está en el notebook viejo. La novedad académica es **leer gradient_norm explícitamente** (no solo loss), porque silent stalls pueden mantener loss decreciente artificialmente mientras los gradientes ya colapsaron.

**Patrón clean shutdown CommitScheduler (T2 + R4):**

```python
import atexit, signal, sys, logging
from huggingface_hub import CommitScheduler

scheduler = CommitScheduler(
    repo_id="mitgar14/embebidos-3-models",
    folder_path="/workspace/embebidos-3/runs",
    every=10,                # 10 min, NO 5 — HF recomienda menos commits
    squash_history=True,     # recomendación oficial HF para repos con muchos commits
    token=os.environ["HF_TOKEN"],
    private=True,
)

def _final_commit():
    try:
        logging.info("atexit: trigger final + stop")
        scheduler.trigger().result(timeout=180)  # bloquea hasta 3 min
        scheduler.stop()
    except Exception as e:
        logging.error(f"atexit commit final: {e}")

def _on_signal(signum, frame):
    logging.warning(f"signal {signum} → sys.exit(0) para disparar atexit")
    sys.exit(0)

atexit.register(_final_commit)
signal.signal(signal.SIGTERM, _on_signal)
signal.signal(signal.SIGINT, _on_signal)
```

**Por qué funciona:** la clase `CommitScheduler` ya registra `atexit.register(self._push_to_hub)` en su `__init__`, pero `atexit` **no se dispara con SIGTERM por default**. El handler que llama a `sys.exit(0)` convierte SIGTERM en "salida normal" → dispara `atexit` → ejecuta `_final_commit` y el push interno. Con `timeout=180` evitamos quedar colgados si HF está caído.

---

### Track B — Búsqueda ampliada

#### Fase 1: `discover.py` (3 queries sobre Exa Search)

**Q1 (R2):** 13 fuentes, dominios docs.astral.sh, github.com, discourse.jupyter.org, pydevtools.com. Hallazgos top:
- `astral-sh/uv#17873` "Jupyter does not have access to the project's venv" (2026-02) — caso del Astral guide.
- `astral-sh/uv#15219` "nbconvert with uv run and `--with` missing templates" (afecta solo a `uv run nbconvert`, no a nuestro caso).
- `jupyterlab/jupyterlab#17375` "%pip magic doesn't work when installed using 'uv tool install jupyterlab'" — confirma que el magic `%pip` está acoplado al método de instalación de jupyterlab.
- `bluss/pyproject-local-kernel` FAQ: prefijar kernel launch con `uv run` permite usar el venv del proyecto. Relevante si quisiéramos eliminar la instalación explícita de ipykernel.

**Q2 (R3):** 15 fuentes, dominios developer.nvidia.com, forums.developer.nvidia.com, github.com (NVIDIA/TensorRT). Hallazgos top:
- Cita verbatim del comment en issue #1538: *"It seems that on Release 8.2.1, the issue has been fixed."* — confirma la hipótesis del fix en binario JetPack 4.6.x.
- `NVIDIA/TensorRT/plugin/efficientNMSPlugin/README.md` (main): *"`EfficientNMS_TRT` is deprecated since TensorRT 10.0"* — válido para TRT 10, irrelevante a 8.2.
- PR #3920 "New Plugin EfficientNMSX" — variante con index output. No aplica a nuestro caso (solo necesitamos boxes+scores+classes).
- `triple-Mu/YOLOv8-TensorRT` — repo más activo de YOLOv8 + TRT con plugin NMS embebido. Patrón replicable.
- `Qengineering/YoloV8-TensorRT-Jetson_Nano` — 46 stars, C++ 92 %. NMS en CPU según comentarios.

**Q3 (R4):** 15 fuentes, dominios blog.melashri.net, dev.to, github.com (jupyter/nbconvert), medium.com (Nexumo + AI Platform). Hallazgos top:
- Mohamed Elashri "Keep Jupyter Notebooks Running" — tutorial canónico `tmux + jupyter + SSH tunnel`. Confirma D18.
- dev.to "Why Your Deep Learning Job Dies After SSH Logout" (2026-02-16) — explicación SIGHUP + recomendación tmux/nohup/disown/systemd ordenada por madurez. **Alerta:** `KillUserProcesses=yes` en `logind.conf` puede matar tmux. Verificar en Vast.ai.
- `jupyter/nbconvert#1436` "Headless remote execution of notebooks without download" — confirma el caso de uso del HANDOFF.
- `SwissDataScienceCenter/renku` commit `f08bdc1` — documentación oficial de Renku sobre tmux para long-running sessions. Reproducible.

#### Fase 1b: MCP YouTube (3 searches paralelas)

900 unidades de quota consumidas (3 searches `include_long_tail=true` + 3 retries simplificados). Resultados:

- **R2 uv:** top picks Changelog "Charlie Marsh on Rust + Python tooling" (`0wmz6RyVoFw`, 98:47), **BugBytes "uv - Python package and project management"** (`igWlYl3asKw`, 27:50, chapters indexados — único con cobertura específica de `uv pip install` desde notebook). Transcript del segmento minuto 10-18 confirma el setup estándar pero **no aporta info adicional al reporte del agente research-web** (cobertura redundante). Cerrado.
- **R3 NMS Maxwell:** top hit `MnaohuzEuhA` JetsonHacks "YOLO speed test" — cubre **Orin Nano Super / AGX Orin / AGX Thor**, NO Jetson Nano B01 Maxwell. **Skip** — Maxwell `sm_53` es legacy y los creadores que lo cubrirían (Qengineering, jkjung-avt) no tienen canal YouTube activo. Pool agotado.
- **R4 papermill / headless:** top picks PyData Chicago "Running Notebooks in Production? Blessing or Curse?" (`ywL9egN1Iyk`, Eduardo Blancas, fundador Ploomber, 37:18), PyData LA "Data and ETL with Notebooks in Papermill" (Matthew Seal, autor papermill, 38:58). **Transcript de Blancas:** los primeros 22 min son intro técnica + setup, baja densidad. La señal técnica clave (patterns concretos production) está en último tercio pero el cascade extracter no alcanzó timestamps específicos en una sola página. **Decisión:** lectura cancelada para no quemar contexto. El blog post de Nexumo (Track B Fase 2) cubre el mismo material en formato más denso.

#### Fase 2: Exa `crawling_exa` profundo (4 URLs)

- **Mohamed Elashri tmux+jupyter** (blog.melashri.net) — pattern paso-a-paso: `tmux new -s training`, `jupyter notebook --no-browser --port=8888`, `Ctrl+B D`, `ssh -L 8888:localhost:8888 user@host`. Confirma D18 verbatim.
- **Nexumo "7 Papermill & nbclient Tricks" (2025-11-30)** — siete patrones accionables:
  1. Parameterize + `kernel_name="python3"` pinned via `pm.execute_notebook(... parameters=..., kernel_name=...)`.
  2. `nbclient.NotebookClient(timeout=120, allow_errors=False)` — fail fast.
  3. Structured logging con run_id + JSON.
  4. Artifacts a run folder con paths deterministas (`Path(f"out/runs/{RUN_DATE}_{COUNTRY}")`).
  5. Cache fingerprint con `hashlib.sha1(json.dumps(params, sort_keys=True))`.
  6. Fan-out concurrente con `ThreadPoolExecutor(max_workers=N)`.
  7. CI/CD smoke notebook con `papermill` + GitHub Actions.
- **dev.to (2026-02-16) "Why Your DL Job Dies After SSH Logout"** — explicación SIGHUP, jerarquía de soluciones (tmux > nohup > disown > systemd), alerta sobre `KillUserProcesses=yes`.
- **docs.astral.sh oficial "Using uv with Jupyter"** (Charlie Marsh autor, 2025-05-18) — patrón recomendado: `uv run --with jupyter jupyter lab` + crear kernel con `uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=project`. **Caveat oficial:** *"Though `uv run --with jupyter` runs in an isolated environment, within the notebook itself, `!uv add` and related commands will modify the project's environment, even without a kernel. However, since the Jupyter server is the 'active' environment, `!uv pip install` will install packages into Jupyter's environment, not the project environment."* Refuerza por qué la ruta absoluta `--python /opt/venv/trackb/bin/python` es la única opción robusta.

---

## Decisiones para el notebook nuevo (siguiente paso fuera de esta ronda)

Estas decisiones no se ejecutan en esta ronda — se documentan acá como entrada al refactor posterior:

| # | Área | Decisión | Razón |
|---|---|---|---|
| 1 | uv en celdas | `subprocess.run(["uv","pip","install","--python","/opt/venv/trackb/bin/python", *pkgs], check=True)` | R2 ganador; docs.astral.sh oficial. |
| 2 | NMS Nano | V0 `cv2.dnn.NMSBoxes` CPU como default + sección "smoke test V1 EfficientNMS_TRT" como célula opcional con `trtexec --iterations=100` validation antes de adoptar. | R3 + SSH verificación. |
| 3 | Persistencia | `CommitScheduler(every=10, squash_history=True)` + signal handler `SIGTERM/SIGINT → sys.exit(0) → atexit → trigger().result(timeout=180) + stop()`. | T2 + R4. |
| 4 | Heartbeat | Thread daemon TRAINCHECK-style logueando `epoch + gradient_norm + loss` cada 60 s a `/workspace/embebidos-3/.heartbeat`. | T2 TRAINCHECK pattern. |
| 5 | Ejecución headless | Mantener D18 (`jupyter nbconvert --execute --inplace` dentro de `tmux new -s training`). | Confirmado por 3 fuentes independientes. |
| 6 | Pre-flight | Antes del training, verificar `cat /etc/systemd/logind.conf | grep KillUserProcesses` y abortar si es `=yes`. | dev.to alerta. |
| 7 | Failure budget | `nbclient.NotebookClient(timeout=120, allow_errors=False)` solo si vamos a script-mode futuro; para nbconvert directo `--ExecutePreprocessor.allow_errors=False`. | Nexumo trick #2. |
| 8 | Manifest | Reemplazar campo `jetpack: 4.6.1` (no 4.6.5) en cualquier output. | SSH verification. |

---

## Discrepancias detectadas con HANDOFF (a corregir en pasada posterior)

| # | Discrepancia | Evidencia | Acción sugerida |
|---|---|---|---|
| Δ1 | HANDOFF §1.2 dice JetPack `4.6.5` | SSH: L4T R32.7.1 = JetPack `4.6.1` | Corregir HANDOFF §1.2 + reemplazar memoria mnemon que lo afirme. |
| Δ2 | HANDOFF D26 dice `roboflow-python >=1.1.27` tiene fix oficial del bug `location` | Notebook viejo cell-8 documenta empíricamente que `v1.3.9` (2026-05-07, SHA `1e4cbc04`) sigue sin fix. NO investigado en esta ronda (R1 excluido por usuario). | Marcar como **gap abierto** en HANDOFF D26 hasta investigarse. Mantener cascada workaround del cell-10 del notebook viejo como seguro. |
| Δ3 | HANDOFF + memoria mnemon `4aa68e6a` dicen `EfficientNMS_TRT` "está roto en Maxwell" | Empíricamente: plugin presente en binario JP 4.6.1 (TRT 8.2.1.8). Fix `3235cc2` de jul-2021 precede al binario nov-2021. Probabilidad alta de funcionar. | Reescribir memoria mnemon como "fix incluido en binario JP 4.6.x con alta probabilidad; requiere smoke test runtime para confirmar". |
| Δ4 | HANDOFF §1.2 indica `torch` instalado en Nano | Empíricamente: `ModuleNotFoundError: No module named 'torch'` en system Python 3.6.9 | Aclarar en HANDOFF: torch NO viene en JetPack base, requiere wheel oficial NVIDIA + install manual. Para nuestro deploy (TRT engine, NO torch runtime), no es bloqueante. |

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-14 | medio | uv en notebook bajo nbconvert + NMS Maxwell `sm_53` alternativas + headless training persistente + CommitScheduler clean shutdown. SSH verificación empírica. |

---

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | Using uv with Jupyter (Astral) | https://docs.astral.sh/uv/guides/integration/jupyter/ | doc oficial | 1 |
| 2 | Using Python environments — uv pip `--python` | https://docs.astral.sh/uv/pip/environments/ | doc oficial | 1 |
| 3 | astral-sh/uv#3060 — Allow `--python` to take venv location | https://github.com/astral-sh/uv/issues/3060 | issue GH | 1 |
| 4 | astral-sh/uv#15219 — nbconvert with `uv run` missing templates | https://github.com/astral-sh/uv/issues/15219 | issue GH | 1 |
| 5 | astral-sh/uv#17873 — Jupyter no ve project venv | https://github.com/astral-sh/uv/issues/17873 | issue GH | 1 |
| 6 | jupyterlab/jupyterlab#17375 — `%pip` no funciona con `uv tool install jupyterlab` | https://github.com/jupyterlab/jupyterlab/issues/17375 | issue GH | 1 |
| 7 | NVIDIA/TensorRT#1538 — EfficientNMS_TRT not working on Jetson Nano | https://github.com/NVIDIA/TensorRT/issues/1538 | issue GH | 1 |
| 8 | NVIDIA/TensorRT commit `3235cc2` — fix EfficientNMS sm_53 regs | https://github.com/NVIDIA/TensorRT/commit/3235cc2ffc04c7819482fe6a512d504843242ad8 | commit GH | 1 |
| 9 | NVIDIA/TensorRT/plugin/efficientNMSPlugin README | https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin | doc OSS | 1 |
| 10 | Qengineering/YoloV8-TensorRT-Jetson_Nano | https://github.com/Qengineering/YoloV8-TensorRT-Jetson_Nano | repo | 1 |
| 11 | triple-Mu/YOLOv8-TensorRT | https://github.com/triple-Mu/YOLOv8-TensorRT | repo | 1 |
| 12 | huggingface_hub `_commit_scheduler.py` source | https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/_commit_scheduler.py | código fuente | 1 |
| 13 | HF Hub: rate limits | https://huggingface.co/docs/hub/rate-limits | doc oficial | 1 |
| 14 | HF Hub: storage limits + `squash_history` | https://huggingface.co/docs/hub/storage-limits | doc oficial | 1 |
| 15 | Si et al. 2024 — Accelerating NMS Graph Theory (QSI/BOE-NMS) | https://arxiv.org/abs/2407.00618 | arXiv | 1 |
| 16 | Alqahtani et al. 2024 — Benchmark edge devices YOLOv8/SSD | https://arxiv.org/abs/2409.16808 | arXiv | 1 |
| 17 | Wang et al. 2024 — YOLOv10 NMS-free | https://arxiv.org/abs/2405.14458 | arXiv | 1 |
| 18 | Jiang et al. 2025 — TRAINCHECK silent errors | https://arxiv.org/abs/2506.14813 | arXiv | 1 |
| 19 | Lao et al. 2024 — TrainMover resilient runtime | https://arxiv.org/abs/2412.12636 | arXiv | 1 |
| 20 | Kokolis et al. (Meta) 2024 — Reliability ML clusters A100 | https://arxiv.org/abs/2410.21680 | arXiv | 1 |
| 21 | Stoyanov et al. 2025 — CRIUgpu transparent GPU checkpointing | https://arxiv.org/abs/2502.16631 | arXiv | 1 |
| 22 | Mohamed Elashri — Keep Jupyter Notebooks Running (tmux pattern) | https://blog.melashri.net/posts/tmux-jupyter/ | blog | 1 |
| 23 | Nexumo (2025-11-30) — 7 Papermill & nbclient Tricks | https://medium.com/@Nexumo_/7-papermill-nbclient-tricks-for-prod-ready-notebooks-dedc88d1455d | medium | 1 |
| 24 | dev.to (2026-02-16) — Why Your DL Job Dies After SSH Logout | https://dev.to/ajitkumar/why-your-deep-learning-job-dies-after-ssh-logout-a-practical-guide-to-persistent-linux-sessions-52ml | blog | 1 |
| 25 | discourse.jupyter.org — Using Jupyter with uv | https://discourse.jupyter.org/t/using-jupyter-with-uv/29207 | foro | 1 |
| 26 | pydevtools.com — How to Run a Jupyter Notebook with uv | https://pydevtools.com/handbook/how-to/jupyter-notebook-with-uv/ | tutorial | 1 |
| 27 | forums.developer.nvidia.com — EfficientNMS plugin to TRT engine | https://forums.developer.nvidia.com/t/efficient-nms-plugin-to-tensorrt-engine-at-runtime/190344 | foro | 1 |
| 28 | jupyter/nbconvert#1436 — Headless remote execution | https://github.com/jupyter/nbconvert/issues/1436 | issue GH | 1 |
| 29 | YouTube BugBytes — uv - Python package and project management | https://youtu.be/igWlYl3asKw | video (27:50) | 1 |
| 30 | YouTube PyData Chicago — Running Notebooks in Production? (Blancas) | https://youtu.be/ywL9egN1Iyk | video (37:18) | 1 |

---

## Próximos pasos sugeridos

1. **Reescribir `notebooks/train_track_b_yolov8.ipynb`** integrando las 8 decisiones de la sección "Decisiones para el notebook nuevo". Target: Vast.ai, no Kaggle/Colab.
2. **Corregir HANDOFF §1.2** (JetPack 4.6.1 ≠ 4.6.5) y D26 (Roboflow location bug status) en un commit aparte.
3. **Refresh memoria mnemon `4aa68e6a`** con el hallazgo empírico sobre `EfficientNMS_TRT` plugin presente en binario JP 4.6.1.
4. **(Opcional, ronda 2)** investigar Roboflow `location` bug en versiones >1.3.9 + alternativas (HF datasets mirror, presigned URL REST). Tópico R1 excluido en esta ronda.
5. **(Opcional, ronda 2)** evaluar YOLOv10n NMS-free como sustituto de YOLOv8n. Implica modificar D2 del HANDOFF.
