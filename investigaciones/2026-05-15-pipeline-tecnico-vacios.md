# Investigación — vacíos técnicos del pipeline dashboard → server → builder en Jetson Nano

**Dominio:** `embebidos-3` — pipeline ONNX → TRT FP16 orquestado por dashboard web con server FastAPI + builder oneshot systemd en Jetson Nano B01.
**Profundidad:** alto.
**Fecha:** 2026-05-15.
**Relacionado:**
- `docs/superpowers/specs/2026-05-15-dashboard-pipeline-request.md`
- `docs/superpowers/specs/2026-05-15-dashboard-pipeline-brainstorm-preguntas.md`
- `docs/superpowers/specs/2026-05-15-dashboard-redesign-design.md`

Stack constreñido: JetPack 4.6.1, L4T R32.7.1, Python 3.6.9, TensorRT 8.2.1.8, pycuda 2019.1.2, Maxwell sm_53, 4 GB RAM unificada.

---

## Resumen ejecutivo

Siete decisiones técnicas cerradas tras esta ronda. Cada una con evidencia primaria verificada (GitHub issues NVIDIA, NVIDIA DevForum con respuesta de moderador, docs oficiales POSIX/systemd, Triton inference server). YouTube/AAI aportó valor parcial: los canales empíricos relevantes (JetsonHacks, DustyNV, Edje Electronics) no surfacearon en queries específicas sobre `trtexec` OOM/workspace; los hallazgos de mayor precisión vinieron de issues GitHub y forums NVIDIA. No se activó AAI fallback (`aai_threshold` no se bajó).

1. **Hot-reload TRT engine in-process**: orden estricto de destrucción `stream → outputs → inputs → context → engine → runtime` con `cu_ctx` activo (pushed), seguido de `pop` y reload. Patrón validado por `triple-Mu/YOLOv8-TensorRT` C++ destructor y `jkjung-avt/tensorrt_demos` Python `HostDeviceMem.__del__`. Bug clásico (#1107) si se destruye con context popped.
2. **`trtexec` parsing + cancel + timeout**: TRT 8.2 emite formato `[HH:MM:SS] [I/V] [TRT] msg`; flag `--verbose` activa `[V]` con timing runner por capa. Sin signal handler → SIGKILL es la única vía para cancelar. El `.engine` se escribe al final (no en streaming) → SIGKILL no produce archivos corruptos. Patrón canónico: `timeout 40m trtexec ... --saveEngine=staging.engine`, luego `mv` atómico solo si exit code = 0.
3. **Atomic file swap**: `rename(2)` POSIX en ext4 es atómico y seguro mientras el server tiene el engine cargado, porque `deserialize_cuda_engine(f.read())` lee todo a RAM y cierra el fd. TRT 8.2 no usa `mmap` persistente del engine file (verificado en código fuente de `jkjung-avt` y `triple-Mu`). Triton confirma el patrón.
4. **`trtexec` memory en Nano**: `--workspace` NO es límite duro (confirmado por NVIDIA staff @zerollzeng en TRT issue #2679). Pre-build obligatorio: `sudo init 3` (libera ~1 GB), `swapoff/swapon`, `vm.swappiness=100`, `drop_caches`. Patrón comunidad: 8-20 GB de file swap en `/mnt/`, no zram (zram comparte RAM). Monitoreo con `tegrastats --interval 2000`. Exit code 137 = OOM-killer.
5. **systemd templated + sudoers wildcard**: `NOPASSWD: /bin/systemctl start embebidos3-builder@*.service` funciona con `fnmatch(3)`, pero el wildcard matchea espacios → riesgo de inyección. Solución limpia: wrapper script `/usr/local/bin/start-builder-job` con validación regex `^[A-Za-z0-9_-]{10,40}$` y `sudoers` apuntando al wrapper, no al `systemctl` directo. Polkit JS descartado: Ubuntu 18.04 trae `policykit-1 0.105` con backend `.pkla`, soporte `rules.d` JS es inconsistente.
6. **`huggingface_hub` en Python 3.6**: **no existe versión compatible**. Verificado en `setup.py` de v0.8.1, v0.9.1, v0.10.1, v0.11.0, v0.12.1: todas declaran `python_requires=">=3.7.0"`. Solución: API REST directa con `requests`. Endpoints clave: `GET /<repo>/resolve/<rev>/<file>` (download), `POST /api/models/<repo>/commit/main` con `encoding: base64` (upload sin LFS).
7. **Job recovery tras crash**: PID file + `fcntl.lockf` (POSIX advisory, se libera automático al morir el proceso) + heartbeat JSON con escritura atómica via `rename`. Detección de vivencia: `os.kill(pid, 0)` (no `psutil`). Watchdog: si `time.time() - heartbeat > STALE_SECS` → marcar STALLED. `RuntimeDirectory=embebidos3` en la unit systemd para `/var/run/embebidos3/` con permisos correctos al boot.

---

## Vacío 1 — Hot-reload de TRT engine en pycuda 2019.1.2 + TensorRT 8.2.1.8

### Decisión

Refactorizar `TRTWorker` para que el método `_load_engine(path)` sea reentrante. Mantener `cu_ctx` activo entre engines (no recrearlo). Orden estricto de destrucción con context activo:

1. `del stream`
2. `del outputs` y `del inputs` (objetos `HostDeviceMem` con `DeviceAllocation` interno)
3. `del context` (`IExecutionContext`)
4. `del engine` (`ICudaEngine`)
5. `del runtime` (opcional, se puede mantener)

`pycuda 2019.1.x` libera GPU memory automáticamente vía `DeviceAllocation.__del__` cuando pierde la última referencia. No hace falta `free()` explícito.

Bug crítico documentado en NVIDIA/TensorRT issue #1107: si se destruye `engine`/`context` con el contexto CUDA popped, el driver no puede liberar GPU memory y se obtiene `PyCUDA ERROR: The context stack was not empty upon module cleanup`. **Regla**: siempre `cu_ctx.push()` antes de destruir, `cu_ctx.pop()` después.

Para hot-swap sin downtime visible: doble-buffer del engine. Deserializar `engine_new` mientras `engine_old` sirve inferencias, esperar fence (todas las inferencias activas terminan), luego destruir el viejo. NVIDIA/TensorRT issue #1255 confirma que el `context` se puede reutilizar para múltiples `execute_v2` sin reinit.

### Snippet (Python 3.6 — para `nano_server.py`)

```python
import threading
import tensorrt as trt
import pycuda.driver as cuda

class TRTWorker(threading.Thread):
    def __init__(self, engine_path):
        super().__init__(daemon=True)
        self._engine_path = engine_path
        self._swap_path = None
        self._swap_event = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()  # protege inferencia vs swap
        # estado del engine activo (se reasigna en swap)
        self._engine = None
        self._context = None
        self._stream = None
        self._inputs = []
        self._outputs = []
        self._bindings = []

    def _load_engine(self, path):
        """Carga engine y bindings. Asume cu_ctx activo (pushed)."""
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        with open(path, 'rb') as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        stream = cuda.Stream()
        inputs, outputs, bindings = [], [], []
        for i in range(engine.num_bindings):
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(dev_mem))
            (inputs if engine.binding_is_input(i) else outputs).append(
                (host_mem, dev_mem))
        return runtime, engine, context, inputs, outputs, bindings, stream

    def _unload_engine(self):
        """Destruccion ordenada. Asume cu_ctx activo (pushed)."""
        del self._stream
        for host_mem, dev_mem in self._outputs:
            del dev_mem
            del host_mem
        for host_mem, dev_mem in self._inputs:
            del dev_mem
            del host_mem
        del self._context
        del self._engine
        # runtime se mantiene reutilizable

    def request_swap(self, new_path):
        with self._lock:
            self._swap_path = new_path
        self._swap_event.set()

    def run(self):
        cuda.init()
        cu_ctx = cuda.Device(0).make_context()
        try:
            cu_ctx.push()
            (self._runtime, self._engine, self._context,
             self._inputs, self._outputs, self._bindings,
             self._stream) = self._load_engine(self._engine_path)
            cu_ctx.pop()
            self._ready.set()

            while True:
                if self._swap_event.is_set():
                    self._swap_event.clear()
                    with self._lock:
                        new_path = self._swap_path
                    cu_ctx.push()
                    self._unload_engine()
                    (self._runtime, self._engine, self._context,
                     self._inputs, self._outputs, self._bindings,
                     self._stream) = self._load_engine(new_path)
                    cu_ctx.pop()
                # ... bucle normal de inferencia con cu_ctx.push/pop
        finally:
            try:
                cu_ctx.push()
                self._unload_engine()
                cu_ctx.pop()
            except Exception:
                pass
            cu_ctx.detach()
```

### Referencias

| Fuente | URL | Relevancia |
|---|---|---|
| NVIDIA/TensorRT issue #1107 | https://github.com/NVIDIA/TensorRT/issues/1107 | Confirma `pop` antes de destroy. Bug "context stack not empty" |
| NVIDIA/TensorRT issue #1255 | https://github.com/NVIDIA/TensorRT/issues/1255 | NVIDIA staff (@ttyio): "reuse the context, no need to create for every inference" |
| NVIDIA/TensorRT issue #1632 | https://github.com/NVIDIA/TensorRT/issues/1632 | Orden destrucción: context → engine → runtime |
| NVIDIA/TensorRT issue #3715 | https://github.com/NVIDIA/TensorRT/issues/3715 | Switch ICudaEngines: "build once, infer many, destroy/recreate para nuevo" |
| `jkjung-avt/tensorrt_demos` | `utils/yolo_with_plugins.py:110-122` | `HostDeviceMem.__del__` patrón canónico |
| `triple-Mu/YOLOv8-TensorRT` | `models/pycuda_api.py` + `csrc/` | Orden destrucción C++ confirmado |
| Stack Overflow (talonmies) | https://stackoverflow.com/questions/56124741 | `DeviceAllocation` libera con `__del__` en pycuda 2019.x |
| Triton model management | https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md | Modos EXPLICIT/POLL para hot-swap en producción |

---

## Vacío 2 — `trtexec` parsing de progreso + manejo de timeout/cancelación

### Decisión

TRT 8.2 emite líneas con prefijos `[HH:MM:SS] [severity] [source]`. Sin `--verbose`, las fases observables son:

```
[I] Finished parsing network model. Parse time: X.XXX           ← FASE 1: parsing
[I] [TRT] [MemUsageChange] TensorRT-managed allocation ...      ← marker memoria
[I] Engine built in XX.XX sec.                                  ← FASE 2: build completo
[I] Engine deserialized in X.XX sec.                            ← FASE 3: serializing
&&&& PASSED TensorRT.trtexec [TensorRT vXXXX]                   ← FASE 4: éxito
&&&& FAILED TensorRT.trtexec ...                                ← FASE 4: fallo
```

Con `--verbose` se obtienen líneas `[V] [TRT] --------------- Timing Runner: <layer>` por cada capa optimizada → granularidad fina del progreso (parseable para barra de progreso).

**Cancelación**: `trtexec` no instala signal handler propio. Durante autotuning (bucle CUDA bloqueante) ignora SIGTERM. Solo SIGKILL termina. El `.engine` se escribe **una sola vez al final** desde `IHostMemory` en RAM → SIGKILL nunca produce archivos parcialmente corruptos: o ya existe la versión vieja, o no existe nada.

**OOM-killer**: el padre recibe `WTERMSIG=9` → exit code 137. Distinguir de error TRT (exit 1 o 255). Verificar `/var/log/kern.log` por `Out of memory: Kill process`.

**Timeout** (40 min según decisión): usar `timeout --kill-after=30s 40m trtexec ...`. El `--kill-after=30s` da una ventana para SIGTERM (ignorada en este caso) antes del SIGKILL.

### Snippet bash (para `scripts/nano_build_engine.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

JOB_ID="$1"
ONNX_PATH="$2"
ENGINE_OUT="$3"
STAGING="${ENGINE_OUT}.building"
TIMEOUT_SEC=2400  # 40 min

cleanup() {
    local exit_code=$?
    rm -f "$STAGING"  # nunca debería existir, defensa
    if [[ $exit_code -ne 0 ]]; then
        echo "[BUILD] FAILED exit=$exit_code" >&2
    fi
}
trap cleanup EXIT

echo "[BUILD] $JOB_ID iniciando trtexec sobre $ONNX_PATH"

# tegrastats en background para monitoreo
tegrastats --interval 2000 > "/home/jetson/embebidos-3/logs/jobs/${JOB_ID}.tegrastats.log" &
TEGRA_PID=$!
trap "kill $TEGRA_PID 2>/dev/null || true; cleanup" EXIT

timeout --kill-after=30s "${TIMEOUT_SEC}s" \
    /usr/src/tensorrt/bin/trtexec \
        --onnx="$ONNX_PATH" \
        --saveEngine="$STAGING" \
        --fp16 \
        --workspace=1024 \
        --buildOnly \
        --verbose \
        2>&1 | tee "/home/jetson/embebidos-3/logs/jobs/${JOB_ID}.log" | \
    grep --line-buffered -E \
        "Finished parsing|Engine built|Engine deserialized|PASSED|FAILED|MemUsageChange|Timing Runner" >&2

BUILD_EXIT="${PIPESTATUS[0]}"
kill $TEGRA_PID 2>/dev/null || true

if [[ "$BUILD_EXIT" -eq 124 ]]; then
    echo "[BUILD] $JOB_ID timeout (40 min)" >&2
    exit 124
elif [[ "$BUILD_EXIT" -eq 137 ]]; then
    echo "[BUILD] $JOB_ID OOM-killed (exit 137)" >&2
    exit 137
elif [[ "$BUILD_EXIT" -ne 0 ]]; then
    echo "[BUILD] $JOB_ID falló exit=$BUILD_EXIT" >&2
    exit "$BUILD_EXIT"
fi

# Tamaño sanity check (engine YOLOv8n@416 FP16 ~12-13 MB)
SIZE=$(stat -c %s "$STAGING")
if [[ "$SIZE" -lt 1000000 ]]; then
    echo "[BUILD] $JOB_ID engine sospechosamente pequeño: $SIZE bytes" >&2
    exit 1
fi

# Swap atómico
mv "$STAGING" "$ENGINE_OUT"
echo "[BUILD] $JOB_ID OK -> $ENGINE_OUT ($SIZE bytes)"
```

### Snippet Python para parsear progreso desde el server

```python
import re, subprocess, time

PHASE_RE = re.compile(
    r'(Finished parsing|Engine built in|Engine deserialized|'
    r'PASSED|FAILED|MemUsageChange|Timing Runner)'
)

def parse_phase_pct(line):
    """Retorna progress_pct aproximado segun la linea observada."""
    if 'Finished parsing' in line:    return 15
    if 'MemUsageChange' in line:      return 20
    if 'Timing Runner' in line:       return 50  # incrementar contra count layers
    if 'Engine built in' in line:     return 90
    if 'Engine deserialized' in line: return 95
    if 'PASSED' in line:              return 100
    return None

# El job state se actualiza desde el wrapper Python del builder
# (no desde el script bash) para tener acceso directo al filesystem.
```

### Referencias

| Fuente | URL | Relevancia |
|---|---|---|
| NVIDIA/TensorRT issue #2478 | https://github.com/NVIDIA/TensorRT/issues/2478 | Logs reales TRT 8.5 formato `[V] Timing Runner` |
| NVIDIA/TensorRT issue #2350 | https://github.com/NVIDIA/TensorRT/issues/2350 | `--verbose` captura a log.txt vía `2>&1`, formato TRT 8.2.1 |
| NVIDIA/TensorRT issue #3281 | https://github.com/NVIDIA/TensorRT/issues/3281 | NVIDIA confirma: "CUDA kernel execution doesn't support interruption" → no cancel limpio |
| NVIDIA/TensorRT issue #3614 | https://github.com/NVIDIA/TensorRT/issues/3614 | NVIDIA explica: "trtexec sin response es esperable, usar --verbose" |
| NVIDIA/TensorRT issue #4258 | https://github.com/NVIDIA/TensorRT/issues/4258 | OOM error format real + exit code 137 |
| NVIDIA/TensorRT issue #4730 | https://github.com/NVIDIA/TensorRT/issues/4730 | `--timingCacheFile` para builds repetidos deterministas |
| Docs TRT 8 → 10 migration | https://docs.nvidia.com/deeplearning/tensorrt/latest/api/tensorrt-8x-to-10x-trtexec.html | `--workspace` (8.x) vs `--memPoolSize` (10.x) |

---

## Vacío 3 — Atomic file swap sobre TRT engine activamente cargado

### Decisión

**`mv` (= `rename(2)`) es seguro** mientras el server tiene el engine cargado. Mecanismo:

1. `runtime.deserialize_cuda_engine(open(path, 'rb').read())` lee todo el archivo a RAM y cierra el fd. TRT 8.2 NO usa `mmap` persistente del engine file (verificado en código fuente de `jkjung-avt/tensorrt_demos` y `triple-Mu/YOLOv8-TensorRT`).
2. Una vez cerrado el fd, el inode viejo persiste en disco hasta que el último link sea removido.
3. `rename(2)` POSIX en ext4 es atómica a nivel VFS: el nombre destino apunta al nuevo inode o al viejo, nunca a un estado intermedio (Open Group spec, `rename` 2018 ed.).
4. ext4 journal en modo `ordered`/`writeback` garantiza que el rename no pueda quedar en estado parcial tras crash.

Triton Inference Server confirma este patrón explícitamente en su docs de model management: los engine files pueden ser reemplazados con `mv`/`rename` mientras el modelo está cargado.

**Verificación práctica de no-mmap** en TRT 8.2 en el Nano: `lsof -p $(pgrep -f nano_server) | grep best_fp16` después de deserialize. Si el fd no aparece, no hay mmap persistente.

### Snippet (incorporado al bash del Vacío 2)

```bash
# El swap final es simplemente:
mv "$STAGING" "$ENGINE_OUT"
```

El server detecta el cambio via inotify o re-fetch periódico. Recomendación: el builder, al terminar el `mv`, hace `systemctl start embebidos3-server.service` que arranca con el engine nuevo (alineado con decisión Q9 "parar server durante compilación"). NO requiere inotify ni hot-swap in-process en este caso.

### Referencias

| Fuente | URL | Relevancia |
|---|---|---|
| POSIX rename(2) | https://pubs.opengroup.org/onlinepubs/9699919799/functions/renameat.html | Garantía atomicidad POSIX |
| Triton model management | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html | Engine files reemplazables con mv durante load |
| Triton server issue #5983 | https://github.com/triton-inference-server/server/issues/5983 | NVIDIA confirma "touch + load API triggers reload keeping previous model in background" |
| `triple-Mu/YOLOv8-TensorRT` | `models/pycuda_api.py` `__init_engine()` | `read_bytes()` carga a RAM, sin fd persistente |
| `jkjung-avt/tensorrt_demos` | `utils/yolo_with_plugins.py` `_load_engine()` | `with open()` cierra fd antes de retornar engine |

---

## Vacío 4 — `trtexec` memory footprint en Jetson Nano 4 GB con YOLOv8n@416 FP16

### Decisión

`--workspace` NO es límite duro. Confirmado por NVIDIA staff (@zerollzeng) en issue #2679: con `--workspace=500`, el consumo real puede llegar a ~1,5 GB por overhead de cuBLAS/cuDNN/timing cache. **Procedimiento obligatorio pre-build** en el Nano 4 GB (libera ~1,5-2 GB efectivos):

```bash
# 1. Stop desktop (libera ~700 MB - 1 GB)
sudo init 3   # equivalente a 'systemctl isolate multi-user.target'

# 2. Swap de disco (no zram - zram comparte la RAM scarce)
sudo systemctl disable nvzramconfig
sudo swapoff -a
sudo fallocate -l 8G /mnt/swap.img
sudo chmod 600 /mnt/swap.img
sudo mkswap /mnt/swap.img
sudo swapon /mnt/swap.img

# 3. Swappiness agresivo
sudo sysctl vm.swappiness=100

# 4. Drop caches
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 5. Monitorear durante build
tegrastats --interval 2000 >> /tmp/trt_build_mem.log &

# 6. Build
trtexec --onnx=... --saveEngine=staging.engine --fp16 --workspace=1024 --buildOnly --verbose
```

**Workspace recomendado**: empezar con `--workspace=512`. Si falla OOM, reducir a `--workspace=256` y reintentar. Si pasa OK, opcionalmente probar `--workspace=1024` para futuras builds más complejas. **Nuestro engine actual** (auditoría 2026-05-15) fue compilado con 512 → validado funcional. Recomendación final del spec: usar 512 como default, exponer como variable de entorno del builder.

**Exit codes** (importantes para distinguir causa raíz del fallo):
- `0`: éxito.
- `1` o `255`: error TRT propio (parse fallido, opset incompatible, etc).
- `124`: SIGTERM por `timeout` (40 min alcanzado).
- `137`: SIGKILL por OOM-killer o por `timeout --kill-after`.

### Snippet (incorporado al script del Vacío 2 — bloque pre-build separado)

```bash
prep_nano_for_build() {
    echo "[PREP] stop desktop..."
    sudo systemctl stop lightdm.service 2>/dev/null || true

    echo "[PREP] swap config..."
    sudo systemctl disable nvzramconfig 2>/dev/null || true
    sudo swapoff -a
    if [[ ! -f /mnt/swap.img ]]; then
        sudo fallocate -l 8G /mnt/swap.img
        sudo chmod 600 /mnt/swap.img
        sudo mkswap /mnt/swap.img
    fi
    sudo swapon /mnt/swap.img
    sudo sysctl vm.swappiness=100

    echo "[PREP] drop caches..."
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null

    free -h
}

restore_nano_after_build() {
    sudo systemctl start lightdm.service 2>/dev/null || true
    sudo sysctl vm.swappiness=60
}
```

### Referencias

| Fuente | URL | Relevancia |
|---|---|---|
| NVIDIA/TensorRT issue #2679 | https://github.com/NVIDIA/TensorRT/issues/2679 | NVIDIA confirma: `--workspace` no es límite duro, ~1,5 GB pico con 500 MB declarado |
| NVIDIA DevForum: TensorRT Build Killed | https://forums.developer.nvidia.com/t/tensorrt-build-killed/161504 | Caso OOM Nano 2 GB con `free -m` real; mod @AastaLLL confirma swap no es GPU-accessible |
| NVIDIA/TensorRT issue #1404 | https://github.com/NVIDIA/TensorRT/issues/1404 | "set workspace 2 GB y stop desktop/wm" - patrón comunidad Nano |
| NVIDIA DevForum: Headed → headless | https://forums.developer.nvidia.com/t/headed-mode-to-headless-mode/120388 | `sudo init 3` libera GUI RAM sin reflash |
| `arvcode/TensorRT_classifier_efficientNet` | https://github.com/arvcode/TensorRT_classifier_efficientNet/blob/main/jetson_nano_setup_instructions.md | Script comunidad con `fallocate 20G` + `swappiness=100` + `drop_caches` |
| NVIDIA DevForum: vm.swappiness | https://forums.developer.nvidia.com/t/changing-the-linux-kernel-vm-swappiness-parameter/173035 | Default 60, comunidad usa 100 para builds TRT |
| NVIDIA DevForum: Unified Memory trtexec | https://forums.developer.nvidia.com/t/unified-memory-management-for-trtexec/302346 | `--noDataTransfers` relevante en Tegra para perfiles correctos |

---

## Vacío 5 — systemd templated services + sudoers granular wildcard

### Decisión

**Wrapper script + sudoers wildcard sobre el wrapper** (no sobre `systemctl` directo). Razones:

- `man 5 sudoers` documenta que `*` usa `fnmatch(3)` sobre argumentos. `NOPASSWD: /bin/systemctl start embebidos3-builder@*.service` SÍ funciona técnicamente.
- **Trampa**: el wildcard también matchea espacios. Un job_id malicioso tipo `"abc; rm -rf /"` puede expandirse y inyectar argumentos. La mitigación canónica de Compass Security: validar input con regex en un wrapper antes de invocar `systemctl`.
- Polkit JS descartado: Ubuntu 18.04 trae `policykit-1 0.105` con backend `.pkla`. `rules.d/` con JavaScript (`RegExp`) requiere `mozjs` compilado, soporte inconsistente en L4T R32.7.1. No worth la fragilidad para el caso simple.

`User=jetson` aceptable para single-user académico. Si se requiere mayor segregación, crear `User=embebidos3` dedicado (no es prioritario).

### Snippets

**Wrapper** `/usr/local/bin/embebidos3-builder-launch`:
```bash
#!/usr/bin/env bash
# Lanzador validado de instancias templated del builder.
# Invocado como: sudo embebidos3-builder-launch <jobid>
set -euo pipefail

JOBID="${1:-}"
if [[ ! "$JOBID" =~ ^[A-Za-z0-9_-]{10,40}$ ]]; then
    echo "JOBID inválido: '$JOBID'" >&2
    exit 1
fi
exec /bin/systemctl start "embebidos3-builder@${JOBID}.service"
```

**Sudoers** `/etc/sudoers.d/embebidos3` (permisos `0440`, validar con `visudo -cf`):
```
jetson ALL=(root) NOPASSWD: /usr/local/bin/embebidos3-builder-launch *
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-builder@*.service
```

**Unit template** `/etc/systemd/system/embebidos3-builder@.service`:
```ini
[Unit]
Description=embebidos-3 TRT engine builder (job %i)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=jetson
Group=jetson
WorkingDirectory=/home/jetson/embebidos-3
EnvironmentFile=/etc/embebidos3/secrets.env
RuntimeDirectory=embebidos3
RuntimeDirectoryMode=0755
ExecStart=/home/jetson/embebidos-3/scripts/nano_build_engine.sh %i
TimeoutStartSec=2700
StandardOutput=append:/home/jetson/embebidos-3/logs/jobs/%i.systemd.log
StandardError=append:/home/jetson/embebidos-3/logs/jobs/%i.systemd.log
```

**Unit del server** `/etc/systemd/system/embebidos3-server.service`:
```ini
[Unit]
Description=embebidos-3 FastAPI/WS inference server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=jetson
Group=jetson
WorkingDirectory=/home/jetson/embebidos-3
EnvironmentFile=/etc/embebidos3/secrets.env
RuntimeDirectory=embebidos3
RuntimeDirectoryMode=0755
ExecStart=/home/jetson/embebidos-3/scripts/nano_start_server.sh
Restart=on-failure
RestartSec=10s

[Install]
WantedBy=multi-user.target
```

### Referencias

| Fuente | URL | Relevancia |
|---|---|---|
| man 5 sudoers | https://man7.org/linux/man-pages/man5/sudoers.5.html | Sintaxis wildcard `*` con `fnmatch(3)` |
| Compass Security blog | https://blog.compass-security.com/2012/10/dangerous-sudoers-entries-part-4-wildcards/ | Riesgos de inyección con wildcards en sudoers |
| Unix SE: polkit systemd templated | https://unix.stackexchange.com/questions/595207/ | `RegExp()` JS para action `org.freedesktop.systemd1.manage-units` |
| Debian PolicyKit wiki | https://wiki.debian.org/PolicyKit | Limitaciones `.pkla` vs `rules.d` en Debian/Ubuntu antiguos |
| man systemd.service | https://man.archlinux.org/man/systemd.service.5.en.txt | Directivas `Type=oneshot`, `RuntimeDirectory`, `EnvironmentFile` |

---

## Vacío 6 — `huggingface_hub` SDK en Python 3.6.9

### Decisión

**No existe versión instalable de `huggingface_hub` en Python 3.6.** Verificado leyendo `setup.py` directo de los releases más antiguos: v0.8.1, v0.9.1, v0.10.1, v0.11.0, v0.12.1 → todos declaran `python_requires=">=3.7.0"`. Python 3.6 fue EOL en diciembre 2021, antes de que `huggingface_hub` se publicara como librería independiente.

**Solución**: API REST directa con `requests` (instalado en el Nano vía system pip). Endpoints documentados oficialmente:

- **Download**: `GET https://huggingface.co/{repo}/resolve/{revision}/{filename}` con `Authorization: Bearer <HF_TOKEN>`. Soporta byte-range, streaming.
- **Listar archivos del repo**: `GET https://huggingface.co/api/models/{repo}?revision={rev}` → JSON con `siblings: [{rfilename, size, ...}]`.
- **Listar commits**: `GET https://huggingface.co/api/models/{repo}/commits/main`.
- **Upload (sin LFS)**: `POST https://huggingface.co/api/models/{repo}/commit/{branch}` con payload JSON `{summary, files: [{path, encoding: "base64", content}]}`.

**Riesgo upload binario 13 MB**: si el `.gitattributes` del repo tiene `*.engine filter=lfs`, el commit API rechaza el base64 inline con 422 y exige negociación LFS previa. Workaround: o instalar `huggingface_hub` en un venv Python 3.9 paralelo (vía `pyenv` o `conda` en el Nano — compila ARM en ~30 min), o usar `git lfs` directo con `subprocess`. Recomendación: empezar sin LFS para `engines-archive/` (cambiar `.gitattributes` del repo) y si funciona, perfecto; si no, instalar Py3.9 paralelo.

### Snippets (helper `scripts/hf_rest.py`)

```python
"""hf_rest.py — cliente minimal HF Hub para Python 3.6 (sin huggingface_hub).
Diseñado para Jetson Nano. Solo requiere `requests`.
"""
import base64
import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import requests

REPO = "mitgar14/embebidos-3-models"
BASE = "https://huggingface.co"
TOKEN = os.environ.get("HF_TOKEN")


def _headers():
    h = {}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


def download(filename: str, local_path: Path, revision: str = "main",
             chunk_size: int = 65536, timeout: int = 120) -> None:
    """Descarga un archivo del repo a local_path. Streaming."""
    url = f"{BASE}/{REPO}/resolve/{revision}/{filename}"
    r = requests.get(url, headers=_headers(), stream=True, timeout=timeout)
    r.raise_for_status()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = local_path.with_suffix(local_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
    tmp.rename(local_path)  # atomic


def list_files(revision: str = "main") -> List[Dict]:
    """Lista siblings del repo en la revision dada."""
    url = f"{BASE}/api/models/{REPO}"
    r = requests.get(url, headers=_headers(),
                     params={"revision": revision}, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("siblings", [])


def repo_info(revision: str = "main") -> Dict:
    """Info de la revision: sha, lastModified, etc."""
    url = f"{BASE}/api/models/{REPO}"
    r = requests.get(url, headers=_headers(),
                     params={"revision": revision}, timeout=30)
    r.raise_for_status()
    return r.json()


def get_head_revision() -> str:
    """SHA del último commit en main."""
    info = repo_info("main")
    return info.get("sha", "")


def upload_file_inline(local_path: Path, remote_path: str,
                       commit_msg: str = "embebidos3 backup",
                       branch: str = "main") -> Dict:
    """Upload sin LFS, base64 inline. Apto para archivos < ~50 MB."""
    content_b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
    payload = {
        "summary": commit_msg,
        "files": [{
            "path": remote_path,
            "encoding": "base64",
            "content": content_b64,
        }]
    }
    url = f"{BASE}/api/models/{REPO}/commit/{branch}"
    r = requests.post(
        url,
        headers={**_headers(), "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=300,
    )
    if r.status_code == 422 and "lfs" in r.text.lower():
        raise RuntimeError("Servidor exige LFS. Ver fallback con git-lfs.")
    r.raise_for_status()
    return r.json()
```

### Referencias

| Fuente | URL | Relevancia |
|---|---|---|
| `huggingface_hub` setup.py v0.8.1 | https://github.com/huggingface/huggingface_hub/blob/v0.8.1/setup.py | `python_requires=">=3.7.0"` confirmado |
| `huggingface_hub` setup.py v0.11.0 | https://github.com/huggingface/huggingface_hub/blob/v0.11.0/setup.py | Misma restricción |
| HF Hub OpenAPI | https://huggingface.co/.well-known/openapi.md | Spec endpoints download/upload/commit |
| HF docs: download files | https://huggingface.co/docs/huggingface_hub/guides/download | `/resolve/<rev>/<file>` documentado |
| HF docs: storage limits | https://huggingface.co/docs/hub/en/storage-limits | Límites LFS y archivos normales |

---

## Vacío 7 — Patrones de retomar jobs persistentes tras crash del orquestador

### Decisión

**PID file + `fcntl.lockf` + heartbeat JSON con `rename` atómico**. Patrón mínimo y correcto para single-job, sin Redis ni SQLite. Justificación:

- `fcntl.lockf` (POSIX `F_SETLK`) se libera automáticamente al morir el proceso (kernel cierra fd). `flock(2)` también lo hace pero es Linux-only. Ambos son equivalentes en práctica para Linux; `fcntl.lockf` es más portable.
- `os.kill(pid, 0)` es la forma canónica de verificar vivencia (no instala signal handler ni envía señal). Levanta `ProcessLookupError` si no existe, `PermissionError` si vive pero pertenece a otro user (también indica vivo).
- Edge case PID reuse: improbable en ventana de segundos. Para robustez extra, verificar `/proc/{pid}/cmdline` contiene el nombre del builder.
- Watchdog stale: si `time.time() - state["heartbeat"] > STALE_SECS` (recomendado 2-3× el intervalo de escritura del heartbeat, ej. STALE_SECS=120 si heartbeat cada 30s) → builder colgado.
- `/var/run/embebidos3/` con `RuntimeDirectory=embebidos3` en la unit systemd: tmpfs, owner correcto, limpio en cada boot (no quedan estados stale).

Patrón confirmado por daemons Unix clásicos (nginx, sshd, postfix), Sidekiq/Dramatiq usan TTL en Redis pero el principio es el mismo.

### Snippets

**Builder (escribe estado y heartbeat)** — `scripts/builder_state.py`:
```python
"""builder_state.py — helper para escribir job state desde el builder."""
import fcntl, json, os, time
from pathlib import Path

STATE_DIR = Path("/var/run/embebidos3")
STATE_FILE = STATE_DIR / "job.json"
LOCK_FILE = STATE_DIR / "builder.lock"


def acquire_lock():
    """Lock exclusivo. Libera automatico al morir el proceso. Levanta si ya bloqueado."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_FILE, "w")
    try:
        fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise RuntimeError("otro builder en curso (lock taken)")
    fh.write(str(os.getpid()))
    fh.flush()
    return fh  # MANTENER referencia viva mientras dure el job


def write_state(job_id, phase, progress_pct,
                phases_completed=None, current_message=None,
                onnx_source=None, eta_seconds=None):
    """Escritura atómica via tempfile + rename."""
    state = {
        "job_id": job_id,
        "pid": os.getpid(),
        "phase": phase,
        "progress_pct": progress_pct,
        "phases_completed": phases_completed or [],
        "current_message": current_message,
        "onnx_source": onnx_source,
        "eta_seconds": eta_seconds,
        "heartbeat": time.time(),
        "started_at_unix": state.get("started_at_unix") if STATE_FILE.exists() else time.time(),
    }
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_FILE)  # atomic en mismo filesystem
```

**Server (recovery al arrancar)** — fragmento de `nano_server.py`:
```python
import os, json, time
from pathlib import Path

STATE_FILE = Path("/var/run/embebidos3/job.json")
STALE_SECS = 120  # builder "colgado" si heartbeat > 2 min sin update


def recover_job_state():
    """Llamado en startup del server. Detecta jobs en curso/huérfanos."""
    if not STATE_FILE.exists():
        return None
    try:
        state = json.loads(STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None  # corruption, ignorar

    pid = state.get("pid")
    if pid is None:
        return None

    # ¿Proceso vivo?
    try:
        os.kill(pid, 0)
        alive = True
    except ProcessLookupError:
        alive = False
    except PermissionError:
        alive = True  # vive pero otro user

    if not alive:
        # Builder murió. Marcar abandoned, mover a logs/jobs/.
        finalize_abandoned(state)
        STATE_FILE.unlink()
        return None

    # Verificar que sea realmente el builder (no PID reuse)
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if cmdline_path.exists():
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        if "nano_build_engine" not in cmdline:
            finalize_abandoned(state)
            STATE_FILE.unlink()
            return None

    # ¿Hay heartbeat reciente?
    age = time.time() - state.get("heartbeat", 0)
    if age > STALE_SECS:
        return {"status": "stalled", "age_seconds": age, **state}

    return {"status": "running", **state}


def finalize_abandoned(state):
    """Persiste el state final marcado como ABANDONED en logs/jobs/."""
    job_id = state.get("job_id", "unknown")
    final = {
        **state,
        "phase": "abandoned",
        "ended_at_unix": time.time(),
        "reason": "builder process died, no heartbeat",
    }
    out = Path(f"/home/jetson/embebidos-3/logs/jobs/{job_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))
```

### Referencias

| Fuente | URL | Relevancia |
|---|---|---|
| man flock(2) | https://man7.org/linux/man-pages/man2/flock.2.html | Liberación automática al morir proceso |
| apenwarr — File locking | https://apenwarr.ca/log/20101213 | Referencia canónica `fcntl` vs `flock` Linux semantics |
| Stack Overflow #29611352 | https://stackoverflow.com/questions/29611352/ | Diferencias prácticas locking |
| kindatechnical PID file pattern | https://kindatechnical.com/shell-scripting-bash/file-locking-preventing-concurrent-execution.html | `kill -0` para stale detection |
| Stack Overflow #19572215 | https://stackoverflow.com/questions/19572215/ | `os.kill(pid, 0)` vs psutil |

---

## Track A — agentes ejecutados

| Agente | Foco | Tokens | Duración | Salida |
|---|---|---|---|---|
| research-code | Vacíos 1, 2, 3 (TRT engine reload + trtexec parsing + atomic swap) | 76.270 | 224 s | Incorporada arriba |
| research-web | Vacíos 5, 6, 7 (sudoers wildcard + HF py36 + job recovery) | 70.006 | 178 s | Incorporada arriba |
| research-video | Vacíos 4 + suplementos 1, 2 (Nano memory + hot-reload empírico + supervision) | 125.909 | 370 s | Incorporada arriba |

### Notas track A

- **research-video**: YouTube/MCP encontró 6 videos relevantes pero solo 2 con transcript procesable (Roboflow `X9jt8qb_igo`, Nicolai Nielsen `nQBOkGR_lg0` parcial). Canales canónicos para Jetson (JetsonHacks `UCQdl0WJ5E-oL9dwAkBNy3vw`, DustyNV `UCVfXTyLEEtHZGBh_GqMZh-A`, Edje Electronics, Paul McWhorter, NVIDIA Developer) NO surfacearon para queries específicas sobre `trtexec` OOM/workspace. Los hallazgos de mayor calidad vinieron de GitHub issues NVIDIA/TensorRT y NVIDIA DevForum donde los maintainers (@zerollzeng, @AastaLLL, @JerryChang, @ttyio, @nvpohanh) responden con comportamiento documentado.
- **AAI fallback**: NO se activó (`aai_threshold` no se bajó). Costo evitado: $0,12/h audio. Cobertura YouTube quedó parcial pero suficiente: la práctica empírica clave estaba en issues/forums, no en videos.

---

## Track B — discover.py + lectura activa

**Comando**: `python scripts/discover.py --tema "TensorRT engine hot-reload pycuda Jetson Nano trtexec progress cancel signal timeout sudoers systemd templated service huggingface_hub python 3.6 job recovery state file orphan" --profundidad alto --dominios github.com forums.developer.nvidia.com docs.nvidia.com huggingface.co stackoverflow.com docs.python.org systemd.io --max-resultados 25`

**Resultados**: 81 URLs descubiertas (8 dominios). Selección de valor confirmada (las usadas en los snippets/refs arriba ya están filtradas). URLs adicionales **no incorporadas** pero potencialmente útiles para futuras rondas:

- arXiv 2406.17749 — *Benchmarking Deep Learning Models on NVIDIA Jetson Nano for Real-Time Systems* — benchmarks empíricos TRT 8 en Nano.
- arXiv 2508.08430 — *Half-core utilization rule on Jetson edge devices* — paper de cita ya conocida del HANDOFF.
- arXiv 2110.11043 — *TensorRT Acceleration Pipeline* — describe el flow de optimización TRT capa por capa.
- NVIDIA/TensorRT issue #3868 — *Reuse engine for multiple consequent runs* — relevante para vacío 1 lado producción.
- NVIDIA/TensorRT issue #4459 — *Multi-process two engine models in one program* — relevante para vacío 1 si en futuro queremos modelos múltiples.
- NVIDIA DevForum: *Loading TensorRT model is very slow on Jetson Nano* (#175829) — relevante para cold-start del server.
- NVIDIA DevForum: *Install PyCuda on Nano* (#119836) — ya cubierto por HANDOFF, refresco con cambios recientes posibles.
- `paddypawprints/model-rt-build` (GitHub) — repo template para builds TRT en múltiples plataformas, patrón cross-platform.

---

## Constraints derivados que afectan el spec final

A integrar en `2026-05-15-dashboard-pipeline-design.md`:

1. **Refactor `TRTWorker`**: añadir métodos `request_swap(path)`, `_unload_engine()` reentrante, y `_swap_event` interno. El método `run()` debe chequear `_swap_event` en cada iteración del bucle de inferencia.
2. **Builder bash + Python helper**: el script `nano_build_engine.sh` orquesta el shell (timeout, swap config, trtexec), pero invoca un helper Python (`builder_state.py`) para escribir `job.json` con `fcntl.lockf` + `rename` atómico. No mezclar JSON writes en bash.
3. **sudoers + wrapper script**: el dashboard NO ejecuta `systemctl` directo. Server llama a `subprocess.run(["sudo", "/usr/local/bin/embebidos3-builder-launch", job_id])`. Wrapper valida regex.
4. **HF cliente custom**: el módulo `scripts/hf_rest.py` reemplaza la dependencia de `huggingface_hub`. Tres operaciones: `download`, `list_files`, `upload_file_inline`. Si upload falla con 422+LFS, abortar el cleanup y dejar el material en `.previous/` (alineado con decisión Q13 "no borrar nada si el backup falla").
5. **Pre-build script de Nano**: el builder ejecuta `prep_nano_for_build()` antes de `trtexec` y `restore_nano_after_build()` al final (también en cleanup trap). Esto incluye `sudo init 3` (no `stop lightdm` directo, más robusto), `swapon /mnt/swap.img`, `vm.swappiness=100`, `drop_caches`.
6. **Recovery en server startup**: `nano_server.py` debe llamar a `recover_job_state()` en `@app.on_event("startup")` y exponer el resultado en `GET /jobs/active`. Si encuentra job RUNNING válido, el dashboard puede reanudar el reporte sin perder progreso.
7. **Workspace default 512 MB**: dejar configurable via `EMBEBIDOS3_TRTEXEC_WORKSPACE` en `EnvironmentFile`. Empezar con 512, subir a 1024 solo si se valida estable. Cambio respecto a la asunción previa del brainstorming.
8. **Timeout 40 min**: `TimeoutStartSec=2700` (45 min, con buffer) en la unit systemd del builder + `timeout --kill-after=30s 40m` en el script. Doble protección.

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco | Output |
|-------|-------|-------------|------|--------|
| 1 | 2026-05-15 | alto | 7 vacíos técnicos del pipeline ONNX→TRT (TRT reload, trtexec parsing/timeout, atomic swap, Nano memory, sudoers wildcard, HF py36, job recovery) | este documento |

## Fuentes consultadas (acumulado ronda 1)

| # | Título | URL | Tipo |
|---|---|---|---|
| 1 | NVIDIA/TensorRT #1107 — context stack not empty | https://github.com/NVIDIA/TensorRT/issues/1107 | Issue |
| 2 | NVIDIA/TensorRT #1255 — reuse context | https://github.com/NVIDIA/TensorRT/issues/1255 | Issue |
| 3 | NVIDIA/TensorRT #1632 — context destroyed orden | https://github.com/NVIDIA/TensorRT/issues/1632 | Issue |
| 4 | NVIDIA/TensorRT #3715 — switch engines pattern | https://github.com/NVIDIA/TensorRT/issues/3715 | Issue |
| 5 | NVIDIA/TensorRT #2478 — logs TRT 8.5 verbose | https://github.com/NVIDIA/TensorRT/issues/2478 | Issue |
| 6 | NVIDIA/TensorRT #2350 — verbose redirect | https://github.com/NVIDIA/TensorRT/issues/2350 | Issue |
| 7 | NVIDIA/TensorRT #3281 — no cancel limpio CUDA | https://github.com/NVIDIA/TensorRT/issues/3281 | Issue |
| 8 | NVIDIA/TensorRT #3614 — trtexec silent expected | https://github.com/NVIDIA/TensorRT/issues/3614 | Issue |
| 9 | NVIDIA/TensorRT #4258 — OOM exit codes reales | https://github.com/NVIDIA/TensorRT/issues/4258 | Issue |
| 10 | NVIDIA/TensorRT #4730 — timingCacheFile | https://github.com/NVIDIA/TensorRT/issues/4730 | Issue |
| 11 | NVIDIA/TensorRT #2679 — workspace no es límite duro | https://github.com/NVIDIA/TensorRT/issues/2679 | Issue |
| 12 | NVIDIA/TensorRT #1404 — workspace + stop desktop | https://github.com/NVIDIA/TensorRT/issues/1404 | Issue |
| 13 | NVIDIA DevForum — TensorRT Build Killed Nano | https://forums.developer.nvidia.com/t/tensorrt-build-killed/161504 | Forum |
| 14 | NVIDIA DevForum — Headed to headless mode | https://forums.developer.nvidia.com/t/headed-mode-to-headless-mode/120388 | Forum |
| 15 | NVIDIA DevForum — vm.swappiness Nano | https://forums.developer.nvidia.com/t/changing-the-linux-kernel-vm-swappiness-parameter/173035 | Forum |
| 16 | NVIDIA DevForum — Unified Memory trtexec | https://forums.developer.nvidia.com/t/unified-memory-management-for-trtexec/302346 | Forum |
| 17 | NVIDIA DevForum — Install PyCuda on Nano | https://forums.developer.nvidia.com/t/install-pycuda-on-nano/119836 | Forum |
| 18 | `triple-Mu/YOLOv8-TensorRT` | https://github.com/triple-Mu/YOLOv8-TensorRT | Repo |
| 19 | `jkjung-avt/tensorrt_demos` | https://github.com/jkjung-avt/tensorrt_demos | Repo |
| 20 | `NVIDIA-AI-IOT/torch2trt` | https://github.com/NVIDIA-AI-IOT/torch2trt | Repo |
| 21 | `triton-inference-server/server` model management docs | https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_management.md | Repo docs |
| 22 | Triton issue #5983 — rename atómico hot-reload | https://github.com/triton-inference-server/server/issues/5983 | Issue |
| 23 | `arvcode/TensorRT_classifier_efficientNet` Nano setup | https://github.com/arvcode/TensorRT_classifier_efficientNet/blob/main/jetson_nano_setup_instructions.md | Repo |
| 24 | POSIX `rename(2)` spec | https://pubs.opengroup.org/onlinepubs/9699919799/functions/renameat.html | Spec |
| 25 | TRT 8→10 trtexec migration | https://docs.nvidia.com/deeplearning/tensorrt/latest/api/tensorrt-8x-to-10x-trtexec.html | Docs |
| 26 | TRT Command-Line Programs docs | https://docs.nvidia.com/deeplearning/tensorrt/10.16.1/reference/command-line-programs.html | Docs |
| 27 | man 5 sudoers — wildcard sintaxis | https://man7.org/linux/man-pages/man5/sudoers.5.html | Man |
| 28 | Compass Security — Dangerous sudoers wildcards | https://blog.compass-security.com/2012/10/dangerous-sudoers-entries-part-4-wildcards/ | Blog |
| 29 | Unix SE — polkit systemd templated rule | https://unix.stackexchange.com/questions/595207/ | Forum |
| 30 | Debian wiki — PolicyKit | https://wiki.debian.org/PolicyKit | Wiki |
| 31 | man 5 systemd.service | https://man.archlinux.org/man/systemd.service.5.en.txt | Man |
| 32 | `huggingface_hub` v0.8.1 setup.py | https://github.com/huggingface/huggingface_hub/blob/v0.8.1/setup.py | Repo |
| 33 | `huggingface_hub` v0.11.0 setup.py | https://github.com/huggingface/huggingface_hub/blob/v0.11.0/setup.py | Repo |
| 34 | HF Hub OpenAPI | https://huggingface.co/.well-known/openapi.md | Spec |
| 35 | HF docs — download files | https://huggingface.co/docs/huggingface_hub/guides/download | Docs |
| 36 | HF docs — storage limits | https://huggingface.co/docs/hub/en/storage-limits | Docs |
| 37 | man flock(2) | https://man7.org/linux/man-pages/man2/flock.2.html | Man |
| 38 | apenwarr — File locking writeup | https://apenwarr.ca/log/20101213 | Blog |
| 39 | Stack Overflow #29611352 — fcntl vs flock | https://stackoverflow.com/questions/29611352/ | Forum |
| 40 | Stack Overflow #19572215 — kill 0 vs psutil | https://stackoverflow.com/questions/19572215/ | Forum |
| 41 | Stack Overflow #56124741 — pycuda GC DeviceAllocation | https://stackoverflow.com/questions/56124741 | Forum |
| 42 | kindatechnical — PID file pattern | https://kindatechnical.com/shell-scripting-bash/file-locking-preventing-concurrent-execution.html | Tutorial |
| 43 | YouTube — Ultralytics YOLOv8 Jetson | https://youtu.be/QGeP-Y6KMLM | Video (sin transcript util) |
| 44 | YouTube — Roboflow Jetson Inference Server | https://youtu.be/X9jt8qb_igo | Video |
| 45 | YouTube — Nicolai Nielsen YOLO11 TRT Jetson | https://youtu.be/nQBOkGR_lg0 | Video (transcript proto) |
| 46 | YouTube — Seeed Studio Jetson Mate Webinar | https://youtu.be/X4aeBefAgGA | Video (sin transcript) |
| 47 | YouTube — Ultralytics DeepStream multi-stream | https://youtu.be/wWmXKIteRLA | Video (transcript proto) |
| 48 | YouTube — Marcos Luciano save RAM Nano | https://youtu.be/Z3K43Q34sIs | Video (sin transcript) |
