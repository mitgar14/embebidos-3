# Dashboard pipeline — diseño aprobado

**Fecha:** 2026-05-16.
**Proyecto:** `embebidos-3` — clasificador glass/paper/plastic, demo 2026-05-26.
**Inputs:**
- `docs/superpowers/specs/2026-05-15-dashboard-pipeline-request.md` — pedido del usuario.
- `docs/superpowers/specs/2026-05-15-dashboard-pipeline-brainstorm-preguntas.md` — Q1-Q13 + Approach B.
- `docs/superpowers/specs/2026-05-15-dashboard-redesign-design.md` — UI/tipografía/tokens del dashboard existente (la pestaña nueva hereda todo).
- `investigaciones/2026-05-15-pipeline-tecnico-vacios.md` — hallazgos sobre 7 vacíos técnicos.

**Stack:** JetPack 4.6.1 · L4T R32.7.1 · Python 3.6.9 · TensorRT 8.2.1.8 · pycuda 2019.1.2 · Maxwell `sm_53` · 4 GB RAM unificada · Tailscale.

---

## 1. Constraints duros

- `.pt → .onnx` queda fuera de scope (sigue en el notebook Vast.ai). El pipeline parte de `exports/best.onnx` ya en HF Hub.
- `.onnx → .engine` se compila **solo en el Nano** (engines TRT no son portables; Maxwell `sm_53` específico).
- Antes de cada build: `sudo init 3` + swap de disco 8 GB + `vm.swappiness=100` + `drop_caches`. **No usar zram** (comparte la RAM scarce).
- `--workspace=512` MB de default (validado con el engine actual de 13 MB). Configurable vía env var.
- Tiempo de compilación esperado: **15-45 min** en YOLOv8n@416 FP16. Timeout duro a 40 min.
- ONNX opset 11, ir_version 6 (TRT 8.2 acepta hasta opset 13; el manifest confirma compat).
- `huggingface_hub` SDK **no instalable en Python 3.6**. Cliente REST propio con `requests`.
- `trtexec` no responde a SIGTERM (CUDA kernel no interrumpible). Cancel = SIGKILL. El `.engine` se escribe al final desde RAM → SIGKILL no produce archivos corruptos.
- Repo HF privado → `HF_TOKEN` obligatorio, vive en `/etc/embebidos3/secrets.env` (0600).
- Cero emojis. Cero scroll global en el dashboard. Tipografía Source Sans 3 + Source Code Pro heredada del redesign del 2026-05-15.

---

## 2. Arquitectura macro

**Dos units systemd** en el Nano, ambas corriendo como `User=jetson`:

| Unit | Tipo | Rol |
|---|---|---|
| `embebidos3-server.service` | `simple`, `Restart=on-failure`, `enable` al boot | Sirve FastAPI/WS de inferencia + endpoints de control + recovery de jobs al arrancar |
| `embebidos3-builder@.service` | `oneshot`, **templated** (instancia por job-id), sin enable | Descarga ONNX, prepara Nano, ejecuta `trtexec`, valida, hace swap, reinicia server |

**Comunicación entre procesos** vía filesystem (sin Redis, sin IPC):

```
/var/run/embebidos3/                    ← tmpfs, RuntimeDirectory=embebidos3
├── job.json                            ← estado del job activo (atomic write)
├── builder.lock                        ← fcntl.lockf, libera al morir el proceso
└── (limpio en cada boot)

/home/jetson/embebidos-3/               ← persistente
├── engines/
│   ├── best_fp16.engine                ← engine activo
│   ├── best_fp16.engine.meta.json      ← metadata del activo
│   ├── .staging/                       ← engines pre-validación
│   └── .previous/                      ← último engine anterior (rollback)
├── onnx/best.onnx                      ← último onnx descargado
└── logs/
    ├── server.log                      ← uvicorn
    └── jobs/<jobid>.{log,json,tegrastats.log}
```

**Flujo de control**:

```
dashboard ──HTTP──> server ──sudo wrapper──> systemctl start builder@<jobid>
                     │                          │
                     │                          ├── escribe /var/run/embebidos3/job.json
                     │                          ├── descarga onnx + verifica SHA contra manifest
                     │                          ├── sudo systemctl stop server (libera GPU)
                     │                          ├── prep Nano (init 3, swap, swappiness, drop_caches)
                     │                          ├── trtexec ... > logs/jobs/<jobid>.log
                     │                          ├── validación in-process (3 imágenes)
                     │                          ├── backup viejo a HF Hub
                     │                          ├── swap atómico staging → engine
                     │                          ├── restore Nano (start lightdm, swappiness=60)
                     │                          └── sudo systemctl start server
                     │
                     └── lee /var/run/embebidos3/job.json on-demand para GET /jobs/active
```

**Privilegios mínimos** vía sudoers granular sobre wrapper scripts validados (no sobre `systemctl` directo, por riesgo de inyección con wildcard).

---

## 3. Estados del modelo y filesystem layout

### 3.1 Cinco estados del modelo

| Estado | Condición | UI dashboard |
|---|---|---|
| `no_model` | sin `engines/best_fp16.engine` ni `engines/.previous/` | banner "descargar y compilar primer engine" + botón |
| `ready` | engine carga OK, metadata válida | tarjeta con metadata: SHA del onnx, commit HF, fecha build, duración |
| `building` | existe `/var/run/embebidos3/builder.lock` activo | barra de progreso + logs en vivo + botón "cancelar" |
| `update_available` | `ready` + último check detectó nuevo commit HF | tarjeta `ready` + banner "actualizar a commit `abc1234`" |
| `degraded` | engine cargado viene del fallback `.previous/` | warning amarillo "usando engine anterior por falla previa" + botón "ver logs del último build fallido" |

### 3.2 Schema `engines/best_fp16.engine.meta.json`

```json
{
  "engine_sha256": "a30f8f5f...",
  "onnx_sha256":   "223f1a71...",
  "hf_revision":   "65c163404ea3...",
  "hf_commit_date": "2026-05-14T18:38:31Z",
  "trtexec_args":   ["--fp16", "--workspace=512", "--buildOnly", "--verbose"],
  "build_started_at":   "2026-05-16T14:39:02-05:00",
  "build_completed_at": "2026-05-16T14:47:18-05:00",
  "build_duration_s": 496,
  "validation": {
    "passed": true,
    "test_images_used": ["test_images/img01.jpg", "test_images/img03.jpg", "test_images/img07.jpg"],
    "detections_per_image": [2, 1, 3]
  }
}
```

### 3.3 Schema `/var/run/embebidos3/job.json` (atomic write cada cambio de fase + heartbeat 15s)

```json
{
  "job_id": "20260516-1422-abc123",
  "pid": 18452,
  "phase": "trtexec_building",
  "phases_completed": ["acquired_lock", "downloaded_onnx", "verified_sha",
                       "stopped_server", "prep_nano"],
  "started_at_unix": 1747424521.42,
  "heartbeat": 1747424793.17,
  "progress_pct": 47,
  "eta_seconds": 850,
  "current_message": "[I] [TRT] Timing Runner: Conv_42",
  "log_tail_path": "/home/jetson/embebidos-3/logs/jobs/20260516-1422-abc123.log",
  "onnx_source": {
    "hf_revision": "65c163404ea3...",
    "hf_commit_date": "2026-05-14T18:38:31Z",
    "onnx_sha256": "223f1a71..."
  },
  "cancellable": true
}
```

### 3.4 Schema final `logs/jobs/<jobid>.json` (estado terminal, mismo formato + `phase: done|failed|cancelled|abandoned` + `result`)

```json
{
  "job_id": "20260516-1422-abc123",
  "phase": "done",
  "phases_completed": ["acquired_lock", "downloaded_onnx", "verified_sha",
                       "stopped_server", "prep_nano", "trtexec_built",
                       "validated", "backed_up_previous", "swapped",
                       "restored_nano", "started_server"],
  "started_at_unix": 1747424521.42,
  "ended_at_unix": 1747425017.89,
  "build_duration_s": 496,
  "result": {
    "exit_code": 0,
    "swap_performed": true,
    "validation_passed": true,
    "hf_backup_uri": "engines-archive/20260516T142201Z__a30f8f5f/best_fp16.engine"
  },
  "onnx_source": { "...": "..." }
}
```

---

## 4. Endpoints FastAPI + data flow

| Método + ruta | Propósito | Auth | Idempotente |
|---|---|---|---|
| `GET /` | Banner texto plano (legado). | — | sí |
| `GET /health` | Estado server + GPU + RAM (legado, extendido con `model_state`). | — | sí |
| `GET /model/state` | Estado actual del modelo (`no_model` / `ready` / `building` / `update_available` / `degraded`) + metadata. | — | sí |
| `POST /model/check-updates` | Consulta HF Hub HEAD y compara contra `hf_revision` del engine actual. | — | sí |
| `POST /model/build` | Dispara un nuevo build (descarga + trtexec + validate + swap). Devuelve `{ job_id }`. **409** si ya hay build activo. | — | no |
| `GET /jobs/active` | Lee `/var/run/embebidos3/job.json`. Devuelve `null` si no hay job. | — | sí |
| `GET /jobs/<job_id>` | Estado terminal o vivo del job (consulta `job.json` o `logs/jobs/<id>.json`). | — | sí |
| `GET /jobs/<job_id>/logs` | **Stream SSE** del log del builder (tail -f equivalente). | — | sí |
| `DELETE /jobs/<job_id>` | Cancela el job en curso (`systemctl stop` + cleanup). | — | no |
| `POST /model/rollback` | Hace swap inverso (.previous/ → engine activo). Solo si existe `.previous/`. | — | no |

**Endpoints existentes que se mantienen sin cambios funcionales**: `WS /ws` (inferencia). Mientras `phase == building` el WS responde a frames con `{"ok": false, "error": "building", "job_id": "..."}` para que el dashboard pueda redirigir al usuario a la pestaña `modelo`.

### 4.1 Request/response — `POST /model/build`

Request:
```json
{
  "force": false,           // si true, ignora si engine actual está al día
  "workspace_mb": 512       // opcional, override del default
}
```

Response 202:
```json
{
  "ok": true,
  "job_id": "20260516-1422-abc123",
  "monitor_url": "/jobs/20260516-1422-abc123",
  "logs_stream_url": "/jobs/20260516-1422-abc123/logs"
}
```

Response 409 (ya hay build):
```json
{
  "ok": false,
  "error": "build_in_progress",
  "active_job_id": "20260516-1019-xyz789"
}
```

### 4.2 Stream de logs — `GET /jobs/<job_id>/logs` (SSE)

Headers: `Content-Type: text/event-stream`. Eventos:

```
event: log
data: {"line": "[I] Finished parsing network model. Parse time: 1.234", "ts": 1747424720.5}

event: phase
data: {"phase": "trtexec_building", "progress_pct": 15, "phases_completed": [...]}

event: heartbeat
data: {"ts": 1747424793.17}

event: done
data: {"phase": "done", "result": {"exit_code": 0, "swap_performed": true}}
```

El server hace tail del archivo `logs/jobs/<id>.log` (escrito por el builder, otro proceso) con `seek + read` cada 250 ms. Cuando detecta líneas conocidas (`Finished parsing`, `Engine built in`, `PASSED`, `FAILED`) emite el evento `phase` con el `progress_pct` derivado. El cliente reconnecta automáticamente vía `EventSource`.

### 4.3 Cancelación — `DELETE /jobs/<job_id>`

Flujo:
1. Server valida que `job_id` coincide con el activo en `/var/run/embebidos3/job.json`.
2. Server ejecuta `sudo systemctl stop embebidos3-builder@<job_id>.service` (sudoers permite esto).
3. systemd manda SIGTERM a la unit. El script tiene trap que:
   - Mata el proceso `trtexec` con SIGKILL (no responde a SIGTERM).
   - Limpia archivos `.staging/` parciales.
   - Marca job state como `cancelled` en `logs/jobs/<id>.json`.
   - Libera `builder.lock`.
   - Ejecuta `restore_nano_after_build` (restaurar lightdm, swappiness).
   - Reinicia el server con `sudo systemctl start embebidos3-server.service`.
4. Response 200 con `{"ok": true, "phase": "cancelling"}`. El cliente sigue polleando `/jobs/<id>` hasta ver `phase: cancelled`.

---

## 5. UI pestaña `modelo`

**Layout heredado del redesign 2026-05-15**: viewport 100vh sin scroll global, dos columnas (canvas left + panel right), tipografía Source Sans 3 + Source Code Pro, tokens OKLCH con accent mostaza.

### 5.1 Navegación entre pestañas

Tabs en el header (a la derecha del brand, antes de los chips):

```
[embebidos-3 · live detection]  [ live ] [ modelo ]  [chips...]  [icon capturar]
```

Routing simple basado en `window.location.hash` (`#live` / `#modelo`). Default: `#live`. Cambiar de pestaña **no interrumpe el job en curso** ni la inferencia. Cada pestaña preserva su estado interno (canvas+ws en `live`, polling de `/jobs/active` en `modelo`).

Mientras `phase == building`, la pestaña `live` muestra un banner discreto sobre el canvas: "modelo no disponible — build en curso, ver pestaña `modelo`". El WS de inferencia sigue conectado y respondiendo errores `building` → el dashboard los ignora y oculta detecciones.

### 5.2 Estructura de la pestaña `modelo`

Replica el layout dos-columnas pero adapta el contenido:

```
viewport (100vh, overflow: hidden)
┌─ header (igual) ────────────────────────────────────────────────────┐
├─────────────────────────────────────────────┬──────────────────────┤
│ main left                                   │ aside right 340 px   │
│                                             │                      │
│   panel principal según estado actual:      │   card Servidor      │
│   - no_model: hero descargar y compilar     │   card HF Hub        │
│   - ready: ficha del engine activo          │   card Acciones      │
│   - building: progress + logs stream         │   card Histórico     │
│   - update_available: ready + banner update │                      │
│   - degraded: warning + ver último fallo    │                      │
│                                             │                      │
└─────────────────────────────────────────────┴──────────────────────┘
```

### 5.3 Panel principal por estado

**`no_model`**:
```
┌─────────────────────────────────────────────────────────────────┐
│  Sin modelo cargado                                              │
│  Para empezar, descargá el último ONNX desde HF Hub y compilá   │
│  el engine TRT optimizado para este Jetson Nano.                │
│                                                                  │
│  [ descargar y compilar engine ]                                 │
│                                                                  │
│  Esto tarda entre 15 y 40 minutos. Podés cerrar la pestaña;     │
│  el proceso sigue corriendo en el servidor.                      │
└─────────────────────────────────────────────────────────────────┘
```

**`ready`** (tipografía mono para SHAs y fechas, tabular-nums):
```
┌─────────────────────────────────────────────────────────────────┐
│  Modelo activo                                                   │
│  ────────────────────────────────────────────────────            │
│  origen     commit 65c1634 · 2026-05-14 18:38                    │
│  onnx       sha 223f1a71... · 12,17 MB · opset 11                │
│  engine     sha a30f8f5f... · 13 MB · FP16                       │
│  compilado  2026-05-16 14:39 · 8 min 16 s                        │
│  workspace  512 MB                                               │
│                                                                  │
│  [ verificar actualizaciones ]   [ forzar recompilación ]        │
└─────────────────────────────────────────────────────────────────┘
```

**`building`** (progress bar + log viewer con scroll interno):
```
┌─────────────────────────────────────────────────────────────────┐
│  Compilando engine — job 20260516-1422-abc123                    │
│  ────────────────────────────────────────────────────            │
│  fase actual    trtexec_building                                 │
│  progreso       ▓▓▓▓▓▓▓░░░░░░░░░░  47 %                          │
│  transcurrido   8 min 12 s · estimado restante  14 min            │
│  origen         commit 65c1634 · 2026-05-14 18:38                │
│                                                                  │
│  [ cancelar build ]                                              │
│                                                                  │
│  logs en vivo  ────────────────────────────────────────          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ [I] Finished parsing network model. Parse time: 1.234   │    │
│  │ [I] [TRT] MemUsageChange Init builder: 234 MiB          │    │
│  │ [V] [TRT] Timing Runner: Conv_38 (CaskConvolution)      │    │
│  │ [V] [TRT] Timing Runner: Conv_42 (CaskConvolution)      │    │
│  │ ...                                                      │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

El log viewer usa Source Code Pro tabular-nums, scroll interno, auto-scroll al final, colores por severity (`[I]` --text, `[V]` --text-muted, `[W]` --warn, `[E]` --err).

**`update_available`**: bloque `ready` + banner amarillo arriba:
```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠ Nuevo entrenamiento disponible en HF Hub                       │
│  commit 7a3b8e2 · 2026-05-16 10:15 (hace 2 h)                    │
│  [ actualizar engine ]                                           │
└─────────────────────────────────────────────────────────────────┘
```

**`degraded`** (warning persistente + acceso a logs):
```
┌─────────────────────────────────────────────────────────────────┐
│  ⚠ Usando engine anterior                                        │
│  El último intento de actualización falló. El engine en uso es  │
│  el anterior validado (commit 65c1634 · 2026-05-14).             │
│                                                                  │
│  [ ver logs del último fallo ]   [ reintentar build ]            │
└─────────────────────────────────────────────────────────────────┘
```

### 5.4 Sidebar de la pestaña `modelo`

- **Card Servidor**: estado systemd (`active`/`inactive`/`failed`), uptime del server, PID, último restart. Solo lectura.
- **Card HF Hub**: repo (`mitgar14/embebidos-3-models`), revision actual del engine, último check, último backup subido a `engines-archive/`. Botón "verificar ahora".
- **Card Acciones**: `forzar recompilación`, `revertir a engine anterior` (si existe `.previous/`), `descargar manifest`. Cada acción muestra confirmación inline antes de disparar.
- **Card Histórico**: últimos 5 jobs (job_id, fecha, resultado, duración) con link a sus logs. Resto en `logs/jobs/`.

---

## 6. Builder oneshot — fases internas

### 6.1 Estructura del proceso

`scripts/nano_build_engine.sh <job_id>` orquesta el shell (timeout, swap config, trtexec), pero invoca un helper Python (`scripts/builder_state.py`) para escribir `job.json` con `fcntl.lockf` y `rename` atómico. No se mezclan JSON writes en bash.

### 6.2 Fases del builder (12 fases)

| # | Fase | Acción | Falla → |
|---|---|---|---|
| 1 | `acquired_lock` | `fcntl.lockf` sobre `/var/run/embebidos3/builder.lock`. Falla si ya hay otro builder. | abort, exit 1 |
| 2 | `downloaded_manifest` | `GET /<repo>/resolve/main/manifests/manifest.json` con `requests`. | retry 3x, luego fail |
| 3 | `downloaded_onnx` | `GET /<repo>/resolve/<rev>/exports/best.onnx` streaming a `onnx/best.onnx.tmp`. | retry 3x |
| 4 | `verified_sha` | SHA256 local vs `manifest.artifacts.best_onnx.sha256`. Si difiere → abort sin tocar server. | abort, exit 2 |
| 5 | `stopped_server` | `sudo systemctl stop embebidos3-server.service`. Espera hasta 30s a que libere GPU. | abort + cleanup |
| 6 | `prep_nano` | `init 3` + swap + `vm.swappiness=100` + `drop_caches`. | abort + restore |
| 7 | `trtexec_built` | `timeout 40m trtexec --onnx=... --saveEngine=staging/new.engine --fp16 --workspace=$WS --buildOnly --verbose`. Captura stdout a `logs/jobs/<id>.log` y feed de progreso a `job.json` cada N líneas. | exit code → result: 124 timeout, 137 OOM, otro = TRT error |
| 8 | `validated` | Helper Python carga staging engine en TRT context separado, corre 3 imágenes de `test_images/`. Pasa si ≥ 2 imágenes producen ≥ 1 detección con conf > 0,3. | abort + restore |
| 9 | `backed_up_previous` | Si existe `.previous/best_fp16.engine.old`: subir a `engines-archive/<timestamp>__<sha8>/` via `hf_rest.upload_file_inline`. **Si falla, NO continuar.** | abort + restore (engine activo intacto) |
| 10 | `swapped` | `mv .previous/* removido`, `mv engines/best_fp16.engine .previous/best_fp16.engine.old`, `mv staging/new.engine engines/best_fp16.engine`. Atomic rename POSIX. | restore from .previous/ |
| 11 | `restored_nano` | `sudo systemctl start lightdm`, `sysctl vm.swappiness=60`. | warn, sigue |
| 12 | `started_server` | `sudo systemctl start embebidos3-server.service`. | warn, sigue |

Al terminar (success o failure) escribir `logs/jobs/<id>.json` con el estado terminal, mover `job.json` → final state file, liberar `builder.lock`.

### 6.3 Esqueleto `nano_build_engine.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

JOB_ID="$1"
ROOT=/home/jetson/embebidos-3
LOG_FILE="${ROOT}/logs/jobs/${JOB_ID}.log"
TEGRA_LOG="${ROOT}/logs/jobs/${JOB_ID}.tegrastats.log"
STAGING_ENGINE="${ROOT}/engines/.staging/best_fp16.engine.new"
ACTIVE_ENGINE="${ROOT}/engines/best_fp16.engine"
PREV_DIR="${ROOT}/engines/.previous"
WORKSPACE="${EMBEBIDOS3_TRTEXEC_WORKSPACE:-512}"

mkdir -p "${ROOT}/logs/jobs" "${ROOT}/engines/.staging" "${PREV_DIR}"

# Lock exclusivo: fd se libera automaticamente al morir el proceso (kernel)
exec {LOCK_FD}<>/var/run/embebidos3/builder.lock
flock -n "$LOCK_FD" || { echo "[BUILD] otro builder en curso, abort" >&2; exit 1; }
echo $$ > "/var/run/embebidos3/builder.lock"

# Helper Python para job.json (escribe heartbeats + phase markers)
JS="python3 ${ROOT}/scripts/builder_state.py ${JOB_ID}"

cleanup() {
    local code=$?
    [[ -n "${TEGRA_PID:-}" ]] && kill "$TEGRA_PID" 2>/dev/null || true
    rm -f "$STAGING_ENGINE"
    # restore Nano (idempotente)
    sudo systemctl start lightdm.service 2>/dev/null || true
    sudo sysctl vm.swappiness=60 >/dev/null 2>&1 || true
    # marcar estado final
    if [[ $code -ne 0 ]]; then
        $JS finalize --phase failed --exit-code $code
        # asegurar que el server vuelva si fue parado
        sudo systemctl is-active --quiet embebidos3-server.service \
            || sudo systemctl start embebidos3-server.service
    fi
}
trap cleanup EXIT

$JS phase --name acquired_lock --pct 5

$JS phase --name downloaded_manifest --pct 8
python3 "${ROOT}/scripts/hf_rest.py" download manifests/manifest.json /tmp/manifest.json

$JS phase --name downloaded_onnx --pct 12
HF_REV=$(jq -r '.recovery.hf_revision // .hf_revision // "main"' /tmp/manifest.json)
python3 "${ROOT}/scripts/hf_rest.py" download exports/best.onnx "${ROOT}/onnx/best.onnx.tmp" --revision "$HF_REV"

$JS phase --name verified_sha --pct 15
EXPECTED_SHA=$(jq -r '.artifacts.best_onnx.sha256' /tmp/manifest.json)
ACTUAL_SHA=$(sha256sum "${ROOT}/onnx/best.onnx.tmp" | awk '{print $1}')
[[ "$EXPECTED_SHA" == "$ACTUAL_SHA" ]] || { echo "[BUILD] SHA mismatch" >&2; exit 2; }
mv "${ROOT}/onnx/best.onnx.tmp" "${ROOT}/onnx/best.onnx"

$JS phase --name stopped_server --pct 18
sudo systemctl stop embebidos3-server.service
sleep 3  # esperar release GPU

$JS phase --name prep_nano --pct 22
sudo systemctl stop lightdm.service 2>/dev/null || true
sudo systemctl disable nvzramconfig 2>/dev/null || true
sudo swapoff -a
if [[ ! -f /mnt/swap.img ]]; then
    sudo fallocate -l 8G /mnt/swap.img
    sudo chmod 600 /mnt/swap.img
    sudo mkswap /mnt/swap.img
fi
sudo swapon /mnt/swap.img
sudo sysctl vm.swappiness=100 >/dev/null
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null

$JS phase --name trtexec_started --pct 25
tegrastats --interval 2000 > "$TEGRA_LOG" 2>&1 &
TEGRA_PID=$!

set +e
timeout --kill-after=30s 40m \
    /usr/src/tensorrt/bin/trtexec \
        --onnx="${ROOT}/onnx/best.onnx" \
        --saveEngine="$STAGING_ENGINE" \
        --fp16 \
        --workspace="$WORKSPACE" \
        --buildOnly \
        --verbose \
        2>&1 | tee "$LOG_FILE" | python3 "${ROOT}/scripts/parse_trtexec_progress.py" "$JOB_ID"
TRTEXEC_EXIT="${PIPESTATUS[0]}"
set -e

kill $TEGRA_PID 2>/dev/null || true; unset TEGRA_PID

case $TRTEXEC_EXIT in
    0)   $JS phase --name trtexec_built --pct 75 ;;
    124) echo "[BUILD] timeout 40m" >&2; exit 124 ;;
    137) echo "[BUILD] OOM-killed" >&2; exit 137 ;;
    *)   echo "[BUILD] trtexec error $TRTEXEC_EXIT" >&2; exit "$TRTEXEC_EXIT" ;;
esac

# Sanity check
SIZE=$(stat -c %s "$STAGING_ENGINE")
[[ "$SIZE" -gt 1000000 ]] || { echo "[BUILD] engine sospechosamente chico: $SIZE" >&2; exit 1; }

$JS phase --name validating --pct 80
python3 "${ROOT}/scripts/validate_engine.py" "$STAGING_ENGINE" || \
    { echo "[BUILD] validación falló" >&2; exit 3; }
$JS phase --name validated --pct 85

# Backup HF antes de borrar nada
if [[ -f "${PREV_DIR}/best_fp16.engine.old" ]]; then
    $JS phase --name backing_up_previous --pct 88
    TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    OLD_SHA=$(sha256sum "${PREV_DIR}/best_fp16.engine.old" | awk '{print $1}' | head -c 8)
    python3 "${ROOT}/scripts/hf_rest.py" upload \
        "${PREV_DIR}/best_fp16.engine.old" \
        "engines-archive/${TIMESTAMP}__${OLD_SHA}/best_fp16.engine" || \
        { echo "[BUILD] backup HF falló, abortando cleanup" >&2; exit 4; }
    python3 "${ROOT}/scripts/hf_rest.py" upload \
        "${PREV_DIR}/best_fp16.engine.old.meta.json" \
        "engines-archive/${TIMESTAMP}__${OLD_SHA}/meta.json" || true
fi
$JS phase --name backed_up_previous --pct 92

# Swap atómico
[[ -f "${PREV_DIR}/best_fp16.engine.old" ]] && rm -f "${PREV_DIR}/best_fp16.engine.old" "${PREV_DIR}/best_fp16.engine.old.meta.json"
[[ -f "$ACTIVE_ENGINE" ]] && {
    mv "$ACTIVE_ENGINE" "${PREV_DIR}/best_fp16.engine.old"
    mv "${ACTIVE_ENGINE}.meta.json" "${PREV_DIR}/best_fp16.engine.old.meta.json" 2>/dev/null || true
}
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"
python3 "${ROOT}/scripts/write_engine_meta.py" "$ACTIVE_ENGINE" "$HF_REV" "$EXPECTED_SHA" "$WORKSPACE"
$JS phase --name swapped --pct 95

$JS phase --name restoring_nano --pct 97
sudo systemctl start lightdm.service 2>/dev/null || true
sudo sysctl vm.swappiness=60 >/dev/null

$JS phase --name starting_server --pct 99
sudo systemctl start embebidos3-server.service

$JS finalize --phase done --exit-code 0 --pct 100
```

### 6.4 Helper Python `scripts/builder_state.py` (resumen)

CLI con dos subcomandos:
- `phase --name <fase> --pct <n> [--message <txt>]`: update incremental al `job.json` (atomic via `tempfile + rename`). Lee estado actual, appendea fase a `phases_completed`, actualiza `phase`/`progress_pct`/`heartbeat`.
- `finalize --phase done|failed|cancelled --exit-code <n>`: cierra el job. Mueve `/var/run/embebidos3/job.json` a `logs/jobs/<id>.json`.

**Lock está fuera del helper**: el script bash adquiere `flock` sobre `/var/run/embebidos3/builder.lock` al inicio (snippet 6.3, líneas `exec {LOCK_FD}<>... flock -n $LOCK_FD`). El fd queda heredado por toda la jerarquía de procesos descendiente. Cuando el script termina (cleanly o por SIGKILL), el kernel cierra el fd y libera el lock automáticamente. El helper Python no necesita conocer el lock — la unicidad ya está garantizada a nivel bash.

### 6.5 Helper Python `scripts/validate_engine.py`

```python
"""Carga el engine staging en TRT context separado y corre 3 imágenes de test_images/.
Pasa si >= 2 imágenes producen >= 1 detección con conf > 0,3.
Exit 0 OK, 1 falla.
"""
import sys, glob
import cv2, numpy as np
import tensorrt as trt
import pycuda.driver as cuda
# ... (reusa letterbox + postprocess de nano_correctness.py)

if __name__ == "__main__":
    engine_path = sys.argv[1]
    test_images = sorted(glob.glob("/home/jetson/embebidos-3/test_images/*.jpg"))[:3]
    cuda.init()
    ctx = cuda.Device(0).make_context()
    try:
        ctx.push()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(open(engine_path, 'rb').read())
        trt_ctx = engine.create_execution_context()
        # ... bindings, stream, etc
        passed = 0
        for img_path in test_images:
            dets = run_inference(img_path, engine, trt_ctx, ...)
            if any(d['conf'] > 0.3 for d in dets):
                passed += 1
        ctx.pop()
        sys.exit(0 if passed >= 2 else 1)
    finally:
        ctx.detach()
```

### 6.6 Helper Python `scripts/parse_trtexec_progress.py`

```python
"""Lee stdin (tee del log de trtexec), detecta fases y actualiza job.json.
Recibe job_id como argv[1].
"""
import sys, re, subprocess
JOB_ID = sys.argv[1]

PHASE_HOOKS = [
    (re.compile(r'Finished parsing network model'), 'parsing_done', 30),
    (re.compile(r'\[MemUsageChange\].*Init builder'), 'mem_init', 35),
    (re.compile(r'Timing Runner'), None, None),  # incremental
    (re.compile(r'Engine built in'), 'engine_built', 65),
    (re.compile(r'Engine deserialized'), 'deserialized', 70),
    (re.compile(r'&&&& PASSED'), 'trtexec_passed', 72),
    (re.compile(r'&&&& FAILED'), 'trtexec_failed', None),
]

timing_count = 0
for line in sys.stdin:
    sys.stdout.write(line); sys.stdout.flush()  # pasa al log file via tee
    for rx, phase, pct in PHASE_HOOKS:
        if rx.search(line):
            if phase == 'parsing_done':
                subprocess.run(['python3', '/home/jetson/embebidos-3/scripts/builder_state.py',
                                JOB_ID, 'phase', '--name', phase, '--pct', str(pct)])
            elif phase is None:  # Timing Runner
                timing_count += 1
                if timing_count % 20 == 0:  # cada 20 capas reporta
                    subprocess.run(['python3', '...builder_state.py',
                                    JOB_ID, 'phase', '--name', 'trtexec_optimizing',
                                    '--pct', str(min(60, 35 + timing_count // 4)),
                                    '--message', line.strip()])
            # ... otras fases
```

---

## 7. Robustez y políticas de retención

### 7.1 Capas de robustez (7, ya aprobadas en brainstorming)

1. **Build a path staging oculto**: engine vivo nunca se sobrescribe durante la compilación.
2. **Validación post-build in-process**: 3 imágenes de `test_images/`, criterio `≥ 2 producen ≥ 1 det con conf > 0,3`.
3. **Swap atómico con backup**: `mv` atómico, viejo a `.previous/` (después de subirlo a HF).
4. **Rollback manual desde el dashboard**: botón `revertir a engine anterior` cuando existe `.previous/`.
5. **Auto-recovery en arranque del Nano**: server al boot intenta engine activo → fallback `.previous/` → modo `no_model`.
6. **Verificación SHA256 de descarga HF**: gate antes de tocar el server.
7. **Logs por job**: `logs/jobs/<id>.log` + `.json` para diagnóstico.

### 7.2 Políticas de retención

| Recurso | Política |
|---|---|
| Engines en `.previous/` | **Solo el último**. Antes de pisar/borrar, subir a HF Hub `engines-archive/<ts>__<sha8>/`. Si la subida falla, NO se borra (capa de seguridad). |
| Engines en HF `engines-archive/` | Sin cap explícito (acumulan en HF). Limpieza manual periódica si crece. |
| Logs `logs/jobs/<id>.{log,json,tegrastats.log}` | TTL **3 días**. Cron diario (`systemd-tmpfiles` con `d` rule) borra mayores a 72 h. |
| `/var/run/embebidos3/` | Limpio en cada boot (tmpfs). |
| Contador de inferencias acumuladas | Reinicia desde 0 al migrar a systemd. |

`systemd-tmpfiles` config en `/etc/tmpfiles.d/embebidos3-logs.conf`:
```
# Tipo Path                                            Mode UID    GID    Age  Argument
d     /home/jetson/embebidos-3/logs/jobs              0755 jetson jetson 3d
```

### 7.3 Recovery tras crash del server durante build

El builder es independiente (otra unit systemd), sigue corriendo. Cuando el server reinicia:

1. `recover_job_state()` lee `/var/run/embebidos3/job.json`.
2. Verifica `pid` vivo con `os.kill(pid, 0)` + cross-check con `/proc/{pid}/cmdline` (debe contener `nano_build_engine` para descartar PID reuse).
3. Si vivo + heartbeat reciente (< 120 s): **retoma el reporte de estado** sin interrumpir el job. Dashboard sigue viendo progreso normal.
4. Si vivo pero heartbeat stale (> 120 s): marca como `stalled` en la API, deja al builder seguir (no lo mata, el operador decide).
5. Si muerto: marca `abandoned` en `logs/jobs/<id>.json`, limpia staging, libera lock.

### 7.4 Timeout y cancelación

- Timeout duro **40 min** de `trtexec` vía `timeout --kill-after=30s 40m`. Doble protección en la unit: `TimeoutStartSec=2700` (45 min).
- Cancel desde dashboard: `DELETE /jobs/<id>` → `sudo systemctl stop embebidos3-builder@<id>.service` → trap en bash mata `trtexec` con SIGKILL, limpia staging, marca `cancelled`, reinicia server.
- El log queda persistido en `logs/jobs/<id>.log` aunque se cancele.

---

## 8. Setup, install y migración

### 8.1 Archivos nuevos en el repo

```
scripts/
├── nano_install_systemd.sh       ← NUEVO: instala units + sudoers + secrets template
├── nano_build_engine.sh          ← NUEVO: builder oneshot (script bash supervisor)
├── builder_state.py              ← NUEVO: helper job.json + lock
├── parse_trtexec_progress.py     ← NUEVO: parsea stdout trtexec → progress
├── validate_engine.py            ← NUEVO: mini-correctness in-process
├── write_engine_meta.py          ← NUEVO: escribe engine.meta.json
├── hf_rest.py                    ← NUEVO: cliente REST HF (download/upload/list)
├── recover_job_state.py          ← NUEVO: helper recovery (importado por nano_server.py)
├── embebidos3-builder-launch     ← NUEVO: wrapper sudoers-safe
├── nano_server.py                ← MODIFICADO: load/unload reentrante + endpoints nuevos + recovery
├── nano_start_server.sh          ← MODIFICADO: simplificado para systemd (sin nohup, sin pkill)
├── nano_stop_server.sh           ← BORRADO: lo reemplaza systemctl
└── nano_install_inference.sh     ← MODIFICADO: añade requests + jq + verifica systemd

systemd/
├── embebidos3-server.service     ← NUEVO
├── embebidos3-builder@.service   ← NUEVO
└── embebidos3-logs.tmpfiles.conf ← NUEVO (retención logs)

dashboard/
├── index.html                    ← MODIFICADO: añade tabs + pestaña modelo
├── app.js                        ← MODIFICADO: routing + módulo modelo + SSE client
└── style.css                     ← MODIFICADO: estilos tabs + pestaña modelo + log viewer
```

### 8.2 Script de instalación `scripts/nano_install_systemd.sh`

```bash
#!/usr/bin/env bash
# Instala las dos units systemd, sudoers, secrets template, wrapper, tmpfiles.
# Idempotente. Requiere sudo.
set -euo pipefail

ROOT=/home/jetson/embebidos-3
REPO_SYSTEMD=$(dirname "$0")/../systemd

echo "[1/8] crear /etc/embebidos3/"
sudo mkdir -p /etc/embebidos3
[[ -f /etc/embebidos3/secrets.env ]] || {
    sudo tee /etc/embebidos3/secrets.env > /dev/null <<EOF
# HF_TOKEN requerido para descargar repo privado mitgar14/embebidos-3-models
HF_TOKEN=hf_REEMPLAZAR
EMBEBIDOS3_TRTEXEC_WORKSPACE=512
EOF
    sudo chown root:jetson /etc/embebidos3/secrets.env
    sudo chmod 0640 /etc/embebidos3/secrets.env
    echo "    creado, editá con 'sudo nano /etc/embebidos3/secrets.env' antes de continuar"
}

echo "[2/8] instalar wrapper sudoers-safe"
sudo install -m 0755 "${ROOT}/scripts/embebidos3-builder-launch" /usr/local/bin/embebidos3-builder-launch

echo "[3/8] instalar sudoers"
sudo install -m 0440 -o root -g root /dev/stdin /etc/sudoers.d/embebidos3 <<EOF
jetson ALL=(root) NOPASSWD: /usr/local/bin/embebidos3-builder-launch *
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start embebidos3-server.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop embebidos3-builder@*.service
jetson ALL=(root) NOPASSWD: /bin/systemctl stop lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl start lightdm.service
jetson ALL=(root) NOPASSWD: /bin/systemctl disable nvzramconfig
jetson ALL=(root) NOPASSWD: /sbin/swapoff -a
jetson ALL=(root) NOPASSWD: /sbin/swapon /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/sysctl vm.swappiness=*
jetson ALL=(root) NOPASSWD: /sbin/fallocate -l 8G /mnt/swap.img
jetson ALL=(root) NOPASSWD: /bin/chmod 600 /mnt/swap.img
jetson ALL=(root) NOPASSWD: /sbin/mkswap /mnt/swap.img
jetson ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches
EOF
sudo visudo -cf /etc/sudoers.d/embebidos3

echo "[4/8] instalar units systemd"
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-server.service" /etc/systemd/system/
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-builder@.service" /etc/systemd/system/

echo "[5/8] instalar tmpfiles config (logs TTL 3d)"
sudo install -m 0644 "${REPO_SYSTEMD}/embebidos3-logs.tmpfiles.conf" /etc/tmpfiles.d/

echo "[6/8] systemctl daemon-reload + enable"
sudo systemctl daemon-reload
sudo systemctl enable embebidos3-server.service
sudo systemd-tmpfiles --create /etc/tmpfiles.d/embebidos3-logs.tmpfiles.conf

echo "[7/8] migrar server actual (si está corriendo con nohup)"
if pgrep -f 'nano_server:app' > /dev/null; then
    echo "    matando server nohup..."
    pkill -f 'nano_server:app' || true
    sleep 2
fi

echo "[8/8] start server vía systemd"
sudo systemctl start embebidos3-server.service
sleep 3
systemctl status embebidos3-server.service --no-pager | head -15

cat <<EOF

==============================================================================
  Setup completo
==============================================================================
  Verificá:
    curl http://localhost:8000/health
    curl http://localhost:8000/model/state
    systemctl status embebidos3-server.service
    sudo -l -U jetson | grep embebidos3

  HF_TOKEN: editá /etc/embebidos3/secrets.env con el token real.
  Después: sudo systemctl restart embebidos3-server.service
EOF
```

### 8.3 Migración del estado actual

1. Antes de correr `nano_install_systemd.sh`: el server actual corre con `nohup` (PID 12211 con 83.454 inferencias acumuladas). El contador se pierde (decisión consciente).
2. El script mata el proceso `nohup`, instala units, arranca via systemd.
3. El archivo `engines/best_fp16.engine` actual (SHA `a30f8f5f...`) se preserva intacto. Se le crea su `engine.meta.json` retroactivo con:
   ```json
   {
     "engine_sha256": "a30f8f5f...",
     "onnx_sha256": "223f1a71...",
     "hf_revision": "65c163404ea379e38522885101222a07242f37f9",
     "hf_commit_date": "2026-05-14T18:38:31Z",
     "trtexec_args": ["--fp16", "--workspace=512"],
     "build_completed_at": "2026-05-14T14:47:18-05:00",
     "build_duration_s": 480,
     "validation": { "passed": null, "note": "pre-migration, sin validación retroactiva" }
   }
   ```
   Esto permite al nuevo dashboard mostrar metadata correcta desde el primer arranque.
4. Bug `nano_server.py:360` (`ConnectionClosedOK`) se corrige en el mismo PR (wrap `await ws.send_text` en try/except).

### 8.4 Cleanup HF Hub (housekeeping post-migración)

Comandos a correr una sola vez (no parte del flujo automático):

```bash
# 1. Borrar 8 placeholders track_a/ y track_b/
hf repos delete-files mitgar14/embebidos-3-models \
  track_a/checkpoints/.gitkeep track_a/exports/.gitkeep \
  track_a/logs/.gitkeep track_a/runs/.gitkeep \
  track_b/checkpoints/.gitkeep track_b/exports/.gitkeep \
  track_b/logs/.gitkeep track_b/runs/.gitkeep

# 2. Subir README local actualizado (alineado a Track-B exclusivo)
huggingface-cli upload mitgar14/embebidos-3-models README.md README.md
```

---

## 9. Hallazgos técnicos críticos (resumen investigación 2026-05-15)

Detalles completos en `investigaciones/2026-05-15-pipeline-tecnico-vacios.md`. Síntesis de los 7 vacíos:

1. **TRT engine reload reentrante**: orden estricto `stream → outputs → inputs → context → engine → runtime` con `cu_ctx` activo. Bug clásico #1107 si se destruye con context popped. pycuda 2019.x libera `DeviceAllocation` automáticamente. Aplicado en `TRTWorker._unload_engine()`.
2. **`trtexec` parsing + cancel**: TRT 8.2 emite `[I/V] [TRT] msg` parseable. `--verbose` activa `[V] Timing Runner` por capa (granularidad fina). Sin signal handler propio → SIGKILL único cancel. `.engine` se escribe al final → SIGKILL nunca corrompe. Aplicado en `parse_trtexec_progress.py` y trap del builder bash.
3. **Atomic swap**: `rename(2)` POSIX en ext4 es seguro mientras el server tiene el engine cargado. TRT 8.2 no usa mmap persistente (verificado en código `jkjung-avt`, `triple-Mu`). Triton lo confirma. Aplicado en fase `swapped` del builder.
4. **Memory en Nano**: `--workspace` NO es límite duro (NVIDIA staff @zerollzeng confirmado #2679). Default `512 MB` (validado). Pre-build obligatorio: `init 3` + swap 8 GB de disco + `swappiness=100` + `drop_caches`. Sin zram. Aplicado en fase `prep_nano`.
5. **sudoers wildcard**: técnicamente funciona pero matchea espacios → riesgo inyección. Wrapper script con regex `^[A-Za-z0-9_-]{10,40}$`. Aplicado en `embebidos3-builder-launch`.
6. **HF SDK en py36**: NO existe. Cliente REST propio con `requests` (`scripts/hf_rest.py`). Endpoints `/resolve/<rev>/<file>` (download), `/api/models/<repo>/commit/main` con base64 (upload sin LFS).
7. **Job recovery**: PID file + `fcntl.lockf` + heartbeat JSON con `rename` atómico. `os.kill(pid, 0)` + cross-check `/proc/{pid}/cmdline`. Watchdog si `time.time() - heartbeat > 120 s`. Aplicado en `recover_job_state.py` (server startup) y `builder_state.py` (helper builder).

---

## 10. Plan de implementación

Fases ordenadas. Después de aprobado este spec, se invoca `writing-plans` para descomponer en tareas paso-a-paso con criterios de aceptación.

| Fase | Foco | Verificación |
|---|---|---|
| A — Infraestructura systemd | Units + sudoers + wrapper + tmpfiles + script install + secrets template | `nano_install_systemd.sh` corre OK, `systemctl status` activo, `sudo -l -U jetson` muestra exactamente las 14 entradas esperadas |
| B — Refactor `nano_server.py` | TRTWorker reentrante (load/unload), bug fix `:360`, endpoints `/model/state`, `/jobs/active`, recovery en startup | `curl /model/state` devuelve JSON correcto, server sobrevive a un swap manual de engine, logs sin excepción ConnectionClosedOK |
| C — Cliente HF REST | `scripts/hf_rest.py` con `download`, `list_files`, `repo_info`, `upload_file_inline`. Test contra `mitgar14/embebidos-3-models` | `python -m hf_rest download manifests/manifest.json /tmp/m.json` funciona, SHA del onnx descargado = SHA del manifest |
| D — Builder | `nano_build_engine.sh` + `builder_state.py` + `parse_trtexec_progress.py` + `validate_engine.py` + `write_engine_meta.py` | `sudo embebidos3-builder-launch test-jobid-abc123` ejecuta un build end-to-end exitoso, `/var/run/embebidos3/job.json` aparece, `logs/jobs/test-jobid-abc123.{log,json}` se generan |
| E — Endpoints job lifecycle | `POST /model/build`, `GET /jobs/<id>`, `GET /jobs/<id>/logs` SSE, `DELETE /jobs/<id>`, `POST /model/check-updates`, `POST /model/rollback` | `curl -X POST /model/build` devuelve 202 + job_id, SSE stream emite eventos, DELETE cancela limpio |
| F — UI pestaña `modelo` | Tabs en header, layout 2-col, los 5 estados (`no_model`/`ready`/`building`/`update_available`/`degraded`), SSE client, log viewer, todas las cards sidebar | Build end-to-end desde el botón "compilar engine" se ve completo en UI, cancelar funciona, rollback funciona |
| G — Cleanup HF Hub | Una sola corrida manual: borrar `.gitkeep` + subir README actualizado | `tree` del repo HF muestra estructura limpia, README dice Track-B-exclusivo |
| H — Validación end-to-end + docs | Build forzado desde UI → engine nuevo en disco + backup en HF + entry en histórico de jobs. Reboot del Nano → server arranca solo, engine cargado, dashboard muestra `ready` | Demo grabada (1-2 min) que muestra flujo completo |

Tiempo total estimado: 2-3 sesiones de trabajo activo. Las fases A, B, C, D son secuenciales (cada una habilita la siguiente). E y F pueden solaparse parcialmente. G es independiente (housekeeping, no bloquea).
