# 02 · FastAPI server + TRTWorker reentrante

El proceso `embebidos3-server.service` es un único `nano_server.py` (FastAPI + uvicorn) que sirve:
- HTTP/REST: estado del modelo, lifecycle de jobs, métricas
- WebSocket: stream de inferencia binaria (JPEG → bboxes JSON)
- SSE: log streaming en vivo durante un build

## Archivos

| Archivo | Propósito |
|---|---|
| `scripts/nano_server.py` | FastAPI app completo (≈760 líneas) |
| `scripts/nano_server_constants.py` | Paths y nombres de archivos compartidos (no negociables) |
| `scripts/nano_start_server.sh` | Wrapper systemd: activa el venv, exporta env, ejecuta uvicorn |

## Estructura del módulo `nano_server.py`

| Líneas | Bloque |
|---|---|
| 1-68 | imports, constantes, lifespan |
| 71-326 | `class TRTWorker(threading.Thread)` — el motor de inferencia |
| 328-368 | `read_gpu_temp_c()`, `read_ram_mb()` — telemetría tegrastats |
| 370-388 | startup/shutdown hooks |
| 389-470 | endpoints `/`, `/health`, `/model/state` |
| 471-560 | endpoints de build/jobs/rollback |
| 569-672 | endpoints de update-check / adopt / cancel |
| 673-704 | SSE `/jobs/{id}/logs` |
| 705-756 | WebSocket `/ws` |

## `TRTWorker` — el componente más sutil

Es un hilo Python con un loop principal que:
1. Drenado de `queue.Queue` para recibir frames JPEG
2. JPEG decode → letterbox → blob HWC→NCHW
3. `IExecutionContext.execute_async_v2`
4. NMS + postprocess
5. Push de resultado por callback

### Por qué es **reentrante**

Originalmente cargaba el engine **una sola vez al startup** y nunca lo soltaba. Esto bloqueaba el hot-reload tras un build: el server tenía que reiniciarse para tomar el engine nuevo.

Refactor (commit `d513f89`): el worker expone:
- `_load_engine(path)` — internal, llamado desde `__init__` y `_swap_loop`
- `_unload_engine()` — internal, libera context+engine+CUDA stream
- `request_swap(new_path)` — **público**, hace que el worker termine el frame actual, unloadee y recargue desde el nuevo path

`request_swap` es thread-safe vía un `threading.Event` y un slot `_pending_swap_path` protegidos por lock. El callsite es el endpoint `/model/build` post-swap (Fase D step 9: `swapped`) y `/model/rollback`.

### Estado interno

```python
self._engine = None              # ICudaEngine
self._context = None             # IExecutionContext
self._stream = None              # cuda.Stream
self._input_buf = None           # GPU mem
self._output_buf = None
self._conf_th = 0.5              # ajustable via WS msg type=set_conf
self._stats = {                  # leído por /health
    "frames_total": 0,
    "infer_ms_ewma": None,
    "last_dets": 0,
}
```

### Letterbox y postprocess

El `_letterbox` y `_postprocess` **deben** ser bit-exactos con `scripts/nano_correctness.py` para que `validate_engine.py` use el mismo path al cargar imágenes test. Es la razón por la que `nano_correctness.py` exporta `letterbox`, `postprocess`, `IMGSZ`, `CLASSES`, `CONF_TH`, `NMS_TH` — son la SSOT compartida entre runtime y validación. **No duplicar.**

## Endpoint `/model/state` — el corazón del dashboard

Devuelve un objeto que enumera todos los estados posibles. Decisión clave (Task E2): un solo endpoint para que el dashboard no tenga que orquestar varias llamadas.

```json
{
  "state": "ready" | "no_model" | "building" | "update_available" | "degraded",
  "active_engine": {
    "hf_revision": "b93964f...",
    "hf_commit_date": "2026-05-16T14:47:01.000Z",
    "onnx_sha256": "223f1a71...",
    "engine_sha256": "a30f8f5f...",
    "build_completed_at": "2026-05-16T18:32:05Z",
    "build_duration_s": 1843,
    "trtexec_args": ["--workspace=512", "--fp16", "--buildOnly"],
    "validation": {...},
    "adopted": false
  },
  "previous_engine": {...} | null,
  "active_job": {...} | null,
  "engine_binary_present": true | false
}
```

### El estado `degraded`

Lo escribe `/model/rollback` cuando se detecta un swap-back desde `.previous`. Marca que el engine actual viene del fallback, no del último build. El dashboard lo muestra con un banner ámbar.

### `engine_binary_present` separado

No siempre el binario `.engine` tiene meta. Si encontramos `.engine` sin `.meta.json`, el estado es `no_model` (estrictamente: no hay tracking) **pero** `engine_binary_present=true` permite ofrecer la acción "adoptar" en el UI.

## Endpoint `/health` — telemetría del runtime

Devuelve fps real, latencia media, temperatura GPU, RAM libre. Lo poll el header del dashboard cada ~3s para las chips de la topbar. **No mezclar con `/model/state`** — `/health` es de alta frecuencia, `/model/state` es de baja frecuencia.

```python
return {
    "ok": True,
    "fps": worker._stats.get("fps", 0),
    "infer_ms_ewma": worker._stats.get("infer_ms_ewma"),
    "gpu_temp_c": read_gpu_temp_c(),
    "ram_mb": read_ram_mb(),  # {"avail": 1234, "total": 3900}
    "model_loaded": worker._engine is not None,
}
```

## WebSocket `/ws` — el path caliente

- Cliente envía frame JPEG binario (`bytes`) — el server lo encola en `worker.submit((bytes, ts_client))`
- Cliente puede enviar JSON `{"type": "set_conf", "value": 0.6}` para ajustar umbral
- Server responde JSON `{"type": "detections", "dets": [...], "ts_client": ..., "ts_server": ...}` por cada frame

### Manejo de errores en el WS

Originalmente el `await ws.send_json({"error": ...})` dentro de `try/except` tiraba un `ConnectionClosedOK` ruidoso cuando el cliente ya se había desconectado. Fix (commit `577ccf0`): catch silencioso de `ConnectionClosedOK` durante el envío de error.

## startup/shutdown hooks

Startup:
1. Lee `ACTIVE_ENGINE_META` — si no existe, no carga worker (estado `no_model`)
2. Si existe, instancia `TRTWorker(ACTIVE_ENGINE_PATH)` y `worker.start()`
3. Llama a `recover_job_state.recover_active_job_if_any()` — ver doc `09-recovery-watchdog.md`

Shutdown (signal `SIGTERM` desde systemd):
1. `worker.stop()` → flushea queue, libera GPU
2. Espera join con timeout 5s
3. Cierra WS abiertos

## Gotchas conocidos

- **Python 3.6 en el Nano** → no `subprocess.run(capture_output=True)`; usar `stdout=PIPE, stderr=PIPE, universal_newlines=True`. Fix en commit `3a54df6`.
- **`uvicorn` + WebSocket** → necesita `websockets` lib (no `wsproto`). Especificado en pyproject.
- **PyCUDA en hilo de Python ≠ hilo principal** → necesita `cuda.Context.push()` al inicio del thread y `pop()` al final, o usar `cuda.init()` + `Device(0).make_context()`. Implementado en `TRTWorker.run()`.
- **Liberación del context** debe pasar antes de `cuda.Context.pop()`, sino segfault al shutdown.
