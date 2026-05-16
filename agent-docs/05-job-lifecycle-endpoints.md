# 05 · Job lifecycle endpoints

Conjunto de endpoints HTTP del server que **gestionan el ciclo de vida del engine TRT** (no la inferencia, eso es el WS). Implementados todos en `scripts/nano_server.py`.

## Endpoints

| Método | Path | Función | Doc detalle |
|---|---|---|---|
| `POST` | `/model/build` | Lanzar un nuevo build | abajo |
| `POST` | `/model/check-updates` | Comparar local vs HF Hub | doc 03 |
| `POST` | `/model/adopt` | Adoptar engine pre-existente sin meta | abajo |
| `POST` | `/model/rollback` | Restaurar engine `.previous` | abajo |
| `GET` | `/model/state` | Estado consolidado del modelo | doc 02 |
| `GET` | `/jobs/active` | Job en curso (o null) | trivial |
| `GET` | `/jobs/{id}` | Estado de un job específico | abajo |
| `GET` | `/jobs/{id}/logs` | SSE stream del log del job | abajo |
| `DELETE` | `/jobs/{id}` | Cancelar un job activo | abajo |

## `POST /model/build`

Lanza un build. Idempotente sobre `active_job.json` (rechaza si hay otro en curso).

### Body

```json
{ "force": false }
```

- `force=false` → si el modelo está al día (verificación HF previa), rechaza con `420` o el frontend lo evita.
- `force=true` → siempre lanza, sin chequear HF Hub.

### Responses

| Status | Body | Cuándo |
|---|---|---|
| `202` | `{"ok": true, "job_id": "...", "monitor_url": "...", "logs_stream_url": "..."}` | Lanzado OK |
| `409` | `{"detail": {"ok": false, "error": "build_in_progress", "active_job_id": "..."}}` | Ya hay otro corriendo |
| `500` | `{"detail": {"ok": false, "error": "launch_failed", "reason": "exit_nonzero", "stderr": "..."}}` | El wrapper exit != 0 |
| `504` | `{"detail": {"ok": false, "error": "launch_timeout", "reason": "systemctl_start_exceeded_5s", "hint": "..."}}` | systemctl tardó >5s en retornar |

### Flujo interno

```python
@app.post("/model/build", status_code=202)
def model_build(req: BuildRequest = BuildRequest()):
    active = _read_active_job()
    if active:
        raise HTTPException(409, {"ok": False, "error": "build_in_progress", ...})
    job_id = _generate_job_id()  # YYYYMMDD-HHMM-aaeaf5
    try:
        subprocess.run(
            ["sudo", "/usr/local/bin/embebidos3-builder-launch", job_id],
            check=True, stdout=PIPE, stderr=PIPE,
            universal_newlines=True, timeout=5,
        )
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, {"ok": False, "error": "launch_failed", ...})
    except subprocess.TimeoutExpired:
        raise HTTPException(504, {"ok": False, "error": "launch_timeout", ...})
    except OSError as e:
        raise HTTPException(500, {"ok": False, "error": "launch_failed", ...})
    return {"ok": True, "job_id": job_id, ...}
```

### Por qué el wrapper externo

El server corre como user `jetson`, no root. Para invocar `systemctl start <unit>` necesita pasar por `sudo`. La regla sudoers concede NOPASSWD solo al wrapper validado `/usr/local/bin/embebidos3-builder-launch`, no a `systemctl` directamente — perimetraje de privilegios.

### El gotcha de `--no-block`

El unit del builder es `Type=oneshot`. Sin `--no-block` en `systemctl start`, el comando bloquea hasta que el ExecStart termine (15-40 min). El wrapper usa `--no-block` para retornar en <1s. Ver doc 01 para detalle.

## `POST /model/adopt`

Caso especial: el binario `best_fp16.engine` existe en el Nano pero no tiene `.meta.json` (típicamente: compilado a mano antes del sistema de tracking).

Adoptar significa:
1. SHA256 del binario actual
2. Llamar HF `head` para obtener revision actual
3. Llamar HF `lfs-sha exports/best.onnx` para obtener SHA del ONNX actual
4. Escribir `best_fp16.engine.meta.json` con esos valores + `adopted: true`

**No recompila**. Es un acto de fe: el operador confirma que el binario corresponde al modelo actual de HF.

### Responses

| Status | Caso |
|---|---|
| `200` | Adoptado OK, devuelve `meta` |
| `404` | No hay binario en disco |
| `409` | Ya tiene meta — no requiere adoptar |
| `500` | HF Hub unreachable u otra falla |

## `POST /model/rollback`

Swap inverso: restaura el engine `.previous/.engine.old` como activo.

```python
# swap inverso: usar nombres temp distintos para no colisionar
tmp_active = ACTIVE_ENGINE.parent / (ACTIVE_ENGINE.name + ".swap_tmp")
tmp_active_meta = ACTIVE_ENGINE_META.parent / (ACTIVE_ENGINE_META.name + ".swap_tmp")
if ACTIVE_ENGINE.exists():
    ACTIVE_ENGINE.replace(tmp_active)
    if ACTIVE_ENGINE_META.exists():
        ACTIVE_ENGINE_META.replace(tmp_active_meta)
PREVIOUS_ENGINE.replace(ACTIVE_ENGINE)
if PREVIOUS_ENGINE_META.exists():
    PREVIOUS_ENGINE_META.replace(ACTIVE_ENGINE_META)
if tmp_active.exists():
    tmp_active.replace(PREVIOUS_ENGINE)
    if tmp_active_meta.exists():
        tmp_active_meta.replace(PREVIOUS_ENGINE_META)
# mark as degraded
meta = _read_engine_meta(ACTIVE_ENGINE_META) or {}
meta["from_fallback"] = True
ACTIVE_ENGINE_META.write_text(json.dumps(meta, indent=2))
# hot-reload worker
worker.request_swap(str(ACTIVE_ENGINE))
```

Marca el engine como `from_fallback=true` en el meta → el endpoint `/model/state` lo expone como `state="degraded"`, el UI muestra banner ámbar.

## `GET /jobs/{id}`

Devuelve estado de un job:
- Si está activo → contenido de `/run/embebidos3/active_job.json`
- Si terminó → contenido de `logs/jobs/<id>.json` (lo escribe `builder_state.py finalize`)
- Si no existe → 404

## `GET /jobs/{id}/logs` (SSE)

Server-Sent Events stream del log file. El cliente abre un EventSource y recibe:

```
event: log
data: {"line": "[05/16/2026-13:45:21] Layer(Conv): ..."}

event: log
data: {"line": "[BUILD] phase=trtexec_running pct=42"}

event: done
data: {"exit_code": 0}
```

El server implementa tail-follow con `select.select` sobre el fd del log. Cuando detecta que el job terminó (active_job desaparece y el `.json` final aparece), emite `event: done` y cierra.

### El gotcha del flush

uvicorn por default agrega buffer al SSE. Para que el browser reciba los eventos en tiempo real, el header `X-Accel-Buffering: no` es obligatorio (presente). También se hace flush explícito de StreamingResponse.

## `DELETE /jobs/{id}`

Cancela un job activo enviando SIGTERM al unit systemd:

```python
subprocess.run(
    ["sudo", "/bin/systemctl", "stop", f"embebidos3-builder@{job_id}.service"],
    ...
)
```

systemd envía SIGTERM al script, que tiene `trap cancel_handler SIGTERM` (ver doc 04). El handler mata trtexec y finaliza el meta con `cancelled`.

## Comportamiento al desconectarse el cliente

El SSE de logs se cierra automáticamente cuando el cliente cierra la conexión (browser refresh, tab close). El server libera el fd y termina la generator function.

El polling de `/model/state` desde el dashboard es cada 3s. Si el server está abajo (durante un build), el fetch falla y el dashboard muestra el card "Servidor no responde" friendly (ver doc 06).

## Gotchas

- **`_generate_job_id` debe ser único** → usa `YYYYMMDD-HHMM-<6hex>`. Colisiones en mismo minuto son improbables pero posibles. No tiene lock — si dos POST llegan al mismo tiempo, el `_read_active_job` del segundo lo rechazará con 409.
- **`active_job.json` puede ser stale** si el builder murió por SIGKILL. La recovery del startup lo limpia (ver doc 09).
- **Sin autenticación**: cualquiera con acceso a Tailscale puede lanzar builds. Aceptable para MVP académico; no para producción.
