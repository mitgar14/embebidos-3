# 09 · Recovery & watchdog

Mecanismos de resiliencia ante crashes inesperados del builder, del server, o reboots del Nano. Aspiración: que el sistema esté en un estado consistente y útil después de cualquier evento adverso, sin intervención manual.

## Archivos

| Archivo | Función |
|---|---|
| `scripts/recover_job_state.py` | Lógica de recovery, llamada desde el startup del server |
| `scripts/pid_utils.py` | Helpers `pid_alive(pid)`, `proc_cmdline(pid)` para verificar procesos |
| `scripts/builder_state.py` | API CLI para escribir/finalizar `active_job.json` |

## Casos cubiertos

### Caso 1: server arranca y encuentra `active_job.json`

¿Hay realmente un builder corriendo?

```python
# Pseudocódigo de recover_active_job_if_any()
job = read('/run/embebidos3/active_job.json')
if not job:
    return  # nada que recuperar

if job.has('builder_pid') and pid_alive(job.builder_pid):
    cmdline = proc_cmdline(job.builder_pid)
    if 'nano_build_engine.sh' in cmdline and job.job_id in cmdline:
        # builder VIVO, el server arrancó solo. Dejar active_job intacto.
        return

# builder muerto. ¿Su .json final ya existe?
final = JOBS_LOGS_DIR / f"{job.job_id}.json"
if final.exists():
    # ya finalizó, solo quedó active_job huérfano. Limpiar.
    unlink('/run/embebidos3/active_job.json')
    return

# builder murió sin finalizar (SIGKILL, crash). Marcar como fallido.
write_final(job.job_id, {**job, "phase": "abandoned", "exit_code": -1, ...})
unlink('/run/embebidos3/active_job.json')
```

Resultado: el dashboard al consultar `/model/state` ve `active_job: null` y `state: ready` (si hay engine válido) o `no_model`. **Nunca queda en `state: building` sin builder real corriendo**.

### Caso 2: builder hace SIGKILL antes del trap cleanup

Por `OOMKill` (el Nano se queda sin RAM), `TimeoutStopSec` excedido, o `systemctl kill -s SIGKILL`. El trap EXIT no se ejecuta.

- El server queda detenido (porque el trap no llegó al `start server`)
- `active_job.json` queda escrito con la última phase reportada
- El binario `.engine.new` puede quedar huérfano en `.staging/`

Mitigación actual:
- systemd reinicia el server solo si `Restart=on-failure` y exit != 0. Si el server fue detenido limpiamente por el builder antes del crash, **NO se reinicia automáticamente** — caso conocido reportado por el usuario, fix iterativo del trap cleanup.
- `recover_job_state` (al próximo arranque manual del server) limpia `active_job.json`.

### Caso 3: el server muere durante inferencia

`Restart=on-failure` + `RestartSec=5` lo levanta. El worker TRT se re-inicializa desde `ACTIVE_ENGINE_META`. Los clientes WS pierden la conexión y reconectan automáticamente.

### Caso 4: reboot del Nano durante un build

- `/run/embebidos3/` (tmpfs) se borra → no hay lock huérfano, no hay active_job stale
- El builder no se relanza solo (no está en `WantedBy=multi-user.target`)
- El server arranca solo (está `enabled`)
- Si hay engine válido en disk → estado `ready`; si no → `no_model`
- El usuario tiene que relanzar el build manualmente

### Caso 5: corrupción del meta JSON

Si `_read_engine_meta()` falla parseando JSON, devuelve `None`. El estado se computa como `no_model` con `engine_binary_present=true`. El usuario puede adoptar (regenera el meta).

## El watchdog del builder en el server

Implementación: commit `d513f89`. El startup hace:
1. Carga TRTWorker si hay engine
2. Llama `recover_active_job_if_any()`
3. Si hay job activo verificable (PID vivo + cmdline matching), no toca nada
4. Si hay job activo zombie, lo finaliza

**Lo que el watchdog NO hace**:
- Reiniciar un builder muerto (no tiene sentido sin contexto)
- Detectar heartbeat stale en mitad de un build (caso: builder corriendo pero stuck en trtexec sin progreso). Para esto, sería necesario un heartbeat con timestamp en `active_job.json` y un timer del server que detecte staleness >5 min.

## Heartbeat actual

`builder_state.py phase` actualiza `active_job.json` con `updated_at: <now>` cada vez que el script avanza de fase. El server **no usa** este campo aún, pero está disponible para una futura mejora.

## Lock file

`/run/embebidos3/builder.lock` con `flock -n`. Liberado automáticamente al morir el proceso (kernel-managed). Si dos builders se invocan concurrentemente, el segundo falla con exit 1 "otro builder en curso, abort".

## Idempotencia de operaciones

| Operación | Idempotente | Mecanismo |
|---|---|---|
| `POST /model/build` | No (devuelve 409 si hay otro) | `active_job.json` + lock file |
| `POST /model/rollback` | No (depende del estado) | check existencia de `.previous` |
| `POST /model/adopt` | Sí (rechaza si ya hay meta) | check existencia de `.meta.json` |
| `POST /model/check-updates` | Sí (read-only) | n/a |
| `DELETE /jobs/{id}` | Sí (no-op si ya terminó) | `systemctl stop` no-op sobre unit inactiva |

## Vacíos conocidos

- **No hay watchdog de staleness**: si el builder está vivo pero stuck (e.g., trtexec colgado), nadie lo mata. `TimeoutStartSec=2700` del unit le da 45 min de gracia antes de SIGTERM por systemd.
- **El swap interrumpido a mitad** (entre los dos `mv` del active y previous) deja sin engine activo. Recovery no detecta este caso. Apuntado en VACIOS.md.
- **No hay reintentos automáticos del builder** ante fallos transitorios (HF rate limit, red intermitente). El usuario tiene que dispararlo manualmente.
- **El histórico de jobs es placeholder**. `logs/jobs/*.json` existen pero el UI no los lee. La data está; falta el frontend.
