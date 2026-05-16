"""Recovery del estado del builder al arrancar el server.
Importable desde nano_server.py o ejecutable como CLI para diagnóstico."""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from nano_server_constants import (
    ACTIVE_ENGINE, ACTIVE_ENGINE_META, ACTIVE_ENGINE_READY,
    ENGINES_DIR, HEARTBEAT_STALE_SEC, JOBS_LOGS_DIR, JOB_STATE_FILE,
    PREVIOUS_ENGINE, PREVIOUS_ENGINE_META, PREVIOUS_ENGINE_READY,
)
from pid_utils import is_pid_alive as _is_pid_alive, check_cmdline as _check_cmdline


def _finalize_abandoned(state):
    """Persiste el state final marcado como ABANDONED en logs/jobs/."""
    job_id = state.get("job_id", "unknown")
    final = {
        **state,
        "phase": "abandoned",
        "ended_at_unix": time.time(),
        "reason": "builder process died, no heartbeat",
    }
    out = JOBS_LOGS_DIR / f"{job_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(final, indent=2))


def recover_job_state():
    """Llamado en startup del server. Retorna estado del job activo o None."""
    if not JOB_STATE_FILE.exists():
        return None
    try:
        state = json.loads(JOB_STATE_FILE.read_text())
    except json.JSONDecodeError:
        return None

    pid = state.get("pid")
    if pid is None:
        return None

    if not _is_pid_alive(pid):
        _finalize_abandoned(state)
        try: JOB_STATE_FILE.unlink()
        except FileNotFoundError: pass
        return None

    if not _check_cmdline(pid):
        _finalize_abandoned(state)
        try: JOB_STATE_FILE.unlink()
        except FileNotFoundError: pass
        return None

    age = time.time() - state.get("heartbeat", 0)
    if age > HEARTBEAT_STALE_SEC:
        return {"status": "stalled", "age_seconds": age, **state}

    return {"status": "running", **state}


def reconcile_engine_state():
    """V-2 fix (2026-05-16): recovery del swap atómico interrumpido.

    Si el builder murió por SIGKILL/OOM entre los dos mv del swap, el sistema
    queda con engines/best_fp16.engine inexistente Y .previous/best_fp16.engine.old
    válido (con su .ready). Esta función detecta esa condición en startup y
    auto-promueve el .previous a activo.

    Casos:
      A. estado consistente (active + .ready coherentes) → no_op
      B. V-2 reproducido (active missing + previous válido con .ready) → auto_promote
      C. degraded (active missing + previous inválido) → degraded (requiere acción manual)

    Ajustes derivados de /investiga ronda 2 (validados por Spike 002+003):
      #1 fsync(parent_dir) tras los renames (durabilidad)
      #2 cubre SIGKILL/OOM; power-loss físico fuera de scope (UPS/batería)
      #3 valida PREVIOUS_ENGINE.is_file() (no solo directorio) antes de promover

    Sustento: investigaciones/2026-05-16-atomic-swap-engine-recovery-mvp.md
    Validación empírica: .planning/spikes/00{1,2,3}-*/
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Caso A: estado consistente, no tocar
    if ACTIVE_ENGINE.is_file() and ACTIVE_ENGINE_READY.is_file():
        return {
            "action": "no_op",
            "reason": "active engine + .ready coherent",
            "timestamp": timestamp,
        }

    # Caso B: V-2 reproducido, recuperar de previous
    # Ajuste #3: validar contenido (is_file), no solo directorio
    if (
        not ACTIVE_ENGINE.is_file()
        and PREVIOUS_ENGINE.is_file()
        and PREVIOUS_ENGINE_READY.is_file()
    ):
        os.rename(str(PREVIOUS_ENGINE), str(ACTIVE_ENGINE))
        if PREVIOUS_ENGINE_META.is_file():
            os.rename(str(PREVIOUS_ENGINE_META), str(ACTIVE_ENGINE_META))
        os.rename(str(PREVIOUS_ENGINE_READY), str(ACTIVE_ENGINE_READY))

        # Ajuste #1: fsync del parent_dir tras los renames (PostgreSQL durable_rename)
        fd = os.open(str(ENGINES_DIR), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        # Log persistente del recovery (auditable post-mortem)
        log_path = JOBS_LOGS_DIR / "v2_recovery_{0}.json".format(
            timestamp.replace(":", "-")
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps({
            "action": "auto_promoted_previous",
            "reason": "V-2 recovery: active missing, previous valid with .ready",
            "timestamp": timestamp,
        }, indent=2))

        return {
            "action": "auto_promoted_previous",
            "reason": "V-2 recovery: active missing, previous valid with .ready",
            "timestamp": timestamp,
        }

    # Caso C: inconsistencia que A++ no resuelve
    return {
        "action": "degraded",
        "reason": "active missing AND previous invalid (no engine or no .ready)",
        "timestamp": timestamp,
        "active_engine_exists": ACTIVE_ENGINE.is_file(),
        "prev_engine_exists": PREVIOUS_ENGINE.is_file(),
        "prev_ready_exists": PREVIOUS_ENGINE_READY.is_file(),
    }


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "job"
    if cmd == "engine":
        result = reconcile_engine_state()
    else:
        result = recover_job_state()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result else 1)
