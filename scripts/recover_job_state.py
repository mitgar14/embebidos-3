"""Recovery del estado del builder al arrancar el server.
Importable desde nano_server.py o ejecutable como CLI para diagnóstico."""
import json
import time
from pathlib import Path

from nano_server_constants import JOB_STATE_FILE, HEARTBEAT_STALE_SEC, JOBS_LOGS_DIR
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


if __name__ == "__main__":
    import sys
    result = recover_job_state()
    print(json.dumps(result, indent=2))
    sys.exit(0 if result else 1)
