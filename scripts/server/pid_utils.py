"""Helpers para inspección segura de PIDs (cross-platform aware).

Comparados entre nano_server.py (endpoint /model/state) y recover_job_state.py
(startup recovery): ambos necesitan defensa contra PID reuse antes de creer que
un job sigue vivo."""
import os
from pathlib import Path

BUILDER_CMDLINE_MARKER = "nano_build_engine"


def is_pid_alive(pid) -> bool:
    """True si el PID existe. Acepta int o str. En Windows también captura OSError
    porque os.kill(pid, 0) lanza OSError genérico para PIDs inexistentes."""
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, OSError):
        return False
    except PermissionError:
        # PID existe pero pertenece a otro user; cuenta como vivo.
        return True


def check_cmdline(pid, marker: str = BUILDER_CMDLINE_MARKER) -> bool:
    """True si /proc/<pid>/cmdline contiene `marker`.
    Defensa contra PID reuse: aseguramos que el proceso es el builder esperado.
    Retorna False en Windows o cuando /proc no existe."""
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return False
    try:
        cmdline = cmdline_path.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
        return marker in cmdline
    except Exception:
        return False
