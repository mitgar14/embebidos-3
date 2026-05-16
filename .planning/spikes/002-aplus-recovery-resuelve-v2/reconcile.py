#!/usr/bin/env python3
"""Spike 002: reconcile_engine_state() prototipo con los 3 ajustes de ronda 2.

A++ recovery con:
  - Ajuste #1: fsync del parent_dir tras los renames (durabilidad)
  - Ajuste #2: solo cubre SIGKILL/OOM (power-loss fuera de scope software)
  - Ajuste #3: valida CONTENIDO de .previous (is_file), no solo directorio

Compat Python 3.6.9 (Jetson Nano JetPack 4.6.1) — sin walrus, sin f"{x=}", sin match/case.
"""
import json
import os
import sys
from pathlib import Path
from datetime import datetime


def reconcile_engine_state(root):
    """Reconcilia el estado del engine post-crash. A++ recovery.

    Casos:
      A. estado consistente (active engine + .ready) → no_op
      B. V-2 reproducido (active missing, previous válido) → auto_promote
      C. degraded (active missing, previous inválido o ausente) → degraded
    """
    engines_dir = Path(root) / "engines"
    active_engine = engines_dir / "best_fp16.engine"
    active_meta = engines_dir / "best_fp16.engine.meta.json"
    active_ready = engines_dir / "best_fp16.engine.ready"
    prev_dir = engines_dir / ".previous"
    prev_engine = prev_dir / "best_fp16.engine.old"
    prev_meta = prev_dir / "best_fp16.engine.old.meta.json"
    prev_ready = prev_dir / "best_fp16.engine.old.ready"

    timestamp = datetime.utcnow().isoformat() + "Z"

    # Caso A: estado consistente, no tocar
    if active_engine.is_file() and active_ready.is_file():
        return {
            "action": "no_op",
            "reason": "active engine + .ready coherent",
            "timestamp": timestamp,
        }

    # Caso B: V-2 reproducido, recuperar de previous
    # === Ajuste #3: validar CONTENIDO de previous (is_file), no solo directorio ===
    if (
        not active_engine.is_file()
        and prev_engine.is_file()
        and prev_ready.is_file()
    ):
        # Promover previous a active
        os.rename(str(prev_engine), str(active_engine))
        if prev_meta.is_file():
            os.rename(str(prev_meta), str(active_meta))
        os.rename(str(prev_ready), str(active_ready))

        # === Ajuste #1: fsync del parent_dir tras los renames ===
        fd = os.open(str(engines_dir), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        return {
            "action": "auto_promoted_previous",
            "reason": "active engine missing, previous valid with .ready",
            "timestamp": timestamp,
        }

    # Caso C: inconsistencia que no podemos resolver con A++ recovery
    return {
        "action": "degraded",
        "reason": "active missing AND previous invalid (no engine or no .ready)",
        "timestamp": timestamp,
        "active_engine_exists": active_engine.is_file(),
        "prev_engine_exists": prev_engine.is_file(),
        "prev_ready_exists": prev_ready.is_file(),
    }


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/jetson/spike-v2/002"
    result = reconcile_engine_state(root)
    print(json.dumps(result, indent=2))
    action = result.get("action", "")
    # Exit 0 si recuperó o estado consistente; 1 si degraded
    sys.exit(0 if action in ("no_op", "auto_promoted_previous") else 1)
