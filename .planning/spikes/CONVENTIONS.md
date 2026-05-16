# Spike Conventions — embebidos-3

Patrones y stack establecidos en la sesión de spikes V-2 (2026-05-16). Nuevos spikes siguen esto a menos que la pregunta requiera otra cosa.

## Stack

- **Bash** para orquestación de filesystem (reproduce_v2, setup_and_race, test_three_scenarios) — mismo lenguaje que `nano_build_engine.sh` productivo, evita drift de semántica.
- **Python 3.6.9** (compat estricto con JetPack 4.6.1) para lógica de recovery (`reconcile.py`) — sin walrus, sin f-string debug, sin match/case.
- **Sin Python helpers** salvo `fsync_path()` (Bash no tiene `fsync` builtin) y `reconcile.py` (lógica de decisión).

## Structure

```
.planning/spikes/
├── MANIFEST.md                          # tabla agregada de spikes con verdicts
├── CONVENTIONS.md                       # este archivo
└── NNN-descriptive-name/
    ├── README.md                        # frontmatter + What/Research/How/Results
    └── *.sh, *.py                       # scripts ejecutables
```

Sandbox en el Nano: `/home/jetson/spike-v2/NNN/`. **NUNCA tocar `/home/jetson/embebidos-3/`** (path productivo).

## Patterns

- **Engines sintéticos** con `dd if=/dev/urandom bs=1M count=10` — ~10 MB, suficiente para validar filesystem semantics sin requerir build TRT real (~500s).
- **Verificación pre-ejecución** del estado del Nano antes de tocar nada: `systemctl is-active embebidos3-server.service` + `ls /home/jetson/embebidos-3/engines/` + `df -h`. Documenta intacto el estado productivo.
- **Reproducción de race** via "no ejecutar segundo mv" (Variante A) en lugar de `kill -9` real durante `sleep`: el estado final es idéntico, sin la complejidad de timing controlado.
- **fsync_path helper** invocable desde Bash:
  ```bash
  fsync_path() {
      python3 -c "
  import os, sys
  fd = os.open(sys.argv[1], os.O_RDONLY)
  try: os.fsync(fd)
  finally: os.close(fd)
  " "$1"
  }
  ```
- **Asserts en Bash con counters**: `errors=$((errors+1))` por fail, `exit 0/1` final. Output prefijado con `[ASSERT]` para grep.
- **Cross-script reuse**: `reconcile.py` del spike 002 reutilizado por spike 003 sin copia (path absoluto `/home/jetson/spike-v2/002/reconcile.py`). Reduce duplicación.

## Tools & Libraries

- **Solo stdlib de Python 3.6**: `os`, `json`, `pathlib`, `datetime`, `sys`. Sin instalación de paquetes.
- **GNU coreutils del Nano**: `dd`, `mv`, `rm`, `stat -c %s`, `sha256sum`, `ls`, `mkdir -p`. Todo en `/bin` y `/usr/bin`.

## Sandbox lifecycle

- Crear: `ssh nano "mkdir -p /home/jetson/spike-v2/NNN"`
- Copiar: `scp .planning/spikes/NNN-name/*.sh *.py nano:/home/jetson/spike-v2/NNN/`
- Permisos: `ssh nano "chmod +x /home/jetson/spike-v2/NNN/*.sh *.py"`
- Cleanup post-validación (opcional): `ssh nano "rm -rf /home/jetson/spike-v2/"`. NO ejecutar mientras un spike posterior depende del anterior (ej. spike 003 reusa `reconcile.py` del spike 002).
