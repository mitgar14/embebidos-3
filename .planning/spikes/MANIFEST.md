# Spike Manifest — V-2 atomic swap recovery

## Idea

Validar empíricamente que la opción A++ (centinela `.ready` + `reconcile_engine_state()`) resuelve el bug V-2 del proyecto embebidos-3 bajo SIGKILL real, antes de comprometerse a la implementación productiva en `main`.

El bug V-2 está documentado en `agent-docs/VACIOS.md`: el swap del engine TRT en `scripts/nano_build_engine.sh:191-197` es una secuencia de dos `mv` consecutivos NO atómicos entre sí. SIGKILL/OOM/power-loss entre los dos `mv` deja `engines/best_fp16.engine` inexistente. La opción A++ fue seleccionada en la ronda 1 de `/investiga` (ver `investigaciones/2026-05-16-atomic-swap-engine-recovery-mvp.md`), sustentada en patrones Cog (`replicate/cog pkg/weights/lockfile/lockfile.go`) + OSTree (`ostreedev/ostree src/libostree/ostree-sysroot-deploy.c`). Esta sesión de spikes cierra el gap de evidencia más importante de la ronda 1: validación empírica en hardware real.

## Requirements

Decisiones que NO son negociables para la implementación productiva derivada de estos spikes:

- **Sandbox aislado en el Nano**: `/home/jetson/spike-v2/`. NO tocar `/home/jetson/embebidos-3/engines/` ni servicios systemd productivos.
- **Engines sintéticos** (bytes random ~10 MB) — no requiere build real de TRT.
- **Python 3.6 compat estricto** — sin f-strings con `:=`, sin `match/case`, sin walrus.
- **No commits a main** hasta autorización explícita del usuario.
- **Centinela escrito como ÚLTIMA operación atómica** del swap (patrón Cog `.cog/ready`, OSTree `loader.tmp`).
- **`reconcile_engine_state()` debe ser idempotente y sin side-effects** cuando el estado es consistente (no false positives).

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | reproducir-v2-baseline | standard | Given swap actual sin fix, when SIGKILL entre los dos mv, then engine queda inexistente | **VALIDATED ✓** | filesystem, race, baseline |
| 002 | aplus-recovery-resuelve-v2 | standard | Given A++ + race reproducido, when reconcile_engine_state en startup, then engine recuperado matches sha256 pre-race | **VALIDATED ✓** | filesystem, recovery, sentinel, fsync |
| 003 | aplus-no-false-positives-integridad | standard | Given estado consistente, when reconcile corre, then no modifica nada. Y sha256 integridad post-recovery + degraded handling | **VALIDATED ✓** | filesystem, idempotencia, integridad, degraded |

## Cross-references

- Investigación bibliográfica ronda 1: `../../investigaciones/2026-05-16-atomic-swap-engine-recovery-mvp.md` (319 líneas, recomendación A++)
- Investigación bibliográfica ronda 2 (en curso, agentes background): mismo archivo, sección "Ronda 2"
- Bug origen: `../../agent-docs/VACIOS.md` (V-2)
- Código afectado: `../../scripts/nano_build_engine.sh:191-197`, `../../scripts/recover_job_state.py`
