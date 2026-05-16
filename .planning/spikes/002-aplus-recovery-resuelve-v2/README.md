---
spike: 002
name: aplus-recovery-resuelve-v2
type: standard
validates: "Given A++ instalado (builder con fsync + centinela .ready) + race de spike 001 reproducido, when reconcile_engine_state() corre en startup, then engines/best_fp16.engine existe AND matches sha256 del .previous pre-race AND recovery <2s"
verdict: VALIDATED
related: [001, 003]
tags: [filesystem, recovery, sentinel, fsync, v2, aplus]
---

# Spike 002: A++ recovery resuelve V-2

## What This Validates

**Given** A++ instalado en una versión throwaway del builder (con los 3 ajustes obligatorios de ronda 2):
- Ajuste #1: `fsync(engine_fd)` antes del `mv` + `fsync(parent_dir_fd)` después
- Ajuste #2: threat model acotado a SIGKILL/OOM (power-loss out of scope software)
- Ajuste #3: `reconcile_engine_state()` valida `prev_engine.is_file()`, no solo directorio

**When** el builder A++ ejecuta el swap y muere ANTES del `mv staging → active` (caso V-2 con builder A++):

```bash
mv "$ACTIVE_ENGINE" "$PREV_ENGINE"          # OK
mv "$ACTIVE_READY" "$PREV_READY"            # OK (.ready heredado)
fsync_path "$PREV_DIR"                      # OK
>>> CRASH SIMULADO <<<
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"       # NO SE EJECUTA
```

Estado post-crash:
- `engines/best_fp16.engine` → **NO EXISTE** (bug V-2)
- `engines/best_fp16.engine.ready` → NO EXISTE
- `engines/.previous/best_fp16.engine.old` → **EXISTE** (10 MB)
- `engines/.previous/best_fp16.engine.old.ready` → **EXISTE** (commit marker válido)
- `engines/.staging/best_fp16.engine.new` → EXISTE (10 MB)

**Then** `reconcile_engine_state()` en startup del server:
- Detecta `not active_engine.is_file() AND prev_engine.is_file() AND prev_ready.is_file()`
- Auto-promueve previous → active (engine, meta, ready)
- `fsync(parent_dir)` para durabilidad
- Retorna `action=auto_promoted_previous`

Y los asserts validan:
1. active engine existe (recuperado)
2. active .ready existe
3. sha256 del active matches sha256 del active ORIGINAL pre-race (integridad confirmada)
4. previous engine ya no está en `.previous/`
5. previous .ready ya no está en `.previous/`

## Research

Cubierto por las 2 rondas de `/investiga` (`../../investigaciones/2026-05-16-atomic-swap-engine-recovery-mvp.md`). Spike implementa la decisión A++ confirmada con los 3 ajustes de ronda 2.

**Patrón base**: Cog `pkg/weights/lockfile/lockfile.go` `.cog/ready` + OSTree `prepare_new_bootloader_link` / `swap_bootloader`.

**Refinamientos de ronda 2 incorporados**:
- fsync explícitos (Ferrite ASPLOS 2016, LevelDB #195, FastForward #386)
- Validación de contenido (no solo existencia) en recovery (analogía mcu-tools/mcuboot #1966)

## How to Run

En el Nano (sandbox `/home/jetson/spike-v2/002/`):

```bash
# 1. Setup + race
bash setup_and_race.sh

# 2. Recovery
python3 reconcile.py /home/jetson/spike-v2/002

# 3. Assert
bash assert_recovery.sh
```

Compat Python 3.6.9 (sin walrus, sin f-string debug, sin match/case).

## What to Expect

```
[ASSERT] PASS: active engine EXISTS (10485760 bytes)
[ASSERT] PASS: active .ready EXISTS
[ASSERT] PASS: integridad sha256 (XXX...)
[ASSERT] PASS: previous engine MOVED
[ASSERT] PASS: previous .ready MOVED
[ASSERT] A++ RECOVERY VALIDADO ✓ (5/5 asserts)
```

## Investigation Trail

**Diseño del race**: la ventana V-2 con builder A++ es entre `fsync(PREV_DIR)` (último write del previous) y `mv $STAGING $ACTIVE` (primer write del nuevo active). Esta es la ventana donde el sistema queda con:
- previous válido (engine + .ready persistidos)
- active vacío
- staging intacto

Es exactamente lo que A++ está diseñado para recuperar.

**Race no cubiertos por A++** (documentados, fuera de scope del spike):
- SIGKILL entre `mv active → previous (engine)` y `mv active → previous (.ready)`: previous engine existe pero sin .ready. `reconcile` retorna `degraded` (correcto: integridad no garantizada).
- SIGKILL entre `mv staging → active` y escritura del `.ready` del active: active engine existe pero sin .ready. `reconcile` retorna `degraded`.
- Power-loss físico: fuera de scope software (ronda 2, ajuste #2).

Estas ventanas son cortas (~µs entre operaciones consecutivas) y los degraded mode son detectables/logueables, no silenciosos. Aceptable para MVP académico.

**Ejecución 2026-05-16 16:37 UTC-5**:
- Setup: active sha original `64c9c55f9bf5398f...`
- Race: previous con engine 10485760 bytes + .ready 123 bytes
- Reconcile: `auto_promoted_previous` retornado en <200ms
- Assert: 5/5 PASS, integridad sha256 confirmada

## Results

**VERDICT: VALIDATED ✓**

Ejecutado en el Nano (kernel 4.9.337-tegra, ext4 sobre SD card):

```
{
  "action": "auto_promoted_previous",
  "reason": "active engine missing, previous valid with .ready",
  "timestamp": "2026-05-16T21:37:52.278105Z"
}
```

```
[ASSERT] A++ RECOVERY VALIDADO ✓ (5/5 asserts)
```

**Hallazgos clave:**

1. **A++ resuelve V-2 empíricamente** en hardware real (no solo en teoría).
2. **Integridad preservada**: el sha256 del active post-recovery matches exactamente el sha256 del active pre-race. No hubo corrupción.
3. **Recovery instantáneo**: <200ms (dos `os.rename` + un `fsync`). Cumple criterio <2s del spike.
4. **Python 3.6 compat verificado**: `reconcile.py` corre sin warnings en el Nano. Sin sintaxis post-3.7.
5. **fsync explícitos NO degradan performance** en este flujo (operaciones one-shot, no loop).

**Habilita Spike 003**: validar (a) no false positives cuando estado es consistente, (b) caso degraded cuando previous también está roto.