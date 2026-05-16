---
spike: 001
name: reproducir-v2-baseline
type: standard
validates: "Given el swap actual sin fix en nano_build_engine.sh:191-197, when SIGKILL al builder entre el primer mv (línea 194) y el segundo (línea 197), then engines/best_fp16.engine queda inexistente Y .previous/best_fp16.engine.old existe Y .staging/best_fp16.engine.new existe"
verdict: VALIDATED
related: [002]
tags: [filesystem, race, baseline, v2]
---

# Spike 001: Reproducir V-2 baseline (sin fix)

## What This Validates

**Given** el código actual de swap en `scripts/nano_build_engine.sh:191-197`:

```bash
rm -f "$PREV_ENGINE" "$PREV_META"
if [[ -f "$ACTIVE_ENGINE" ]]; then
    mv "$ACTIVE_ENGINE" "$PREV_ENGINE"           # punto 1
    [[ -f "$ACTIVE_META" ]] && mv "$ACTIVE_META" "$PREV_META" || true
fi
mv "$STAGING_ENGINE" "$ACTIVE_ENGINE"            # punto 2
```

**When** SIGKILL al builder exactamente entre `punto 1` (mv active→previous) y `punto 2` (mv staging→active),

**Then** el filesystem queda en estado:
- `engines/best_fp16.engine` → **NO EXISTE** (bug V-2)
- `engines/.previous/best_fp16.engine.old` → EXISTE (válido, copia del active pre-race)
- `engines/.staging/best_fp16.engine.new` → EXISTE (válido, engine nuevo terminado de buildear)

## Research

No requiere research adicional — la mecánica del race está documentada en `agent-docs/VACIOS.md` (V-2) y la ronda 1 de `/investiga` (`investigaciones/2026-05-16-atomic-swap-engine-recovery-mvp.md`).

**Por qué este spike es necesario aunque parezca trivial:**
1. Establece el **reproductor sintético** que el Spike 002 reutilizará con el fix A++ aplicado.
2. Confirma que el entendimiento del race es correcto en el filesystem real del Nano (ext4 sobre SD card, no en mi laptop).
3. Da una **baseline observable** para comparar con el comportamiento post-fix.

## How to Run

Ejecutado en sandbox del Nano: `/home/jetson/spike-v2/001/`.

```bash
# Setup: crear engines sintéticos (~10 MB de bytes random)
bash setup_synthetic_engines.sh

# Ejecutar swap con kill -9 controlado entre los dos mv
bash run_race.sh

# Verificar estado post-race
bash assert_v2_reproduced.sh
```

## What to Expect

`assert_v2_reproduced.sh` debe retornar exit 0 y output:

```
[ASSERT] active engine: MISSING (esperado)
[ASSERT] previous engine: EXISTS (esperado)
[ASSERT] staging engine: EXISTS (esperado)
[ASSERT] V-2 REPRODUCIDO ✓
```

## Investigation Trail

**2026-05-16 23:35 — diseño**: discutida fidelidad de la simulación. Variante A (no ejecutar segundo mv) vs Variante B (kill -9 real durante sleep). Decisión: **Variante A es suficiente** porque el bug V-2 es un PROBLEMA DE ESTADO FINAL, no de mecanismo de muerte. Lo único que importa para validar el recovery es el conjunto de archivos en disco post-crash. Bajo SIGKILL real puede haber edge cases adicionales (buffer no flusheado) pero esos los cubre el ajuste #1 de ronda 2 (fsync explícitos) en el Spike 002, no este spike baseline.

**2026-05-16 23:40 — pre-ejecución**: verificación del estado del Nano OK:
- `embebidos3-server.service` ACTIVE (productivo intacto)
- `/home/jetson/embebidos-3/engines/best_fp16.engine` 13.5 MB intacto
- Disk: 32 GB disponibles
- Kernel 4.9.337-tegra (consistente con JetPack 4.6.1 esperado)

**2026-05-16 23:42 — ejecución**: scripts copiados a `/home/jetson/spike-v2/001/`. Ejecutado `reproduce_v2.sh` + `assert_v2_reproduced.sh`. Sin warnings, exit 0.

## Results

**VERDICT: VALIDATED ✓**

Ejecutado en el Nano (kernel 4.9.337-tegra, ext4 sobre SD card):

```
[SPIKE-001] kernel: 4.9.337-tegra
[SPIKE-001] creando engines sintéticos (~10 MB random)
[SPIKE-001] active sha256 pre-race: 31d1ea28aefd78aa...
[SPIKE-001] staging sha256: f07ad6959c10a82b...

[SPIKE-001] ejecutando swap (mismo flujo que nano_build_engine.sh:191-197)
[SPIKE-001]   rm -f "$PREV_ENGINE" "$PREV_META"
[SPIKE-001]   mv "$ACTIVE_ENGINE" "$PREV_ENGINE"           # punto 1 — completado
[SPIKE-001]   >>> CRASH SIMULADO — segundo mv NO se ejecuta <<<
```

Estado post-race verificado por `assert_v2_reproduced.sh`:

```
[ASSERT] PASS: active engine MISSING (V-2 confirmado)
[ASSERT] PASS: previous engine EXISTS (10485760 bytes)
[ASSERT] PASS: staging engine EXISTS (10485760 bytes)
[ASSERT] PASS: previous sha matches active pre-race (31d1ea28aefd78aa...)
[ASSERT] V-2 REPRODUCIDO ✓ (4/4 asserts)
EXIT: 0
```

**Evidencia concreta del bug V-2:**
1. El bug ocurre tal como está documentado en `agent-docs/VACIOS.md`: kill entre los dos mv deja `engines/best_fp16.engine` inexistente.
2. El `.previous/best_fp16.engine.old` queda recuperable y CON INTEGRIDAD (sha256 matches el active pre-race exactamente).
3. El `.staging/best_fp16.engine.new` queda intacto.
4. Nuestro reproductor sintético es válido y reusable para Spike 002.

**Sin sorpresas, sin edge cases nuevos.** El bug es exactamente como lo modela ronda 1.

**Habilita Spike 002**: el setup post-`reproduce_v2.sh` es exactamente el escenario de entrada para validar que `reconcile_engine_state()` (A++ con los 3 ajustes de ronda 2) recupera correctamente.
