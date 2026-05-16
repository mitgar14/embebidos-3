---
spike: 003
name: aplus-no-false-positives-integridad
type: standard
validates: "Given estado consistente, when reconcile corre, then no modifica nada. Y given recovery, when validamos sha256, then matches pre-race. Y given previous corrupto, when reconcile corre, then retorna degraded sin auto-promover."
verdict: VALIDATED
related: [001, 002]
tags: [filesystem, idempotencia, integridad, degraded, ajuste-3, aplus]
---

# Spike 003: A++ no false positives + integridad + degraded handling

## What This Validates

**Tres escenarios complementarios al spike 002:**

**Escenario A — no false positives (estado consistente)**

Given un build exitoso con `active_engine` + `active_engine.ready` coherentes, when `reconcile_engine_state()` corre, then:
- Retorna `action=no_op`
- NO modifica `active_engine` (sha256 + size matches pre-reconcile)
- NO toca `.ready`

**Escenario B — degraded sin .ready (validación del centinela)**

Given `active_engine` missing + `previous_engine` presente PERO sin `.ready`, when `reconcile_engine_state()` corre, then:
- Retorna `action=degraded`
- NO auto-promueve (sin centinela = sin garantía de integridad)
- Exit code != 0 (señaliza al caller que se requiere intervención)

**Escenario C — degraded sin engine en previous (ajuste #3 de ronda 2)**

Given `active_engine` missing + `previous .ready` presente PERO sin `previous_engine`, when `reconcile_engine_state()` corre, then:
- Retorna `action=degraded`
- Validó CONTENIDO de previous (no solo existencia del directorio) — **ajuste #3 funcionando**
- NO inventa active engine de la nada

## Research

Sin research adicional. Spike empírico de los 3 ajustes de ronda 2 aplicados a casos edge.

## How to Run

En el Nano (sandbox `/home/jetson/spike-v2/003/`, reutiliza `reconcile.py` del spike 002):

```bash
bash test_three_scenarios.sh
```

## What to Expect

```
[SPIKE-003-A] errors: 0
[SPIKE-003-B] errors: 0
[SPIKE-003-C] errors: 0
[SPIKE-003] A++ COMPORTAMIENTO CORRECTO en los 3 escenarios ✓
EXIT: 0
```

## Investigation Trail

**Diseño**: tres escenarios elegidos por su valor probatorio:

- A valida **idempotencia**: si reconcile se llama en estado consistente, no debe ejecutar side effects (importante porque el server lo llamará en cada startup).
- B valida que el centinela `.ready` es **necesario** para auto-promote — sin él, A++ degrada graciosamente en lugar de promover engine potencialmente incompleto.
- C valida el **ajuste #3 de ronda 2**: validar CONTENIDO de previous (is_file), no solo existencia del directorio o de algún archivo dentro.

**Ejecución 2026-05-16 16:39 UTC-5**: los 3 escenarios pasaron en una sola ejecución. 0 errors agregados.

**Sin sorpresas, sin edge cases nuevos.** El comportamiento de `reconcile_engine_state()` matches la especificación derivada de ronda 2.

## Results

**VERDICT: VALIDATED ✓**

Ejecutado en el Nano:

**Escenario A (no_op):**
```
{
  "action": "no_op",
  "reason": "active engine + .ready coherent"
}
[ASSERT-A] PASS: action=no_op
[ASSERT-A] PASS: active engine NO modificado (sha + size matches)
[ASSERT-A] PASS: .ready intacto
```

**Escenario B (degraded sin .ready):**
```
{
  "action": "degraded",
  "reason": "active missing AND previous invalid (no engine or no .ready)",
  "active_engine_exists": false,
  "prev_engine_exists": true,
  "prev_ready_exists": false
}
[ASSERT-B] PASS (3/3)
```

**Escenario C (degraded sin engine):**
```
{
  "action": "degraded",
  "reason": "active missing AND previous invalid (no engine or no .ready)",
  "active_engine_exists": false,
  "prev_engine_exists": false,
  "prev_ready_exists": true
}
[ASSERT-C] PASS (2/2)
```

**Hallazgos clave:**

1. **A++ es idempotente**: ejecuciones repetidas sobre estado consistente no causan side effects. Crítico para el server startup.
2. **Centinela `.ready` es necesario para auto-promote**: ausencia de centinela = degraded, nunca auto-promote especulativo.
3. **Ajuste #3 de ronda 2 verificado**: validación de contenido (`is_file()`) previene auto-promote de `.previous` con directorio existente pero engine ausente.
4. **Exit codes informativos**: 0 = consistent/recovered, != 0 = degraded (caller sabe que requiere intervención manual).
5. **Sin race entre el spike 002 y 003** (corren en sandboxes separados `/A_consistent/`, `/B_degraded_no_ready/`, `/C_degraded_no_engine/`).

Combinado con Spike 001 + 002: **A++ funciona correctamente en happy path Y en edge cases**.