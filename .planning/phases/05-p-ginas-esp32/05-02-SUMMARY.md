---
phase: "05"
plan: "02"
subsystem: guia-3d
tags: [three-js, solidjs, webgl, esp32, servo, i2c]

dependency_graph:
  requires:
    - "05-01"   # skeleton de guia.tsx + connections.ts ya existían
  provides:
    - guia-3d-scene  # escena Three.js montada y limpiada correctamente
    - guia-3d-ui     # overlays CSS del sitio para tooltip, pin-detail, welcome-modal
  affects:
    - web/src/styles/tokens.css   # nueva sección Guia 3D overlays

tech_stack:
  added:
    - three@0.176    # ya en package.json desde plan 05-01
    - three/addons/controls/OrbitControls  # via bundler (no CDN)
  patterns:
    - SolidJS onMount/onCleanup con setAnimationLoop de Three.js
    - ResizeObserver en canvas (no window.resize)
    - createEffect reactivo al signal theme() para scene.background
    - scene.traverse + geometry/material dispose en cleanup

key_files:
  created: []
  modified:
    - web/src/routes/guia.tsx
    - web/src/styles/tokens.css

decisions:
  - "renderer.setAnimationLoop reemplaza requestAnimationFrame para que SolidJS
     pueda detenerlo en onCleanup sin fugar el contexto WebGL"
  - "createEffect anidado dentro de onMount (no al nivel del componente) para que
     se limpie automaticamente cuando el componente se desmonta"
  - "PinLabelOverlay y WelcomeModal leen IDs del DOM del JSX de guia.tsx;
     los IDs deben coincidir exactamente con los que los modulos JS esperan"
  - "Chunk guia ~634 kB (Three.js es el driver): aceptable en MVP donde la guia
     es una ruta lazy y no bloquea la carga del hub"

metrics:
  duration: "~20 min (incluyendo contexto compactado)"
  completed: "2026-05-25T13:15:00Z"
  tasks_completed: 2
  files_modified: 2
  build_result: "OK (908 ms, 69 módulos, 0 errores)"
---

# Phase 05 Plan 02: Guia 3D Three.js en guia.tsx — Summary

**One-liner:** Escena Three.js del repo esp32-pca9685-servo-guide portada a SolidJS con onMount/onCleanup/createEffect y overlays CSS del sitio.

## Tareas completadas

| # | Tarea | Commit | Archivos clave |
|---|-------|--------|----------------|
| 1 | Copiar y adaptar módulos de escena a `web/src/guia/scene/` | `387b80d` | 18 archivos .js en `scene/` |
| 2 | Implementar `guia.tsx` completo + overlays en tokens.css | `2653d03` | `routes/guia.tsx`, `styles/tokens.css` |

## Qué se construyó

**Tarea 1 (ya completa del contexto anterior):** Porta los 18 módulos JS del repo fuente (`scene/core/`, `scene/components/`, `scene/ui/`) a `web/src/guia/scene/`. Cambios realizados:

- `SceneManager.js`: reemplazó `window.addEventListener('resize')` por `ResizeObserver` en canvas; expone `this.observer` para `onCleanup`.
- `CameraController.js`: agregó método `dispose()` para liberar OrbitControls y FlyController.
- Tres archivos con `from '../data/connections.js'`: ruta cambiada a `../../connections.js`.
- `WelcomeModal.js`: eliminados emojis (sin icono div), sin atributo `title` en botón cerrar.
- `TextDecal.js`, `Servo.js`, `connections.ts`: secuencias `--` en comentarios/strings reemplazadas por `-`.

**Tarea 2:** `guia.tsx` completo con:

- Layout 3 columnas en JSX: sidebar 256 px con info de placas, BOM, lista de pasos y atajos de teclado; canvas Three.js con controles flotantes, SVG de leader-lines, overlay de etiquetas, loader, HUD de vuelo; panel derecho 288 px con badge, color del cable, título, descripción, código y dos secciones `<details>` con snippets de I²C scanner y test de servo.
- `onMount` sigue secciones 1-10 de `main.js` del repo fuente:
  1. `SceneManager` + `CameraController` + `InteractionManager`
  2. Placas (`ExpansionBoard`, `PCA9685`, `ExternalPSU`) + registro de pines en el raycaster
  3. 4 servos SG90 en CH0-CH3 con `ServoCable`
  4. `Wire` por cada `STEP` de `connections.ts`
  5. `PinLabelOverlay` registrando todos los pines de exp/pca/psu
  6. `InfoPanel`, `Tooltip`, `StepController` + listeners de botones
  7. Listeners `onChange` de `StepController` con `highlightStep()` que hace flyTo
  8. Hover/leave/click sobre pines via `InteractionManager` + `PinDetail`
  9. Sweep de servos (CH0-CH3) con oscilación senoidal
  10. `renderer.setAnimationLoop(loop)` con `THREE.Clock`
  11. `WelcomeModal` + botón `?` + atajo `?`/`Shift+/`
- `createEffect` reactivo a `theme()` para `scene.background` y `scene.fog.color`.
- `onCleanup` con `setAnimationLoop(null)`, `cam.dispose()`, `sm.observer.disconnect()`, `scene.traverse` + dispose de geometrías y materiales, `renderer.dispose()`.
- Atajos: `←`/`→` pasos, `A` vista general, `R` reset, `L` etiquetas, `F`/`f` vuelo, `?`/`Shift+/` welcome modal.

**tokens.css:** Sección `Guia 3D overlays` con estilos para `.tooltip`, `.pin-detail`, `.welcome-overlay/.welcome-card`, `.fly-hud`, `#guia-step-list`, badges y `.guia-toggle`.

## Desvíos del plan

### Ajustes automáticos

**1. [Rule 2 - Seguridad/accesibilidad] Sin atributo `title` nativo en botones**
- Aplicado en `WelcomeModal.js` (tarea 1) y en `guia.tsx` (tarea 2): todos los botones usan `aria-label` en lugar de `title`.

**2. [Rule 1 - Bug] Creación de badge-overview en InfoPanel**
- `InfoPanel.showOverview()` asigna la clase `badge-overview` que no estaba en el plan CSS. Se agregó el fondo neutral al badge de resumen dentro del estilo genérico del panel (el badge hereda `var(--bg-surface)`).

Ningún desvío arquitectónico. El plan se ejecutó como estaba escrito.

## Stubs conocidos

Ninguno. La escena carga datos reales de `connections.ts` (6 pasos hardcodeados, correctos para el MVP).

## Criterios de éxito verificados

- [x] `bun run build` pasa (908 ms, 69 módulos, 0 errores)
- [x] Todos los módulos de escena importados sin CDN (via Vite/bundler)
- [x] `onMount` instancia y arranca la escena con `setAnimationLoop`
- [x] `onCleanup` detiene el loop y dispone recursos WebGL
- [x] `createEffect` reactivo a `theme()` cambia `scene.background`
- [x] IDs del JSX coinciden con lo que esperan `InfoPanel`, `PinLabelOverlay`, `Tooltip`, `PinDetail`
- [x] Sin em-dashes ni `--` en ningún archivo generado
- [x] Tildes correctas en todo texto en español

## Self-Check: PASSED

Archivos verificados:
- `web/src/routes/guia.tsx` — existe (693 líneas añadidas)
- `web/src/styles/tokens.css` — existe (sección overlays añadida)
- Commit `387b80d` — tarea 1 (escena portada)
- Commit `2653d03` — tarea 2 (guia.tsx + tokens.css)
