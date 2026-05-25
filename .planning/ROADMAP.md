# Roadmap: embebidos-3 — Consola Web

## Overview

Greenfield SolidJS + Vite + Bun que consolida cuatro superficies (Hub, Dashboard live, Engine del Modelo, Labelling) sobre un backend FastAPI existente en la Jetson Nano. El camino crítico para la demo del 2026-05-26 es: fundación + sistema de diseño + cliente WS reconectante -> Hub -> Dashboard live a 14 fps. Engine y Labelling se construyen en paralelo pero tienen menor prioridad de demo. La orquestación con `web.ps1` entra en la fundación para que el frontend sea levantable desde el primer commit. La Fase 5 suma dos superficies de hardware (Guía 3D del cableado y Control de servos del ESP32 por MQTT), integradas al hub.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, 4, 5): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Fundación** - Proyecto SolidJS inicializado, sistema de diseño (claro/oscuro conmutable) aplicado, cliente WS y script `web.ps1` listos
- [x] **Phase 2: Hub** - Pantalla de inicio command-center con estado inline del Nano, navegable a las 3 superficies
- [x] **Phase 3: Dashboard Live** - Canvas con overlay de detección a 14 fps, métricas, controles y reconexión robusta
- [x] **Phase 4: Engine y Labelling** - Centro del Engine del Modelo con logs SSE y Labelling con drag/resize/export
- [ ] **Phase 5: Páginas ESP32** - Guía 3D interactiva del cableado (ESP32 + PCA9685 + servos) y panel de control de servos vía MQTT, integradas al hub (5 destinos)

## Phase Details

### Phase 1: Fundación

**Goal**: El proyecto SolidJS compila y sirve; el sistema de diseño unificado (con tema claro/oscuro conmutable) está disponible en toda la app; el cliente WebSocket reconectante existe como módulo; `web.ps1` levanta y apaga el frontend en Windows y macOS
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: DES-01, DES-02, DES-03, DES-04, DES-05, CONN-01, CONN-02, CONN-03, OPS-01, OPS-02, OPS-03
**Success Criteria** (what must be TRUE):

  1. `bun run dev` (o `web.ps1 -Action start`) arranca la app y el navegador abre `/#/` sin errores de consola
  2. Los tokens semánticos (Radix Slate claro/oscuro + IBM Cyan) y la tipografía IBM Plex Sans/Mono son visibles en cualquier componente de prueba; no hay `border-left` de acento ni emojis como iconos
  3. El módulo `lib/ws.ts` (`ReconnectingWebSocket`) se puede importar y conectar al WS del Nano; muestra estado `conectando`/`activa`/`reconectando` ante caídas simuladas
  4. `web.ps1 -Action stop` mata el proceso del frontend (y sus hijos) tanto en Windows como en macOS/pwsh
  5. La detección de SO (Windows vs macOS) retorna el valor correcto y se puede usar para ramificar comandos
  6. El usuario puede alternar tema claro/oscuro con un toggle; la preferencia persiste entre recargas y el primer arranque respeta `prefers-color-scheme`; ambos temas mantienen la estética fría sin romper contraste

**Plans**: 3 planes
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Walking Skeleton: scaffold web/ + sistema de diseño dual-theme + routing HashRouter + componente de prueba

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Cliente WS reconectante: lib/ws.ts + wsStore.ts + indicador de estado en Placeholder
- [x] 01-03-PLAN.md — Orquestación: web.ps1 start/stop/restart/status cross-platform

**UI hint**: yes

### Phase 2: Hub

**Goal**: El usuario puede ver el hub de inicio y navegar a las tres superficies de la app; el estado del Nano aparece inline sin salir del hub
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: HUB-01, HUB-02
**Success Criteria** (what must be TRUE):

  1. El hub muestra tres destinos (Dashboard, Engine del Modelo, Labelling) en columna vertical; un clic en cualquiera navega a la superficie correspondiente
  2. El hub muestra inline el estado del Nano (conectado / última inferencia / salud) sin requerir acción del usuario
  3. La pantalla del hub respeta el sistema de diseño: sin cards, sin border-left, estado comunicado con dots/pills/tint y SVG

**Plans**: completada en camino pragmático (sin PLAN.md formal). El hub HOME (3 destinos + diseño) se construyó durante el pulido de la Fase 1; HUB-02 (estado del Nano inline) se cerró en el commit 2fb6ee6. Verificada con Playwright en claro/oscuro.
**UI hint**: yes

### Phase 3: Dashboard Live

**Goal**: El usuario puede ver el stream de cámara con overlay de detección a ~14 fps, métricas en vivo, controles y reconexión automática; la sustentación puede demostrar detección fluida aunque la red caiga
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: DASH-01, DASH-02, DASH-03, DASH-04, DASH-05
**Success Criteria** (what must be TRUE):

  1. El canvas muestra frames JPEG con bounding boxes superpuestos (clases vidrio/papel/plástico con colores Wong) a ~14 fps sin congelarse
  2. El panel lateral muestra métricas en vivo: fps, latencia, temperatura GPU, RAM y conteo por clase, actualizándose con cada frame
  3. El usuario puede cambiar la fuente de cámara, el umbral de confianza y el fps objetivo; los cambios se aplican en vivo al Nano sin recargar
  4. Si la conexión WS cae, el Dashboard muestra un indicador de reconexión y se reconecta automáticamente sin intervención del usuario
  5. El usuario puede exportar un snapshot PNG del frame actual con sus bounding boxes desde el Dashboard

**Plans**: TBD
**UI hint**: yes

### Phase 4: Engine y Labelling

**Goal**: El usuario puede consultar el estado del modelo, disparar un build y ver los logs en vivo; también puede anotar imágenes del dataset con bounding boxes y exportar en formato YOLO
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: ENG-01, ENG-02, ENG-03, LBL-01, LBL-02, LBL-03, LBL-04, LBL-05
**Success Criteria** (what must be TRUE):

  1. El Centro del Engine muestra el estado del modelo (cargado, motor/engine, timestamp del último build) con indicador semafórico sin border-left
  2. El usuario puede disparar un build desde la interfaz; los logs del build aparecen en streaming SSE con auto-scroll y diferencian stdout (gris/verde) de stderr (rojo)
  3. El usuario puede navegar entre imágenes del dataset (anterior/siguiente y mediante una tira de thumbnails), dibujar bboxes, asignar una de las 3 clases (con color por clase) y exportar las anotaciones en formato YOLO

**Plans**: completada en camino pragmático. Engine del Modelo (ENG-01..03) contra el backend del Nano (commit 57d5aff); Labelling (LBL-01..05) como editor client-side con export YOLO en .zip vía JSZip (commit 55498f0). Ambos verificados con Playwright
**UI hint**: yes

### Phase 5: Páginas ESP32

**Goal**: El usuario puede abrir desde el hub una guía 3D interactiva del cableado ESP32 + PCA9685 + servos (con la fidelidad gráfica del original revestida con la estética del sitio) y un panel de control que mueve los servos del ESP32 por MQTT cuando el firmware está en línea, o muestra el proceso de flasheo cuando no lo está
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: HUB-03, GUIA-01, GUIA-02, GUIA-03, CTRL-01, CTRL-02, CTRL-03
**Success Criteria** (what must be TRUE):

  1. El hub muestra cinco destinos en este orden: 1) Guía de conexión, 2) Dashboard, 3) Engine del Modelo, 4) Labelling, 5) Control de servos; un clic en Guía o Control navega a `/guia` y `/control` respectivamente, sin romper el layout ni las convenciones de diseño
  2. La guía renderiza la escena 3D con los 6 pasos de conexión, las fichas de pin (click), la BOM y la prueba de servos; conserva los colores funcionales de los cables y responde al tema claro/oscuro del sitio
  3. El control conecta por MQTT sobre WebSocket (`wss://broker.emqx.io:8084/mqtt`) sin backend intermedio; con el ESP32 en línea presenta el panel autoconfigurado por la telemetría y mueve los servos (verificable por el `state` que retorna el ESP32); con el ESP32 ausente muestra el instructivo de flasheo
  4. El estado de la conexión (broker y ESP32) es visible y la UI evita mover varios servos a la vez (protección de brownout)
  5. Ninguna de las dos páginas rompe el sistema de diseño (sin cards, sin border-left de acento, SVG no emojis, animaciones solo transform/opacity)

**Plans**: 3 planes
Plans:
**Wave 1**

- [x] 05-01-PLAN.md — Andamiaje compartido: bun add three mqtt, rutas /guia y /control (lazy), hub extendido a 5 destinos

**Wave 2** *(parallel, blocked on Wave 1 completion)*

- [x] 05-02-PLAN.md — Guía 3D: porte de la escena Three.js a web/src/guia/ + guia.tsx con montaje SolidJS, overlays con tokens y respuesta al tema
- [ ] 05-03-PLAN.md — Control MQTT: mqtt.ts + servoProtocol.ts + servoStore.ts + control.tsx con estado dual y brownout protection

**UI hint**: yes

## Progress

**Execution Order:**
Las fases 3, 4 y 5 dependen de la Fase 2 (Hub). La Fase 5 es independiente de las Fases 3 y 4 (toca el hub y dos rutas nuevas), así que puede ejecutarse sin bloqueos una vez cerrado el Hub.

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Fundación | 3/3 | Completa | 2026-05-25 |
| 2. Hub | pragmática | Completa | 2026-05-25 |
| 3. Dashboard Live | pragmática | Completa (validada en vivo) | 2026-05-25 |
| 4. Engine y Labelling | pragmática | Completa | 2026-05-25 |
| 5. Páginas ESP32 | 2/3 | In Progress|  |
