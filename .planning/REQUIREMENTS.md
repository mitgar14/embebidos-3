# Requirements: embebidos-3 — Consola Web

**Defined:** 2026-05-24
**Core Value:** El Dashboard live (canvas de detección vía WebSocket al Nano, reconexión robusta) debe verse fluido y profesional en la sustentación del 2026-05-26.

## v1 Requirements

Requisitos para la sustentación. Cada uno mapea a una fase del roadmap.

### Fundación y Sistema de Diseño

- [ ] **DES-01**: La app arranca como un proyecto SolidJS + Vite + Bun + Tailwind 4 en `web/`, navegable con HashRouter
- [ ] **DES-02**: La UI usa un sistema de diseño unificado (tokens Radix Slate + IBM Cyan, IBM Plex Sans/Mono) con estética plana inspirada en Vercel/Geist
- [ ] **DES-03**: La UI no usa cards ni `border-left` de acento; el estado se comunica con dots/pills/tint; los iconos son SVG (no emojis)
- [ ] **DES-04**: Las animaciones son mínimas y solo sobre `transform`/`opacity` (compositor), respetando `prefers-reduced-motion`
- [ ] **DES-05**: El usuario puede alternar entre tema claro y oscuro mediante un toggle; la preferencia se persiste y el primer arranque respeta `prefers-color-scheme` (fallback oscuro). Ambos temas usan la paleta fría (Radix Slate claro/oscuro + IBM Cyan) sin romper contraste

### Conexión con el Nano

- [ ] **CONN-01**: Existe un cliente WebSocket reutilizable que se reconecta automáticamente con backoff exponencial + jitter + heartbeat (ping/pong)
- [ ] **CONN-02**: El frontend detecta el sistema operativo (Windows/macOS) para decidir comandos OS-específicos
- [ ] **CONN-03**: El usuario puede configurar la URL del servidor (WS del Nano) y la app la persiste

### Hub de Inicio

- [ ] **HUB-01**: El usuario ve un hub command-center con 3 destinos (Dashboard, Engine del Modelo, Labelling) y navega a cualquiera con un clic
- [ ] **HUB-02**: El usuario ve el estado del Nano (conectado / última inferencia / salud) inline en el hub
- [x] **HUB-03**: El hub muestra cinco destinos (Dashboard, Engine del Modelo, Labelling, Guía de conexión y Control de servos), siguiendo el patrón de cubos existente (sin romper el layout con `flex-wrap`)

### Dashboard Live

- [ ] **DASH-01**: El usuario ve el stream de cámara con overlay de bounding boxes de detección a ~14 fps
- [ ] **DASH-02**: El usuario ve métricas en vivo: fps, latencia, temperatura GPU, RAM y conteo por clase (vidrio/papel/plástico)
- [ ] **DASH-03**: El usuario puede ajustar fuente de cámara, umbral de confianza y fps objetivo, y los cambios aplican en vivo
- [ ] **DASH-04**: El Dashboard usa el cliente WS reconectante y muestra el estado de conexión sin congelarse ante caídas
- [ ] **DASH-05**: El usuario puede capturar un snapshot (PNG) del frame actual con sus bounding boxes

### Centro del Engine del Modelo

- [ ] **ENG-01**: El usuario ve el estado actual del modelo (cargado, motor/engine, timestamp del último build)
- [ ] **ENG-02**: El usuario puede disparar un build del modelo desde la interfaz
- [ ] **ENG-03**: El usuario ve los logs del build en vivo (stream SSE) con auto-scroll

### Labelling

- [ ] **LBL-01**: El usuario puede ver una imagen del dataset y navegar entre imágenes (anterior/siguiente)
- [ ] **LBL-02**: El usuario puede dibujar, mover y redimensionar bounding boxes sobre la imagen
- [ ] **LBL-03**: El usuario puede asignar una de las 3 clases (vidrio/papel/plástico) a cada caja, con color por clase
- [ ] **LBL-04**: El usuario puede exportar las anotaciones del frame en formato YOLO
- [ ] **LBL-05**: El usuario puede navegar entre imágenes mediante una tira de thumbnails (miniaturas)

### Guía de Conexión (ESP32 · PCA9685 · servos)

- [x] **GUIA-01**: El usuario ve una guía interactiva en 3D del cableado entre la placa expansora del ESP32, el PCA9685 y los servos SG90, y recorre los 6 pasos de conexión (GND, VCC, SDA, SCL, V+, OE) con la escena animándose en cada paso
- [x] **GUIA-02**: La guía conserva la riqueza del original (cámara de órbita y de vuelo, cables que aparecen de forma progresiva, fichas técnicas por pin al hacer clic, lista de materiales y prueba de servos) pero revestida con la estética del sitio (Geist, Radix Slate monocromo, sin cards, SVG no emojis), conservando los colores funcionales de los cables (GND/VCC/SDA/SCL/V+/OE)
- [x] **GUIA-03**: La guía responde al tema claro/oscuro del sitio (fondo de la escena incluido) y es accesible desde el hub

### Control de Servos (ESP32 vía MQTT)

- [ ] **CTRL-01**: El usuario controla los servos conectados al ESP32 desde el navegador, que habla MQTT sobre WebSocket directamente con el broker (sin backend intermedio), publicando los comandos en los topics del firmware
- [ ] **CTRL-02**: La página tiene dos estados según el firmware: si el ESP32 está en línea (Last Will retenido) muestra el panel de control (un control por servo con ángulo y presets, autoconfigurado por la telemetría que reporta el ESP32); si no, muestra el instructivo para flashear y poner en marcha el firmware
- [ ] **CTRL-03**: La UI muestra el estado de la conexión (broker y ESP32) y respeta la restricción de alimentación (evita mover varios servos a la vez para no provocar brownout)

### Orquestación

- [ ] **OPS-01**: El usuario puede levantar todo el frontend con `web.ps1` (start) y apagarlo limpiamente con `web.ps1 -Action stop` (mata el árbol de procesos)
- [ ] **OPS-02**: `web.ps1` funciona en Windows y macOS (pwsh), ramificando los comandos OS-específicos con `$IsWindows`/`$IsMacOS`
- [ ] **OPS-03**: Al arrancar, `web.ps1` construye/sirve el frontend, verifica la salud del Nano y abre el navegador

## v2 Requirements

Diferido a futuro. Registrado pero fuera del roadmap actual.

### Mejoras

- **V2-01**: Command palette (⌘K) propia
- **V2-02**: Offload del decode JPEG a Web Worker + OffscreenCanvas (solo si el decode tranca el main thread)

## Out of Scope

Excluido explícitamente para prevenir scope creep.

| Feature | Reason |
|---------|--------|
| Reescribir/modificar el backend FastAPI del Nano | Ya existe y funciona; el frontend solo lo consume |
| Puente FastAPI/paho del repo de control de servos | El navegador habla MQTT sobre WebSocket directo; el bridge Python sobra en una SPA estática |
| Re-flashear el ESP32 desde el navegador | No es posible/seguro; la Guía y el instructivo de Control explican el proceso con Arduino IDE |
| Autenticación / multiusuario | Innecesario para la demo local |
| Servidor de producción endurecido | `vite preview` del build basta para la sustentación |
| Re-entrenar o cambiar el modelo de detección | Fuera del alcance del refactor de interfaces |

## Traceability

Qué fases cubren qué requisitos.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DES-01 | Phase 1 — Fundación | Pending |
| DES-02 | Phase 1 — Fundación | Pending |
| DES-03 | Phase 1 — Fundación | Pending |
| DES-04 | Phase 1 — Fundación | Pending |
| DES-05 | Phase 1 — Fundación | Pending |
| CONN-01 | Phase 1 — Fundación | Pending |
| CONN-02 | Phase 1 — Fundación | Pending |
| CONN-03 | Phase 1 — Fundación | Pending |
| OPS-01 | Phase 1 — Fundación | Pending |
| OPS-02 | Phase 1 — Fundación | Pending |
| OPS-03 | Phase 1 — Fundación | Pending |
| HUB-01 | Phase 2 — Hub | Pending |
| HUB-02 | Phase 2 — Hub | Pending |
| DASH-01 | Phase 3 — Dashboard Live | Pending |
| DASH-02 | Phase 3 — Dashboard Live | Pending |
| DASH-03 | Phase 3 — Dashboard Live | Pending |
| DASH-04 | Phase 3 — Dashboard Live | Pending |
| DASH-05 | Phase 3 — Dashboard Live | Pending |
| ENG-01 | Phase 4 — Engine y Labelling | Pending |
| ENG-02 | Phase 4 — Engine y Labelling | Pending |
| ENG-03 | Phase 4 — Engine y Labelling | Pending |
| LBL-01 | Phase 4 — Engine y Labelling | Pending |
| LBL-02 | Phase 4 — Engine y Labelling | Pending |
| LBL-03 | Phase 4 — Engine y Labelling | Pending |
| LBL-04 | Phase 4 — Engine y Labelling | Pending |
| LBL-05 | Phase 4 — Engine y Labelling | Pending |
| HUB-03 | Phase 5 — Páginas ESP32 | Complete |
| GUIA-01 | Phase 5 — Páginas ESP32 | Complete |
| GUIA-02 | Phase 5 — Páginas ESP32 | Complete |
| GUIA-03 | Phase 5 — Páginas ESP32 | Complete |
| CTRL-01 | Phase 5 — Páginas ESP32 | Pending |
| CTRL-02 | Phase 5 — Páginas ESP32 | Pending |
| CTRL-03 | Phase 5 — Páginas ESP32 | Pending |

**Coverage:**
- v1 requirements: 33 total
- Mapped to phases: 33
- Unmapped: 0 ✓

---
*Requirements defined: 2026-05-24*
*Last updated: 2026-05-25 — añadida Fase 5 (integración de páginas ESP32): GUIA-01..03, CTRL-01..03 y HUB-03; cobertura 33/33*
