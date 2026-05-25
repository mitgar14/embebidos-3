# Integración de páginas ESP32 al frontend Tiny Trash

Dominio: traer dos proyectos externos de Nicolás Cuarán al frontend SolidJS de
embebidos-3, adaptándolos a la estética Vercel:

1. **Guía** (`esp32-pca9685-servo-guide`): escena 3D interactiva (Three.js) del
   cableado ESP32 + placa expansora + PCA9685 + 4×SG90.
2. **Control** (`esp32-servo-mqtt-control`): control de servos del ESP32 vía MQTT,
   con estado dual (panel si el firmware está en línea; instructivo de flasheo si no).

---

## Ronda 1 — 2026-05-25 (media)

**Foco:** cerrar tres incógnitas técnicas antes de planear la Fase 5 GSD:
(1) portar una app Three.js de import-map/CDN a un bundler (Vite/Bun) dentro de
SolidJS; (2) hablar MQTT desde el navegador con `mqtt.js` sobre WSS contra el
broker público EMQX, con Last Will retenido para el estado dual; (3) gotchas de
Vite (polyfills de `mqtt.js`, imports de addons de three).

### Track A — agentes (research-code + research-web)

#### Three.js dentro de SolidJS / Vite

- **Montaje imperativo en `onMount`, limpieza en `onCleanup`.** Es el patrón
  canónico: crear `WebGLRenderer`/`Scene`/`Camera` en `onMount`, correr el loop, y
  registrar `onCleanup` *dentro* del mismo `onMount` (queda atado al scope reactivo
  del componente y se ejecuta al desmontar y en cada HMR). Confirmado por el patrón
  de `solidjs-community/solid-three` (dispose recursivo de la escena en `onCleanup`)
  y por la PR solidjs/solid#2323.

- **Loop con `renderer.setAnimationLoop(animate)`** en vez de `requestAnimationFrame`
  manual: three gestiona el rAF y basta `setAnimationLoop(null)` para detenerlo antes
  del `dispose()` (evita renders sobre un contexto WebGL ya liberado).

- **Imports con bundler** (`bun add three`):
  ```ts
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
  ```
  El `package.json` de three mapea `"./addons/*" -> "./examples/jsm/*"` en `exports`;
  Vite lo resuelve sin configuración. La ruta física `three/examples/jsm/...` también
  funciona y es más robusta si algún bundler ignora `exports`.

- **Resize:** `ResizeObserver` sobre el contenedor (mejor que `window.resize`, porque
  el layout tiene panel lateral). En el callback: `camera.aspect`,
  `camera.updateProjectionMatrix()`, `renderer.setPixelRatio(devicePixelRatio)`,
  `renderer.setSize(w, h)`.

- **Tema claro/oscuro en runtime:** `scene.background = new THREE.Color(hex)` (tiene
  prioridad sobre `setClearColor`). Reaccionar al signal de tema con un `createEffect`
  dentro del `onMount`.

- **Raycaster** para hover/click de pines (fichas técnicas): normalizar el puntero
  con `renderer.domElement.clientWidth/clientHeight` (NO `window.inner*`, porque el
  canvas no ocupa toda la ventana), `raycaster.setFromCamera(pointer, camera)`,
  `intersectObjects(meshArray, false)`.

- **`dispose()` al desmontar:** `renderer.setAnimationLoop(null)` → `controls.dispose()`
  → `scene.traverse` liberando `geometry`/`material`/`material.map` de cada mesh →
  `renderer.dispose()` → quitar el `domElement` del contenedor.

- **Versión:** estable actual **r184** (`three@0.184.0`, abr-2026). La guía original
  usa r160. Sin breaking changes en lo que usamos entre r160 y r184, salvo: builds UMD
  eliminados (r161; ya usamos ESM) y `OrbitControls` deriva de `Controls` (r168; solo
  afecta anotaciones TS). Decisión: usar la versión que instale `bun add three`.

#### mqtt.js en el navegador (MQTT sobre WebSocket)

- **Import sin polyfills:** `import mqtt from 'mqtt'`. El `vite-example` oficial de
  `mqttjs/MQTT.js` usa exactamente eso con un `vite.config` mínimo y **sin** plugins de
  polyfill (`Buffer`/`process`/`readable-stream` ya van embebidos en el bundle de la
  librería). Para Vite 8 con target moderno no hace falta nada extra.
  - Issue #1733 (`ambiguous indirect export: default`) afectaba solo a v5.2.0; resuelto
    en versiones posteriores. Si reaparece, fijar versión reciente o
    `import { connect } from 'mqtt/dist/mqtt.min'`.

- **Conexión:**
  ```ts
  import mqtt, { type IClientOptions } from 'mqtt';
  const opts: IClientOptions = {
    clientId: `tiny-trash-${Math.random().toString(16).slice(2, 8)}`,
    clean: true,
    reconnectPeriod: 2000,   // ms; 0 = sin reconexión
    connectTimeout: 30_000,
    keepalive: 60,
    // Last Will del navegador (opcional; el LWT que importa es el del ESP32):
  };
  const client = mqtt.connect('wss://broker.emqx.io:8084/mqtt', opts);
  ```

- **Eventos:** `connect`, `message`, `reconnect`, `close`, `error`. Suscribirse
  dentro del handler de `connect` para que las suscripciones se re-registren en cada
  reconexión (con `clean: true`).

- **Mensajes retenidos:** al suscribirse a un topic con un retained, el broker lo
  entrega de inmediato por el mismo callback `message`, con `packet.retain === true`.
  No hay evento especial. Esto es exactamente lo que necesita el estado dual: el ESP32
  publica `online` retenido al conectar y declara un Will `offline` retenido; el
  navegador, al suscribirse a `servos/<id>/online`, recibe el último estado al instante.

- **Last Will (`will`)** en las opciones del cliente: `{ topic, payload, qos, retain }`
  (+ `properties.willDelayInterval` solo en MQTT 5.0). El will se dispara únicamente en
  desconexión anómala; un `client.end()` limpio no lo lanza.

#### Broker público EMQX (`broker.emqx.io`)

| Protocolo | Puerto | URL | Uso |
|-----------|--------|-----|-----|
| WS plano | 8083 | `ws://broker.emqx.io:8083/mqtt` | navegador sobre HTTP |
| WSS (TLS) | 8084 | `wss://broker.emqx.io:8084/mqtt` | navegador sobre HTTPS |

- El path **`/mqtt` es obligatorio**; omitirlo rompe la conexión.
- **Mixed content:** una SPA servida por HTTPS DEBE usar `wss://` (el navegador bloquea
  `ws://` plano). En `localhost` (HTTP) sirve cualquiera; usaremos `wss` por defecto,
  que también funciona desde HTTP.
- Sin usuario/contraseña (anónimo). Es **público y compartido**: "todos los mensajes son
  visibles, no enviar datos sensibles". Mitigación para el MVP: `DEVICE_ID` propio,
  largo y poco adivinable, para evitar colisiones de topics con otros usuarios.

### Track B — discover.py + lectura activa

Hallazgos más útiles del descubrimiento amplio (22 resultados):

- **`emqx/MQTT-Client-Examples` → `mqtt-client-WebSocket/ws-mqtt.html`**: ejemplo
  mínimo oficial de MQTT sobre WebSocket en HTML plano. Buena referencia del flujo
  connect/subscribe/publish sin framework.
- **`mqttjs/MQTT.js` → `examples/vite-example/`**: prueba de que `mqtt.js` corre en
  Vite sin polyfills.
- **`UstymUkhman/threejs-boilerplate`**: plantilla Three.js + TypeScript + **SolidJS** +
  Vite + Vitest. Referencia directa del stack exacto que vamos a usar.
- **`solidjs-community/solid-three`** y **`vorth/solid-three-sample`**: port de
  react-three-fiber a Solid; útiles para el patrón de ciclo de vida (no los usaremos
  como dependencia: montamos three imperativo, más fiel al repo original de la guía).
- **EMQX blogs** (`mqtt-js-tutorial`, `how-to-use-mqtt-in-react`,
  `connect-to-mqtt-broker-with-websocket`): confirman endpoints, will y el patrón de
  reconexión.

### Decisiones de arquitectura derivadas (para la Fase 5)

**Control de servos (`/control`):**
- El navegador habla **MQTT directo sobre WSS** con `mqtt.js`; se elimina el puente
  FastAPI/paho del repo original (no encaja en una SPA estática sin backend propio, y
  el backend del Nano es read-only de visión). El firmware del ESP32 **no cambia**.
- Topics derivados del `DEVICE_ID`: `servos/<id>/cmd` (publish), `servos/<id>/state`
  (subscribe, telemetría), `servos/<id>/online` (subscribe, LWT retenido).
- **Estado dual** por el retenido de `online`: `online` → panel de control; `offline`
  o sin retenido tras un timeout → instructivo de flasheo (Arduino IDE, librerías,
  `secrets.h`/`config.h`, subir, monitor serie) + botón de reintento.
- La UI **se autoconfigura** con el `state` (num_servos, channels, angles, presets):
  no se hardcodean canales ni calibración. Resuelve la discrepancia
  `{0,7,8,15}` (firmware) vs `CH0–CH3` (guía).
- **Brownout** (4×SG90 desde USB, AMS1117 ~600 mA vs picos de varios servos): la UI
  serializa los movimientos masivos ("centrar todos"/"cargar preset en todos") en pasos
  pequeños y con retardo, y desalienta mover varios a la vez. Comandos de a uno.
- Comandos JSON (del repo): `move`, `move_all`, `save_preset`, `load_preset`,
  `load_all_preset`, `request_state`. Se publican tal cual en `cmd`.

**Guía de conexión (`/guia`):**
- **La guía SIGUE SIENDO 3D (Three.js), tal cual el repo.** No se sustituye por nada 2D:
  el 3D es la esencia y se conserva entero (escena, cámara de órbita y de vuelo, cables
  progresivos, fichas de pin, prueba de servos). Montarla en un componente SolidJS
  significa solo que la MISMA escena 3D pasa a vivir dentro del sitio (su contenedor es
  ahora un componente Solid con `onMount`); no se reescribe ni se aplana. Se reutilizan
  los archivos del repo (`components/`, `core/`, `data/connections.js`, `ui/`) casi sin
  tocar: lo único que cambia es el arranque (`onMount` en vez de `DOMContentLoaded`), el
  import de three (`bun add three` en vez de import-map CDN) y el revestimiento del
  cromado HTML (paneles, tooltips, modal, fondo de la escena según el tema) con los
  tokens del sitio.
- `bun add three`; montar en `onMount`, `setAnimationLoop`, `onCleanup` con dispose.
- Reutilizar `connections.js` (pines, 6 pasos, BOM) como fuente de verdad.
- Re-estilar los overlays HTML (welcome modal, BOM, info panel, tooltip, ficha de pin)
  con los tokens del sitio (Geist, Radix Slate, monocromo, tema claro/oscuro, sin cards).
- **Conservar los colores funcionales de los cables** (GND negro, VCC rojo, SDA azul,
  SCL amarillo, V+ rojo grueso, OE blanco): son pedagógicos, como los colores Wong.
- `scene.background` reacciona al signal de tema.

**Hub:** de 3 → 5 destinos, reutilizando el patrón `DestTile` (`flex-wrap` reacomoda).

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-25 | media | Three.js en SolidJS/Vite; mqtt.js en navegador (WSS + LWT retenido); endpoints EMQX; gotchas Vite |

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | Free Public MQTT Broker (EMQX) | https://www.emqx.com/en/mqtt/public-mqtt5-broker | Doc oficial | 1 |
| 2 | A Quickstart Guide to MQTT over WebSocket (EMQX) | https://www.emqx.com/en/blog/connect-to-mqtt-broker-with-websocket | Blog oficial | 1 |
| 3 | EMQX Listener Configuration (WebSocket path) | https://docs.emqx.com/en/emqx/latest/configuration/listener.html | Doc oficial | 1 |
| 4 | MQTT.js README (IClientOptions, will, reconnectPeriod) | https://github.com/mqttjs/mqtt.js/blob/main/README.md | README oficial | 1 |
| 5 | MQTT.js llms.txt (Context7) | https://context7.com/mqttjs/mqtt.js/llms.txt | Doc derivada | 1 |
| 6 | MQTT.js vite-example | https://github.com/mqttjs/MQTT.js/tree/main/examples/vite-example | Código oficial | 1 |
| 7 | EMQX MQTT-Client-Examples (ws-mqtt.html) | https://github.com/emqx/MQTT-Client-Examples/blob/master/mqtt-client-WebSocket/ws-mqtt.html | Código oficial | 1 |
| 8 | JavaScript MQTT Client: A Beginner's Guide to MQTT.js | https://www.emqx.com/en/blog/mqtt-js-tutorial | Blog oficial | 1 |
| 9 | How to Use MQTT in The React Project | https://www.emqx.com/en/blog/how-to-use-mqtt-in-react | Blog oficial | 1 |
| 10 | Three.js homepage (r184) | https://threejs.org | Doc oficial | 1 |
| 11 | Three.js Migration Guide (r160–r184) | https://github.com/mrdoob/three.js/wiki/Migration-Guide | Wiki oficial | 1 |
| 12 | Three.js OrbitControls (docs) | https://threejs.org/docs/#examples/en/controls/OrbitControls | Doc oficial | 1 |
| 13 | npm three@0.184.0 | https://www.npmjs.com/package/three | Registro npm | 1 |
| 14 | solidjs-community/solid-three | https://github.com/solidjs-community/solid-three | Repo | 1 |
| 15 | UstymUkhman/threejs-boilerplate (Three+TS+Solid+Vite) | https://github.com/UstymUkhman/threejs-boilerplate | Repo | 1 |
| 16 | PR solidjs/solid#2323 (onCleanup dentro de onMount) | https://github.com/solidjs/solid/pull/2323 | PR | 1 |
| 17 | donmccurdy/three-gltf-viewer (setClearColor + resize) | https://github.com/donmccurdy/three-gltf-viewer | Repo | 1 |
