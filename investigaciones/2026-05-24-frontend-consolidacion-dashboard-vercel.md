# Consolidación de interfaces web — refactor estilo Vercel/Geist

**Proyecto:** `embebidos-3` — clasificador YOLOv8n FP16 (3 clases `plastic`/`paper`/`glass`) sobre banda transportadora, deployado en Jetson Nano. Demo **2026-05-26**.

**Objetivo del refactor:** consolidar las interfaces web (dashboard de detección en vivo + gestión de modelo, y la UI de revisión de labels) en **una carpeta dedicada en la raíz**, bajo una estética tipo **Vercel/Geist**, multiplataforma Windows/Apple, con **detección de SO** para mostrar comandos correctos, y robustecimiento de la **conexión WS al Nano**.

**Estado de partida (auditoría de código, 2026-05-24):**
- `scripts/dashboard/` — vanilla JS (ES2020+), 2 pestañas (`live`/`modelo`), sistema OKLCH con acento **mostaza/ocre tibio** (hue 80), `ui.js` con toast/modal/`withButtonLoading`, WS binario al Nano, polling de `/health` y `/model/state`, SSE de logs de build. Sirve estático **en local**.
- ⚠️ **Fuentes borradas**: `SourceSans3VF.woff2` y `SourceCodeVF.woff2` ya no existen → `@font-face` roto, cae a fallback del sistema. Es el momento natural para migrar tipografía.
- ⚠️ **`scripts/launch_demo.py` referenciado en README pero NO existe** (drift doc↔código). El refactor incluirá un launcher multiplataforma propio.
- `scripts/labeling/label_review/app.py` — UI funcional pero dark básica con estilos inline (`#1a1a1a`), sin design system.
- ⚠️ **Inconsistencia clase→color**: labeling usa `plastic=rojo / paper=verde / glass=azul`; dashboard usa `glass=verde / paper=azul / plastic=naranja`. Incompatibles entre sí.
- ⚠️ **Gap de robustez WS**: `app.js` abre el socket una vez; `onclose`/`onerror` solo marcan "inactiva". Sin auto-reconnect, backoff ni heartbeat — frágil sobre DERP/Headscale.
- El `nano_server.py` NO sirve la UI (root devuelve `PlainTextResponse`); expone WS `/ws` (frames JPEG binarios → JSON `bboxes`+`t_infer_ms`+`seq`; control `{type:conf}`→`conf_ack`, `{type:ping}`→`pong`), `/health`, `/model/*`, `/jobs/*` (SSE).

---

## Ronda 1 — 2026-05-24 (alto)

Foco: lenguaje de diseño Geist y tokens · detección de SO en browser · reconexión WS robusta · craft de interacción (Sonner/Rauno/Emil) · arquitectura de consolidación.

### Track A — agentes de research

#### A.1 · Geist: distribución de fuente y tokens

**Distribución self-hosted de la fuente.** El repo `vercel/geist` es privado; el público es **`vercel/geist-font`** (SIL OFL 1.1, v1.7.1 del 2026-05-20). La **variable font** cubre todo el rango `font-weight: 100 900` en un solo archivo por familia:

| Opción | Archivo clave | Peso |
|---|---|---|
| npm `geist` | `dist/fonts/geist-sans/Geist-Variable.woff2` | 68 KB (todos los pesos) |
| `@fontsource-variable/geist` | `Geist[wght].woff2` | eje `wght` 100-900 |
| repo `vercel/geist-font` | `packages/next/dist/fonts/…` | self-host directo |

```css
@font-face {
  font-family: 'Geist Sans';
  src: url('fonts/Geist-Variable.woff2') format('woff2');
  font-weight: 100 900; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'Geist Mono';
  src: url('fonts/GeistMono-Variable.woff2') format('woff2');
  font-weight: 100 900; font-style: normal; font-display: swap;
}
```

**Semántica de la escala de color** (de `vercel.com/geist/colors`, Track B): 10 escalas (Gray, Gray-alpha, Blue, Red, Amber, Green, Teal, Purple, Pink + Backgrounds). El sistema es **numerado y reutilizable** por rol:

| Pasos | Rol |
|---|---|
| **1-3** | Fondos de componente (1=default, 2=hover, 3=active) |
| **4-6** | Bordes (4=default, 5=hover, 6=active) |
| **7-8** | Fondos alto contraste (7=default, 8=hover) |
| **9-10** | Texto/iconos (9=secundario, 10=primario) |

En **dark mode la escala se invierte funcionalmente** (gray-100 = más oscuro/superficie, gray-1000 = más claro/texto) pero **el rol se mantiene**.

**Token set dark canónico** (cross-verificado: `vercel.com/geist/colors` + CSS de Sonner leído por GitHub API + gist de Anthony Shew, ex-Vercel). Los valores semánticos success/error/warning/info son byte-by-byte del CSS de Sonner:

```css
:root, [data-theme="dark"] {
  --font-sans: 'Geist Sans', ui-sans-serif, system-ui, sans-serif;
  --font-mono: 'Geist Mono', ui-monospace, 'Cascadia Code', monospace;

  /* Grises (dark): 100=superficie … 1000=texto high-contrast */
  --gray-100:#1A1A1A; --gray-200:#1F1F1F; --gray-300:#292929;
  --gray-400:#2E2E2E; --gray-500:#454545; --gray-600:#878787;
  --gray-700:#8F8F8F; --gray-800:#A1A1A1; --gray-900:#EDEDED; --gray-1000:#FFFFFF;

  --gray-alpha-100:rgba(255,255,255,.04); --gray-alpha-200:rgba(255,255,255,.06);
  --gray-alpha-400:rgba(255,255,255,.1);  --gray-alpha-700:rgba(255,255,255,.3);

  --background-100:#0A0A0A;  /* página */   --background-200:#000;  /* fondo absoluto */
  --foreground-100:#EDEDED;  --foreground-600:#878787;

  /* Azul de acento Vercel */
  --blue-700:#0070F3;  --blue-900:#52A8FF;

  /* Semánticos (dark) — del CSS de Sonner */
  --success-bg:hsl(150,100%,6%);  --success-border:hsl(147,100%,12%);  --success-text:hsl(150,86%,65%);
  --error-bg:hsl(358,76%,10%);    --error-border:hsl(357,89%,16%);     --error-text:hsl(358,100%,81%);
  --warning-bg:hsl(64,100%,6%);   --warning-border:hsl(60,100%,9%);    --warning-text:hsl(46,87%,65%);
  --info-bg:hsl(215,100%,6%);     --info-border:hsl(223,43%,17%);      --info-text:hsl(216,87%,65%);

  /* Radios (de vercel.com/geist/materials) */
  --radius-sm:4px;   --radius-md:6px;  /* base/tooltip */
  --radius-lg:12px;  /* menu/modal */  --radius-xl:16px; /* fullscreen */ --radius-full:9999px;

  /* Sombras: Vercel prefiere BORDES; sólo floating lleva sombra mínima */
  --shadow-tooltip:0 4px 12px rgba(0,0,0,.35);
  --shadow-menu:0 8px 30px rgba(0,0,0,.45);
  --shadow-modal:0 12px 48px rgba(0,0,0,.55);

  /* Espacio (escala 4px) */
  --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px;
  --space-5:20px; --space-6:24px; --space-8:32px; --space-10:40px;
}
[data-theme="light"] {
  --background-100:#FFF; --background-200:#FAFAFA; --foreground-100:#171717; --foreground-600:#666;
  --gray-100:#FAFAFA; --gray-200:#EAEAEA; --gray-300:#E0E0E0; --gray-400:#CBCBCB;
  --gray-500:#999; --gray-600:#666; --gray-700:#444; --gray-800:#333; --gray-900:#171717; --gray-1000:#000;
}
```

**Tipografía** — escala oficial (`text-heading-*`, `text-label-*`, `text-copy-*`, valor = px). Para dashboard técnico:

| Token | px | Uso |
|---|---|---|
| `heading-32/24/20/16` | 32/24/20/16 | título de página → card title (peso 500-600, ls −0.02em→−0.01em) |
| `label-14` | 14 | el más común: menús, tablas, listas |
| `label-13` / `label-13-mono` | 13 | secundario / **datos técnicos densos** |
| `label-12` | 12 | terciario, category labels (caps, ls 0.06em) |
| `copy-14` / `copy-13-mono` | 14/13 | cuerpo / código inline, IPs, paths |

**Regla mono:** todo lo técnico/tabular/longitud variable va en Geist Mono con `font-variant-numeric: tabular-nums`: clase predicha (`PLÁSTICO`), confianza (`0,94`), timestamp, IP, temperatura, nº de frame, hashes. **Pesos nunca > 600**; el contraste lo da el tracking negativo, no el bold.

**Bordes vs sombras:** componentes "on page" (`material-base/small`) usan `border: 1px` SIN sombra; solo floating (menu/modal/tooltip) agrega sombra mínima. Es deliberado.

#### A.2 · WebSocket: reconexión robusta

**Algoritmos reales analizados:**
- `pladaria/reconnecting-websocket` (MIT): `delay = minDelay · growFactor^(retry-1)`, cap a `maxDelay`. Defaults: `min 1000+random·4000` (jitter en la base, una vez), `grow 1.3`, `connectionTimeout 4000`, `minUptime 5000` (uptime mínimo para resetear el contador). **Sin heartbeat integrado.**
- `zimv/WebSocketHeartBeat` (MIT): `pingTimeout 15000` (idle antes de ping), `pongTimeout 10000` (espera del pong) → si no llega, `ws.close()` dispara reconexión.

**Clave de browser (Track B, websocket.org):** el browser **NO expone** ping/pong de protocolo a JS; hay que hacer heartbeat **a nivel app** con `{type:ping}`/`{type:pong}` — que **el `nano_server.py` ya implementa**. Enviar heartbeat inmediato en `visibilitychange` al volver a foreground (laptop que despierta de sleep en la sustentación).

**Clase destilada `ReconnectingWebSocket` (vanilla, ~170 LOC):** full jitter + heartbeat ping/pong + cola offline + reintentos infinitos con tope, `binaryType=arraybuffer` para frames JPEG, filtra pongs silenciosamente.

```javascript
/** ReconnectingWebSocket — vanilla, sin deps. Frames JPEG binarios + control JSON.
 *  Server responde {type:"pong"} a {type:"ping"}.
 *  Fuentes: pladaria/reconnecting-websocket (MIT), zimv/WebSocketHeartBeat (MIT). */
class ReconnectingWebSocket extends EventTarget {
  static CONNECTING=0; static OPEN=1; static CLOSING=2; static CLOSED=3;
  constructor(url, {
    minDelay=1000, maxDelay=30000, growFactor=1.5, connectionTimeout=5000,
    minUptime=3000, pingInterval=20000, pongTimeout=8000, binaryType='arraybuffer',
  } = {}) {
    super();
    this._url=url;
    this._opts={minDelay,maxDelay,growFactor,connectionTimeout,minUptime,pingInterval,pongTimeout,binaryType};
    this._ws=null; this._retryCount=0; this._messageQueue=[];
    this._shouldReconnect=true; this._connectLock=false; this._closeCalled=false;
    this._connectTimer=this._uptimeTimer=this._pingTimer=this._pongTimer=null;
    this._connect();
  }
  get readyState(){ return this._ws ? this._ws.readyState : ReconnectingWebSocket.CONNECTING; }
  send(data){ if (this._ws?.readyState===WebSocket.OPEN) this._ws.send(data); else this._messageQueue.push(data); }
  close(code=1000, reason=''){ this._closeCalled=true; this._shouldReconnect=false; this._clearTimers(); this._ws?.close(code,reason); }
  reconnect(){ this._shouldReconnect=true; this._closeCalled=false; this._retryCount=0; this._ws ? this._ws.close(1000) : this._connect(); }
  _getDelay(){ // full jitter
    const exp=this._opts.minDelay*Math.pow(this._opts.growFactor,this._retryCount);
    return Math.random()*Math.min(exp,this._opts.maxDelay);
  }
  _connect(){
    if (this._connectLock || !this._shouldReconnect) return;
    this._connectLock=true;
    const delay = this._retryCount===0 ? 0 : this._getDelay();
    setTimeout(() => {
      if (this._closeCalled){ this._connectLock=false; return; }
      const ws=new WebSocket(this._url); ws.binaryType=this._opts.binaryType;
      this._ws=ws; this._connectLock=false;
      this._connectTimer=setTimeout(()=>ws.close(), this._opts.connectionTimeout);
      ws.addEventListener('open', e => {
        clearTimeout(this._connectTimer);
        this._uptimeTimer=setTimeout(()=>{ this._retryCount=0; }, this._opts.minUptime);
        this._messageQueue.forEach(m=>ws.send(m)); this._messageQueue=[];
        this._schedulePing();
        this.dispatchEvent(Object.assign(new Event('open'),{originalEvent:e}));
      });
      ws.addEventListener('message', e => {
        this._resetHeartbeat();
        if (typeof e.data==='string'){ try { if (JSON.parse(e.data)?.type==='pong') return; } catch(_){} }
        this.dispatchEvent(new MessageEvent('message',{data:e.data}));
      });
      ws.addEventListener('close', e => {
        this._clearTimers();
        this.dispatchEvent(Object.assign(new Event('close'),{code:e.code,reason:e.reason}));
        if (this._shouldReconnect){ this._retryCount++; this._connect(); }
      });
      ws.addEventListener('error', () => this.dispatchEvent(new Event('error')));
    }, delay);
  }
  _schedulePing(){
    clearTimeout(this._pingTimer);
    this._pingTimer=setTimeout(()=>{
      if (this._ws?.readyState===WebSocket.OPEN){
        this._ws.send(JSON.stringify({type:'ping'}));
        this._pongTimer=setTimeout(()=>this._ws?.close(), this._opts.pongTimeout);
      }
    }, this._opts.pingInterval);
  }
  _resetHeartbeat(){ clearTimeout(this._pongTimer); this._schedulePing(); }
  _clearTimers(){ clearTimeout(this._connectTimer); clearTimeout(this._uptimeTimer); clearTimeout(this._pingTimer); clearTimeout(this._pongTimer); }
}
```

**Parámetros para el dashboard** (endpoint real `ws://100.64.0.2:8000/ws` sobre Headscale/DERP):

| Param | Valor | Razón |
|---|---|---|
| `minDelay` / `maxDelay` | 1000 / 30000 | primer reintento rápido, tope 30 s |
| `growFactor` | 1.5 | más suave que 1.3 para evitar bursts |
| `connectionTimeout` | 5000 | el Nano puede tardar en levantar el server |
| `pingInterval` / `pongTimeout` | 20000 / 8000 | detecta Nano dormida; 8 s generoso para DERP |

> Nota de integración: la clase es **capa de transporte**. La lógica del dashboard (correlación por `seq`, `inFlight`, `pendingFrames`) se monta encima. Durante streaming activo cada respuesta de frame resetea el heartbeat (los pings solo disparan en idle, p. ej. cámara detenida).

#### A.3 · Detección de SO en browser

**Estado 2026:** `navigator.userAgentData.platform` → `"macOS"`/`"Windows"`/`"Linux"` (Chromium ≥90, **solo secure context HTTPS**; **undefined en Firefox y Safari**). Fallback obligatorio a `navigator.platform` (deprecado pero universal: `"MacIntel"`/`"Win32"`/`"Linux x86_64"`), luego `userAgent`.

> **Crítico para este proyecto:** el dashboard corre en **HTTP local** (no HTTPS) → `userAgentData` **nunca** estará disponible. El camino correcto y confiable es `navigator.platform`.

```javascript
/** detectOS — 'mac'|'windows'|'linux'|'unknown'. Fuentes: darkreader, zellij (MIT). */
function _platformString(){
  if (typeof navigator==='undefined') return '';
  if (navigator.userAgentData?.platform) return navigator.userAgentData.platform.toLowerCase();
  return (navigator.platform ?? '').toLowerCase();
}
function detectOS(){
  const p=_platformString();
  if (p.startsWith('mac')||p.includes('darwin')) return 'mac';
  if (p.startsWith('win')) return 'windows';
  if (p.includes('linux')) return 'linux';
  const ua=(navigator.userAgent||'').toLowerCase();
  if (ua.includes('mac')) return 'mac';
  if (ua.includes('win')) return 'windows';
  if (ua.includes('linux')) return 'linux';
  return 'unknown';
}
function modifierKey(){ return detectOS()==='mac' ? '⌘' : 'Ctrl'; }
```

**Componente `Kbd` adaptativo** (convención Geist `vercel.com/geist/keyboard-input`): "hard-codear `⌘` envía el glifo equivocado a la mitad de tus usuarios". Mapear modificadores por SO:

| SO | meta | alt | shift |
|---|---|---|---|
| mac | ⌘ | ⌥ | ⇧ |
| windows/linux | Ctrl/Super | Alt | Shift |

**Bloques de comando por SO** (caso conexión Headscale/SSH) — el patrón que el usuario pidió explícitamente:

```javascript
function commandFor(os, { windows, unix }){ return os==='windows' ? windows : unix; }
// Ej. Headscale:
//   windows: tailscale up --login-server=https://80-241-217-130.nip.io --authkey=… --accept-dns=false
//   unix:    sudo tailscale up --login-server=https://80-241-217-130.nip.io --authkey=… --accept-dns=false
// Ej. SSH:   ssh -i ~/.ssh/id_ed25519 jetson@100.64.0.2   (idéntico, label de shell distinto)
// label de shell: windows='PowerShell', mac='Terminal (zsh)', linux='Terminal (bash)'
```

#### A.4 · Sonner y patrones de toast/animación vanilla

**Constantes de timing (de `emilkowalski/sonner` `src/index.tsx`):** `VISIBLE_TOASTS 3`, `VIEWPORT_OFFSET 24px`, `LIFETIME 4000`, `WIDTH 356`, `GAP 14`, `SWIPE_THRESHOLD 45`, `TIME_BEFORE_UNMOUNT 200`. Entrada 400ms `ease` (`transform,opacity,height,box-shadow`); salida `transform` 400ms / `opacity` 200ms; swipe-out 200ms `ease-out`.

**Patrones replicables en vanilla** (el proyecto ya tiene toast/modal propios en `ui.js` — elevarlos sin React):
- **Stacking:** cada toast `position:absolute` con `--offset` (suma de alturas previas + GAP) y `--toasts-before` recalculados en JS al agregar/quitar.
- **Expand on hover:** togglear `data-expanded` en el contenedor.
- **Mount:** `appendChild` → forzar reflow (`getBoundingClientRect()`) → `data-mounted=true` dispara la transición.
- **Swipe-to-dismiss:** pointer events, umbral 45px, `data-swipe-out` + `setTimeout(onDismiss, 200)`.
- **reduced-motion:** un bloque `@media (prefers-reduced-motion){ transition:none!important; animation:none!important; }`.

#### A.5 · Craft de interacción (Rauno Freiberg + Emil Kowalski)

**Checklist accionable** (priorizado para el refactor):

*Obligatorios:*
- Focus ring por **`box-shadow`**, nunca `outline` crudo. Patrón Geist: `0 0 0 2px bg, 0 0 0 4px color` (ring sobre ring).
- `font-variant-numeric: tabular-nums` en confianza, timestamps, contadores, temperatura.
- `-webkit-font-smoothing: antialiased` + `text-rendering: optimizeLegibility` en body.
- Bordes como `box-shadow: 0 0 0 1px var(--gray-alpha-400)` (preserva box model, combina con sombras).
- `@media (hover:hover)` para todos los `:hover` (evita flash en touch).
- `prefers-reduced-motion` en toda animación no trivial.
- Botones disabled SIN tooltip (no entran en tab order).

*Alto impacto visual:*
- Tokens `--gray-*` para todo fondo/borde, **cero hex fijos** en elementos que cambian de tema.
- Geist Mono para datos técnicos.
- Scale `0.97` en botones `:active`; dropdowns/tooltips escalando desde `0.93` (no desde 0).
- `transform-origin` de popovers relativo al trigger.
- Entrada con `ease-out`, ≤200ms en UI frecuente; **sin animación en acciones de teclado** (command palette, atajos).

*Detalles:*
- Labels hacen focus en su input al click; inputs dentro de `<form>` (Enter funciona).
- Toggles con efecto inmediato (UI optimista), sin modal.
- "Copiar" muestra **checkmark inline**, no toast global.
- Animar solo `transform`/`opacity` (nunca `height`/`padding`/`margin` → fuerzan layout).
- El peso de fuente NO cambia en hover/selected (layout shift); cambiar color/bg.
- Listas sin "dead areas": padding en vez de margin entre ítems.
- Deshabilitar transiciones temporalmente al togglear tema (evita disparar hovers).

### Track B — búsqueda ampliada (`discover.py` + lectura activa)

- **Geist colors/materials (firsthand):** confirmada la semántica numerada de escala y los radios (6/12/16px). La página de colores es SPA (valores raw vía JS) — los hex exactos se obtuvieron por Track A cruzando Sonner + gist.
- **websocket.org/heartbeat (firsthand, Matthew O'Riordan, 2026-03):** los 3 niveles de keep-alive; el browser solo puede a nivel app; **regla 75%** del menor timeout de proxy; **server-initiated + client-initiated** combinados; heartbeat inmediato en `visibilitychange`. Nuestro WS es directo al Nano por tailnet (sin reverse proxy en el camino del WS; Caddy solo cubre el control plane de Headscale), así que 20s es holgado.
- **linuru / command palette (firsthand):** convención Cmd↔Ctrl, notación macOS por defecto; **Cmd+K** estándar moderno (Linear/GitHub/Vercel). Launchers OS: Spotlight ⌘Space, Raycast ⌥Space, Windows Win+S, PowerToys Run Alt+Space.
- **Arquitectura de servido (firsthand, código):** el dashboard es estático servido en local; el Nano no lo sirve. `nano_server.py` root = texto plano. Confirmado el WS handler (binario→JSON, control conf/ping). `launch_demo.py` **ausente** del repo pese al README.

### Aplicación al refactor de embebidos-3 (decisiones)

**1. Carpeta dedicada en raíz.** Mover de `scripts/dashboard/` a una carpeta raíz, p. ej. `web/` (o `ui/`). Estructura propuesta:
```
web/
  index.html              # live + modelo (SPA hash routing #live/#modelo)
  labeling.html           # label review (consume API local cuando esté disponible)
  assets/
    fonts/                # Geist-Variable.woff2, GeistMono-Variable.woff2 (self-hosted)
    css/tokens.css        # design system Geist (dark/light)
    css/components.css     # botones, inputs, chips, tabs, cards flat, toast, modal
    js/rws.js             # ReconnectingWebSocket
    js/platform.js        # detectOS, modifierKey, Kbd, commandFor
    js/ui.js              # toast/modal/withButtonLoading (migrado, elevado a Sonner-like)
  serve.py                # launcher multiplataforma (http.server + abre browser)
```

**2. Migración de tokens (warm OKLCH → Geist neutro).** El dashboard actual usa neutros tibios hue 80 + acento mostaza. Vercel/Geist es **neutro puro + azul de acento**. Mapear:
`--bg→--background-100`, `--bg-elev-*→--gray-100..300`, `--border→--gray-300/alpha-400`, `--text→--gray-1000`, `--text-muted→--gray-800`, `--text-dim→--gray-600`, acento mostaza→`--blue-700` (o conservar un acento de marca neutro). Semánticos ok/warn/err→tokens Geist success/warning/error.

**3. Mapeo unificado clase→color (resolver la inconsistencia).** Adoptar UNA fuente de verdad por **nombre de clase** (los índices YOLO 0=plastic/1=paper/2=glass NO cambian; el color es presentación). Recomendado, alineado con convención de reciclaje y con el dashboard actual, expresado en hues Geist:
```css
--cls-glass:   var(--success-text);   /* verde  — vidrio */
--cls-paper:   var(--blue-900);       /* azul   — papel */
--cls-plastic: var(--warning-text);   /* ámbar  — plástico */
```
El `label_review` se recolorea a este mapa (hoy usa rojo/verde/azul distinto).

**4. Robustez WS.** Reemplazar el `WebSocket` directo de `app.js` por `ReconnectingWebSocket`; añadir indicador de estado (`conectando`/`activa`/`reconectando`/`sin conexión`) y banner de reconexión; heartbeat en `visibilitychange`. Mantener la correlación por `seq` encima de la capa de transporte.

**5. Detección de plataforma.** `platform.js` alimenta: (a) `Kbd` para atajos visibles, (b) bloques de comando Headscale/SSH PowerShell↔bash, (c) el launcher (`serve.py` detecta SO para abrir el browser correcto). Como es HTTP local, vía `navigator.platform`.

**6. Build vs no-build → NO-BUILD.** Mantener vanilla sin bundler: Geist self-hosted (woff2), CSS tokens + módulo de componentes, launcher Python. Razón: deploy trivial, sin paso de build, funciona offline, alineado con "MVP universitario, sustentación fluida". Vite/npm queda como opción futura, no para el demo.

**7. label_review alineado al design system.** Reescribir su HTML/CSS inline para consumir `tokens.css` + `components.css`; conservar toda la lógica de drag/resize/export (funciona). Estado de "no disponible" elegante (la herramienta es local y puede no estar corriendo).

### Brechas / baja confianza

- **Valores hex exactos de Geist:** la **semántica** de escala viene de docs oficiales (alta confianza); los **hex** se cruzaron de Sonner (CSS de Vercel/Emil) + gist de ex-Vercel + comunidad (confianza media-alta). Los tokens de sombra NO están documentados como CSS oficial.
- **`userAgentData` en HTTP local:** confirmado que no estará disponible → `navigator.platform` es el camino (no es brecha, es restricción asumida).
- **No se cubrió** (fuera de scope de esta ronda): command palette Cmd+K propio (existe patrón claro si se quiere en una ronda futura), theming light/dark switch real (el dashboard es dark-only hoy).

---

## Ronda 2 — 2026-05-24 (alto)

Redirección del usuario con **restricciones duras**: evitar a toda costa los AI/Claude frontend tropes de 2026, **SVGs nunca emojis**, **sin cards**, **sin bordes laterales de acento**, **colores planos y fríos**, **framework obligatorio**, y **pantalla hub de inicio** que enrute a 3 destinos (Dashboard · Labelling · Centro del Engine del Modelo). Esto **supersede** varias decisiones de Ronda 1 (no-build → framework; acento tibio/azul → frío; flat-section con border-left → sin border-left).

### Track A — agentes de research

#### A.1 · AI tropes 2026 a EVITAR (catálogo condensado)

Insight central: el **borde lateral de acento** (`border-left`/`border-top` de color sobre card redondeada — el "Side-Tab Card") es **el tell #1 de UI generada por IA**, "tan confiable como el em-dash en texto IA" (Adrian Krebs; catálogo Impeccable `impeccable.style/slop`). El dashboard actual lo abusa → erradicar.

| Tell IA | Reemplazo elegante |
|---|---|
| **border-left/top de color** en cards/alerts/toasts | tint de fondo a baja opacidad del color semántico + icono leading; o borde completo 1px a baja opacidad; o solo spacing. Nunca `border-left` saturado + `radius>4px` |
| **gradiente morado/índigo/violeta** (`#6366f1/#8b5cf6/#7c3aed`) | paleta propia en OKLCH/hex; acento no-morado. (Adam Wathan se disculpó ago-2025 por el default `indigo-500` de Tailwind que saturó el corpus IA) |
| **glassmorphism** (blur+transparencia+borde claro) | profundidad por borde 1px semitransparente; blur solo con capas reales |
| **cards idénticas en grid / cards anidadas / bento de relleno** | grid asimétrico, listas densas, jerarquía por tipo+espacio; máximo 2 niveles de contenedor |
| **emojis como iconos** | SVG (Lucide/Phosphor) 16-20px, stroke consistente, `aria-hidden` |
| **gradient text** | color sólido alto contraste; impacto por tamaño/peso |
| **glow/orbes/blobs de fondo** | gradiente CSS estructurado o textura sutil; nada decorativo |
| **hero centrado + CTA gigante / todo centrado** | composición asimétrica, texto a la izquierda |
| **Inter en todo** | familia con punto de vista (Geist, IBM Plex, Söhne); 4-6 tamaños, pesos extremos |
| **botón primario morado** | acento de marca propio (aquí: frío) |
| **shadcn sin personalizar** | tematizar tokens desde el primer commit (`--primary`, `--radius`, font) |
| **all-caps en labels/headings** | label pequeño semibold en case normal; all-caps solo ≤4 palabras con tracking amplio |
| **mono indiscriminado para "lo técnico"** | mono solo para código/IDs/hashes; `font-variant-numeric: tabular-nums` en la sans para cifras |
| **bounce/elastic easing** | `ease-out` (entrada), `ease-in` (salida); la animación se siente como física, no se nota |

**Qué hacen los frontends elegantes que NO parecen IA** (Linear/Stripe/Vercel, vía Mantlr + Devouring Details): paleta construida (no heredada de Tailwind por nombre), tipografía como firma, **6 estados por elemento** (default/hover/focus/active/disabled/loading), spacing rítmico (no uniforme), asimetría compositiva, **restraint de color** (casi monocromático, un acento por pantalla), estados vacíos/error diseñados. Herramienta de verificación: `npx impeccable detect`.

#### A.2 · Patrones a ADOPTAR + paletas frías (hex)

Referentes: **Linear** (dark nativo `#08090a`/`#0f1011`/`#191a1b`, cuadrícula borde-a-borde sin gaps separada por líneas 1px, profundidad por bordes `rgba(255,255,255,.05–.08)`, color cromático mínimo), **Vercel/Geist** (bordes casi invisibles, flat total, radius 4-6px UI / pill solo para pills, base 8px), **Resend** (dark-first, pares bg/fg semánticos), **Swiss/editorial** (grilla rígida, jerarquía por tamaño+peso no color, whitespace activo).

**Paleta fría canónica (Radix dark, hex):**

*Slate (neutral frío azulado)* — `1 #111113` app · `2 #18191b` sidebar/panel · `3 #212225` componente · `4 #272a2d` hover · `5 #2e3135` active · `6 #363a3f` border sutil · `7 #43484e` border · `8 #5a6169` border fuerte · `11 #b0b4ba` texto 2° · `12 #edeef0` texto 1°.

*Acentos fríos (step 9 solid / 11 texto / 12 texto fuerte):* **Teal** `9 #12a594 · 11 #0bd8b6 · 12 #adf0dd` (el más elegante para sistemas técnicos) · **Cyan** `9 #00a2c7 · 11 #4ccce6` · **Blue** `9 #0090ff · 11 #70b8ff` · **Indigo** `9 #3e63dd` (⚠ cercano al "índigo IA" — evitar como primario).

*Semánticos de estado:* success verde, error rojo, warning ámbar, info cyan/blue — usar par bg-tint (`rgba(estado,.08–.12)`) + texto step-11, **sin border-left**.

Alternativa **IBM Carbon** (cool): Cool Gray `90 #21272A / 100 #121619` fondos · Blue `60 #0F62FE` interactive · Cyan `40 #33B1FF`.

**Combo elegido:** Slate (neutral) + **Teal** (acento primario frío, NO el índigo de Linear, para esquivar el tell IA) + Cyan (secundario/info). Profundidad solo con bordes semitransparentes.

#### A.3 · Layouts SIN cards
1. **Lista densa con divider horizontal** (estilo issues de Linear): fila full-width, `border-bottom: 1px`, jerarquía por tipo, metadata a la derecha. 2. **Tabla densa sin wrapper** (radius 0, celdas tiled separadas por 1px). 3. **Secciones planas con header tipográfico** (label pequeño + contenido plano, separación por whitespace 32-48px o `border-top` fino). 4. **Split panes asimétricos** (sidebar fija 240-280px + contenido). 5. **Métrica como número grande + label inline** (32-48px bold + label 12px), separada por divider/whitespace, no card. 6. **Full-bleed** con jerarquía tipográfica.

#### A.4 · Estado SIN border-left
- **Status dot** 8px (`success #27a644 · active #10b981 · warning #f59e0b · error #ef4444 · muted slate-8`) antes del texto.
- **Pill/badge inline** con `background: rgba(estado,.08–.12)` + texto step-11, sin borde o borde 1px a `rgba(estado,.2)`.
- **Tint de fondo de fila** completo (`rgba(estado,.05)`) — no rompe alineación.
- **Icono SVG leading** (patrón Carbon: ≥3 de 4 señales color/forma/símbolo/texto).
- **Color de texto** para info pasiva (cyan-11).
- **top-border fino** para "sección activa".

#### A.5 · Hub / launcher de inicio
Patrón **command-center**: lista vertical centrada de 3 ítems (Dashboard · Labelling · Engine/Modelo), cada fila con hover (`slate-3`) + chevron, sin iconos decorativos ni cards. Display 32-48px arriba con el nombre + **status del sistema inline** ("Engine: activo · última inferencia hace 3 min"). Fondo `slate-1`. Transición fade sutil. Alternativas: 3 columnas full-height separadas por línea vertical; split 70/30 (preview + nav).

#### A.6 · Selección de framework → **SolidJS + Vite + Bun**

Matriz (criterios: reactividad alta frecuencia · bundle · canvas imperativo · ecosistema Geist · Bun+Vite · build estático · curva · fit MVP):

| | React 19+Vite | **SolidJS+Vite** | Svelte 5/Kit | Vue 3 | Astro | Preact |
|---|---|---|---|---|---|---|
| Reactividad 14fps | Medio (VDOM diffing) | **Alto (sin VDOM)** | Alto (runes) | Medio-alto | Bajo (antipatrón) | Medio |
| Bundle | ~45KB | **~7KB** | ~15KB | ~22KB | variable | ~4KB |
| Canvas imperativo | useRef+useEffect | **onMount+createEffect+untrack** | bind:this+$effect | ref+watchEffect | n/a | useRef |
| Ecosistema Geist | Alto (React) | Bajo (Kobalte/solid-ui) | Bajo (Melt) | Medio | — | bridge |
| Bun+Vite / build estático | Excelente | **Excelente** | Excelente (caveat WS Kit #18191) | Excelente | ok | Excelente |

**Decisión: SolidJS 1.9 + vite-plugin-solid + @solidjs/router (HashRouter) + Tailwind 4.** Argumento decisivo: sin VDOM, `setFrameBitmap(blob)` re-ejecuta **solo** el `createEffect` del canvas — a 14 fps React reconciliaría el árbol 14×/s. `untrack()` lee los bboxes sin suscribir el efecto. La clase `ReconnectingWebSocket` (Ronda 1) entra **sin cambios** como `lib/ws.ts`. Geist es React-only pero son tokens CSS → replicables con Tailwind; `@kobalte/core` para Select accesible. Svelte 5 es equivalente en perf pero se descarta por el caveat bun+SvelteKit-WS (issue #18191) y por DX JSX/TS consistente en Solid.

**Patrones de integración (código):**
```ts
// stores/frameStore.ts
export const [frameBitmap, setFrameBitmap] = createSignal<ImageBitmap|null>(null);
export const [bboxList, setBboxList] = createSignal<BBox[]>([]);
// hooks/useFrameStream.ts — la clase vanilla vive fuera del árbol reactivo
onMount(() => { const ws = new ReconnectingWebSocket(url); ws.binaryType='arraybuffer';
  ws.addEventListener('message', async ev => {
    if (ev.data instanceof ArrayBuffer) setFrameBitmap(await createImageBitmap(new Blob([ev.data],{type:'image/jpeg'})));
    else { const m=JSON.parse(ev.data); if(m.type==='bboxes') setBboxList(m.payload); } });
  onCleanup(() => ws.close()); });
// components/FrameCanvas.tsx — efecto desacoplado
createEffect(() => { const bmp=frameBitmap(); if(!bmp||!canvasRef) return;
  const boxes=untrack(bboxList); const ctx=canvasRef.getContext('2d')!;
  ctx.clearRect(0,0,canvasRef.width,canvasRef.height); ctx.drawImage(bmp,0,0); drawOverlay(ctx,boxes); });
```
Para labeling: `<img>` de fondo + divs absolutos como bboxes interactivos (drag/resize con `createStore`) + canvas superior solo para el cursor de dibujo (evita reimplementar hit-testing). Repos de referencia: `VKWHM/Altair-GUI` (SolidJS+WS dashboard), `agnaistic/agnai` (SolidJS WS+SSE en prod, 735★), `younesZdDz/react-bbox-annotator` (drag/resize portable).

**Setup Bun:** `bunx degit solidjs/templates/vanilla/with-solid-router web` → `bun install` → `bun run dev` / `bun run build` (→ `dist/`). HashRouter (`/#/dashboard`) sirve estático desde cualquier file server (incluido `python -m http.server` o el launcher propio).

### Track B — búsqueda ampliada (`discover.py` + lectura activa)
- **Anti-slop (firsthand):** developersdigest "15 patterns" confirma el tell #3 = *"colored borders on cards, usually on the top or left edge"*. Múltiples anti-slop skills (Impeccable, Nutlope/hallmark, refero, anti-slop-ui). Paleta IA a evitar = morado/índigo.
- **Paletas (firsthand):** IBM Carbon hex extraídos (Cool Gray, Blue 60, Cyan). Radix slate/blue/cyan/teal del agente.
- **Framework (firsthand):** dayzero (telemetría alta frecuencia) → Svelte 5 Runes −65% TTI vs React 19 VDOM; sitepoint mide updates cada 100 ms; pkgpulse "signals won 2026". Confirma que un sistema de señales (Solid/Svelte) supera al VDOM en nuestro caso. Bun+SvelteKit WS roto (#18191) → favorece SolidJS+Vite. Vite 8 + Rolldown (mar-2026) cierra brecha de build.
- **Layouts/CSS:** dense tables, tablas sin dividers, `editorial-ui`, y **CSS gap decorations** (separadores sin bordes ni pseudo-elementos, Chrome 149+).

### Decisiones de refactor (Ronda 2 — SUPERSEDEN a Ronda 1)
1. **Framework:** SolidJS + Vite + Bun, HashRouter. Carpeta raíz **`web/`** (proyecto SolidJS; reemplaza `scripts/dashboard/`).
2. **Paleta:** Slate dark (neutral frío) + **Teal** acento primario + Cyan secundario. **Cero morado/índigo.** Profundidad por **borde 1px semitransparente**, sin sombras grandes, sin glassmorphism.
3. **Sin cards:** listas densas con divider, secciones planas, split panes, métrica como número grande inline.
4. **Estado sin border-left:** status dots + pills con tint + color de texto + icono SVG.
5. **Iconos:** SVG (Lucide), nunca emoji (el labeling actual usa `◀ ▶` unicode → SVG).
6. **Hub command-center:** lista vertical de 3 destinos + status del sistema inline.
7. **Reuso:** `ReconnectingWebSocket` (R1) → `lib/ws.ts`; `detectOS()`/platform (R1) → comandos PowerShell↔bash; toast/modal elevados a patrones Sonner.
8. **Clase→color bbox:** 3 hues distinguibles sobre cámara (definir en impl; prioridad legibilidad, sesgo frío salvo que comprometa distinción).

### Brechas / baja confianza (Ronda 2)
- Hex de Radix/Carbon: de repos oficiales (alta confianza). Tailwind v4 usa OKLCH; los hex son aproximación visual.
- Ecosistema SolidJS < React para componentes Geist → mitigado con Tailwind + Kobalte (trabajo real solo en `Select` accesible).
- Mapa clase→color frío vs distinción de 3 clases sobre cámara: tensión a resolver en implementación (posible 1 hue cálido funcional para una clase).
- Pendiente (futuras rondas si se desea): command palette ⌘K propio, theming light/dark switch, animaciones de transición entre páginas.

---

## Ronda 3 — 2026-05-24 (alto)

Foco pedido: SoTA de frontends livianos para plataformas de nuestro estilo, **layouts ante todo** (cómo disponer todos los elementos), **animaciones justas y necesarias** sin afectar rendimiento, **paletas que no asemejen vibe-coding** (ampliar más allá de Slate+Teal), y **orquestación del frontend con un único script de PowerShell** (arranque + apagado). 6 agentes en paralelo (3 research-web + 3 research-video) + 3 `discover.py`.

### Track A — research

#### A.1 · LAYOUTS por superficie (prioridad)

Dos patrones canónicos 2026 para herramientas técnicas: **sidebar fija 256px** (item 36px, activo = `bg` acento al 8% SIN border-left, colapsa a 64px) y **split-pane stage/panel** cuando hay un elemento visual dominante (stage 65-75%, panel **fijo** 320-380px a la **derecha**). Jerarquía de filas de Grafana por urgencia (status → señales → recursos → detalle; fila 1 legible a 3 m). Jerarquía SIN cards (Linear/Stripe): dividers horizontales solo entre grupos, headers `text-xs uppercase tracking-widest`, número grande inline (`text-2xl`) + label `text-xs muted`, whitespace rítmico (8px intra-grupo / 24px entre grupos / 48px entre secciones), tablas densas con borde solo horizontal. Las métricas **nunca** se superponen al video: van en el panel o en un strip de 40-48px; overlay sobre canvas solo el conteo del frame (`bg-black/60 text-xs`).

**Especificación concreta por superficie:**

- **HUB** — columna única centrada, `max-w-lg` (512px), sin sidebar. Cada destino = fila `h-16`, `py-4 px-5`, separada por `border-b` (no cards), icono SVG 20px a la izquierda + título `text-base font-semibold` + descripción `text-sm muted` + chevron derecha; hover `bg-surface/50`. Header con nombre del proyecto + badge de estado del servidor; footer con URL del Nano + ping. Columna vertical (no grilla) porque con 3 destinos crea orden de importancia y se escanea de un vistazo.
- **DASHBOARD live** — `h-screen flex-col`: topbar `h-12` (URL + estado WS + badge fps + badge latencia) / cuerpo `flex-row`: **stage `flex-1` (canvas + overlay absoluto)** + **panel derecho `w-[360px] shrink-0 overflow-y-auto`**. Panel en orden de jerarquía: MÉTRICAS (números `text-2xl font-mono` + label `text-xs uppercase muted`, `grid-cols-2 gap-4`) → divider → CONTEO POR CLASE → divider → CONTROLES (URL, cámara, sliders confianza/fps). 360px (no 280px) porque los sliders + 3 clases lo necesitan. Estado de conexión/fps como chip `absolute top-3 left-3` sobre el canvas. Grid robusto: `grid-cols-[1fr_360px]`.
- **MODEL ENGINE** — 3 filas verticales: estado (`py-5`, pill semafórico + nombre modelo + timestamp) → divider → controles build (botones + barra de progreso lineal si hay build) → divider → **terminal de logs SSE `flex-1 min-h-0`**, `font-mono text-xs`, fondo `#09090b`, stdout gris/verde y stderr rojo, auto-scroll al fondo. Sin sidebar: el log domina y necesita el ancho completo. Si se quiere ver video+logs a la vez en la demo, split horizontal 40/60.
- **LABELLING** — `h-screen flex-col`: topbar `h-11` (imagen actual + progreso X/N + export) / cuerpo `flex-row`: **toolbar vertical `w-12`** (selector, bbox, zoom, delete) + **canvas `flex-1`** (imagen `object-contain` + cajas editables) + **sidebar derecha `w-[300px]`** (selector de clase = 3 botones grandes con color por clase, lista de cajas del frame con color-dot + coords + delete, acciones) / barra inferior `h-[52px]` (← Anterior · 3/47 · Siguiente →). Sin thumbnails (innecesario para el MVP; prev/next con teclado basta). Convergente con CVAT/Roboflow/Label Studio.

Repo de referencia más cercano: `niklasfrick/spark-dashboard` (GPU + métricas WS, Tailwind 4 + Vite) — usa `bg-[#111115] border border-white/[0.04]` sin sombra ni border-left, hover = acento al 10%.

#### A.2 · Animaciones + frontend liviano

Regla física única: **animar solo `transform` y `opacity`** (corren en el compositor, fuera del main thread); jamás `width/height/top/left/margin/padding` (fuerzan layout reflow y compiten con el canvas). Con el canvas a 14 fps (~71 ms/frame) el main thread tiene ventanas libres; CSS/WAAPI sobre transform/opacity **no compiten** porque el compositor thread los maneja aunque el main thread esté pintando.

**Estrategia en dos capas (decisión):**
1. **CSS transitions + keyframes (80% de casos, 0 KB):** hover (`transform: scale(.97)` en `:active`, 120 ms), estado de conexión (`transition: color/box-shadow` 200 ms), toast (keyframe `translateY+scale+opacity` 200-250 ms), skeleton (keyframe pulse linear infinite). Curvas custom: `--ease-snappy: cubic-bezier(.16,1,.3,1)` (entradas), `--ease-out-expo: cubic-bezier(.19,1,.22,1)` (transiciones).
2. **`solid-transition-group` + WAAPI (<2 KB)** solo para transición de ruta Hub↔página y exits. **View Transitions API** como progressive enhancement en la navegación del HashRouter (`document.startViewTransition` con feature-detect; CSS `::view-transition-old/new(root)` 200 ms).

**NO** usar Motion/solid-motionone/Framer/GSAP (30-50 KB, dependencia poco mantenida, overhead injustificado). `prefers-reduced-motion: reduce` global obligatorio. Bundle liviano: `lazy()` por ruta (el Hub eager, Dashboard/Engine/Labeling diferidos — Vite los parte en chunks), sin chart libs (se pinta en canvas), sin date-fns (toLocaleTimeString), SVG inline.

Tabla de costo (gzip): CSS/WAAPI 0 KB · solid-transition-group <2 KB · Motion `animate()` mini 2,6 KB · Motion full 18 KB · Framer 30-50 KB.

Reglas de Emil Kowalski (animations.dev): nunca animar acciones de teclado ni valores que cambian en tiempo real (los números de fps/latencia **no** se animan); easing por intención (entra/sale → ease-out; mueve → ease-in-out; hover → ease); duraciones 100-150 ms micro, 150-250 ms estándar, 200-300 ms modales; exits 20% más rápidos.

#### A.3 · Paletas que NO parecen vibe-coding

El tell #1 sigue siendo **indigo/violeta** (`#6366f1/#8b5cf6`, el default literal de Tailwind en el corpus IA). Otros tells acumulables: múltiples acentos compitiendo, grises **puros** sin matiz (`#808080`), negro/blanco puros sin intermedios, saturación >80%, gradientes radiales decorativos, Inter + cards `rounded-xl` en grilla simétrica. Lo que se ve diseñado: **un solo acento** <70-80% sat., grises con **matiz frío sutil** (no puro), jerarquía por contraste/peso (no por color extra), color cromático solo para datos semánticos. Filosofía Geist: "achromatic by design".

**3 candidatas con hex completos (en el reporte del agente):**
- **A · Carbon/IBM** — Cool Gray 90/100 (`#13171a`→`#262b31`) + **IBM Blue 60 `#0f62fe`**. La más institucional/misión-crítica.
- **B · Cordum** — Slate hue-260 (`#111827`→`#253248`) + teal luminoso `#00e5a0`. Riesgo: el teal puede leerse "eco".
- **C · Zinc/Steel + Cyan IBM** — Radix Slate dark (`#111113` app · `#18191c` panel · `#212225` surface · `#3b3d40` borde · `#b0b4ba`/`#ededef` texto) + **IBM Cyan 50 `#1192e8`** (hover `#0072c3`, bg-badge `#012749`, texto `#82cfff`). La más fría y técnica (terminal de monitoreo / sistema embebido).

**Recomendación (decisión): Paleta C — Radix Slate dark + IBM Cyan 50**, sobre el Teal de Ronda 2. Por qué no parece IA: gris base con matiz azulado (no zinc/gray puro de shadcn); acento cyan IBM (hue ~210° con verde visible, reconocible de instrumentación industrial, ni `blue-500` ni indigo); **un solo acento** (success/warning/error vienen de escala IBM aparte, no variantes del acento); profundidad por luminancia de capas, sin gradientes; tipografía **IBM Plex Sans + IBM Plex Mono** (mono para métricas/confianza/IDs) que es el par más alejado del Inter genérico y refuerza el carácter de instrumento. Alternativa conservadora si se prefiere: Paleta A (Carbon + Blue 60).

**Mapeo clase→color (cajas de detección, paleta de Bang Wong 2011, color-blind safe):** vidrio = **Sky `#56B4E9`**, papel = **Ámbar `#E69F00`** (único cálido, necesario para distinguir en daltonismo), plástico = **Bluish-green `#009E73`**. Variante reforzada para legibilidad sobre cámara variada (con outline 2px oscuro del mismo hue): vidrio `#00B4D8`, papel `#F59E0B`, plástico `#10B981`. Clave: estos colores **solo viven en el canvas**; el chrome de la UI es 100% frío (Slate + cyan). Esa separación de capas es sello de diseño intencional, y la procedencia (Wong, Nature Methods 2011 + IBM Carbon) da narrativa técnica para la sustentación.

### Track B — video (3 agentes) + discover + PowerShell

**Craft de animación (video):** David Khourshid — "CSS para estados finitos conocidos (transitions + keyframes, suben al compositor); JS solo para valores dinámicos en runtime (cursor, scroll, confianza) o animaciones interrumpibles" (`BuNksQ5PkxM` 01:33/01:59). "Mucha gente usa GSAP hasta para un hover de botón, y no hace falta" (t=190). FLIP (Alex Holachek) para transiciones de layout costosas sin reflow (`s06Z_e8ac0Y` t=33). Josh Comeau: accordions mal hechos animan `height`; correcto es `transform: scaleY()` (`UGRNoYuEDLk`). **Canvas = zona protegida**: ninguna animación JS encima; overlays en `<canvas absolute pointer-events:none>`, fade solo con `opacity` CSS.

**Diseño de dashboards densos (video):** Steve Schoger — la brecha genérico↔pulido es "un montón de detalles pequeños", no arquitectura (`EjEYTRD-W-M` t=210); jerarquía sin cards = gestionar el espacio entre elementos del mismo nivel, "darles aire" (t=2580); anatomía densa = topbar + sidebar izquierda + canvas central (t=2660). Eleken — "el instinto errado es mostrar todo a la vez; abruma" (`rP-I4Oihqc8` t=33); disclosure progresivo: insight de alto nivel primero, detalle en sidebars expandibles (t=144); "reconstruir la arquitectura de información priorizando claridad sobre densidad" (t=79). Densidad proposicional (Don Norman): un mark que codifica varias dimensiones (barra de fps = valor + % de capacidad; badge de confianza con color = valor + estado). Escala tipográfica = 3-4 tamaños (display/label/body/micro).

**SolidJS liviano y fluido (video):** Ryan Carniato — Solid rastrea **lecturas, no escrituras**: al cambiar un signal solo se actualiza el nodo DOM que lo leyó, sin re-render de árbol (`bJyREHmgo5E` t=486); los componentes corren **una sola vez** (setup), el JSX compila a ops DOM directas (t=823). Práctica clave: **estratificar signals por frecuencia** — `frameBlob` (alta, ~14/s, único subscriber del canvas) separado de `detections`/conteos (baja). **Canvas imperativo = `ref` + `createEffect`** que depende solo del signal del frame; nunca pasar el frame como prop. **WS desacoplado del redibujo** con throttle por `requestAnimationFrame` (guardar último frame, dibujar máx. 1/raf). Si el decode JPEG >5-8 ms: **Web Worker + `createImageBitmap` + `OffscreenCanvas`**, devolver `ImageBitmap` como transferable (cero copia; Vite soporta `new Worker(new URL(...), {type:'module'})`). Usar `createSignal` (no `createStore`) para el frame: los proxies de store tienen overhead por acceso; el frame se reemplaza atómicamente.

**`discover.py` (43+44+48 resultados):** arXiv *Dashboard Design Patterns* (2205.00757, 42 patrones de 144 dashboards); Carbon *Dashboards* (priorizar por importancia, mayor contraste+área al dato clave); **ISA-101** jerarquía de 4 niveles para HMI industrial; motion.dev *reduce bundle size* y *CSS reemplaza librerías JS de animación* (wunderlandmedia); MotherDuck *Vibe-Coding a Dashboard* (color con intención = paleta de severidad, no arcoíris) y Fountain Institute *7 signs vibe-coded UI* (neón, sin jerarquía). Repos reales: spark-dashboard, ev-fleet-telemetry, uav-telemetry, cordum (design language "quiet confidence", teal con significado real).

**Orquestación PowerShell (Track B fase 2 — diseño):** un **único `web.ps1` (pwsh 7, cross-platform)** con `-Action start|stop|restart|status` (default `start`), aprovechando que pwsh corre igual en Windows y macOS y ramifica los pocos comandos OS-específicos vía `$IsWindows`/`$IsMacOS` (esto reconcilia el requisito original "detectar dónde se ejecuta" con el nuevo "solo PowerShell"):
- **start:** verifica `bun` (`Get-Command`), `bun install` si falta `node_modules`, `bun run build` si `dist/` está ausente/viejo, levanta `bun run preview` (Vite preview del build fresco — estable para demo, sin sorpresas de HMR) con `Start-Process -PassThru -RedirectStandardOutput .run/web.log`, guarda `$proc.Id` en `.run/web.pid`, chequea salud del Nano (`Invoke-WebRequest http://100.64.0.2:8000/health -TimeoutSec 3`, warn si no responde), abre el navegador (`Start-Process http://localhost:4173` en Win / `open` en mac).
- **stop:** lee `.run/web.pid` y mata el **árbol** (bun/vite generan hijos esbuild): Windows `taskkill /PID $id /T /F`; macOS/Linux `pkill -P $id; Stop-Process -Id $id`. Borra el pid file.
- **status:** PID vivo + salud del Nano. `restart` = stop+start.
`vite preview` no es servidor de producción, pero para un MVP de demo es la ruta más fluida y robusta. `-Dev` opcional → `bun run dev` con HMR para preparar.

### Decisiones de refactor (Ronda 3 — refinan Ronda 2)
1. **Layouts:** bloqueados por superficie con dimensiones (arriba). Dashboard = split 70/30 con panel **derecho fijo 360px**; Hub = columna centrada `max-w-lg`; Engine = 3 filas con terminal `flex-1`; Labelling = toolbar 48px + canvas + sidebar 300px + nav inferior. **Cero cards, cero border-left**; separación por espacio y dividers.
2. **Animación:** dos capas — CSS/WAAPI (transform/opacity, compositor) para el 80%, `solid-transition-group` + View Transitions (feature-detect) solo para rutas. **Sin librería de animación.** `prefers-reduced-motion` global. Los números en vivo no se animan; el canvas es zona protegida.
3. **Paleta:** **Radix Slate dark + IBM Cyan 50 `#1192e8`** (supersede el Teal de Ronda 2; alternativa conservadora: Carbon + Blue 60). Tipografía **IBM Plex Sans + IBM Plex Mono**. Un solo acento; estados de escala IBM aparte. Clase→color Wong (sky/ámbar/bluish-green) solo en el canvas.
4. **Performance SolidJS:** signals estratificadas por frecuencia; WS→frame con throttle rAF; `createSignal` (no store) para el frame; `lazy()` por ruta; Web Worker + OffscreenCanvas como optimización opcional si el decode tranca.
5. **Orquestación:** un único `web.ps1` (pwsh, start/stop/restart/status), `Start-Process -PassThru` + PID file, `taskkill /T /F` (Win) ↔ `pkill`/`Stop-Process` (mac vía `$IsMacOS`), `vite preview` del build, health-check del Nano, abre navegador.

### Brechas / baja confianza (Ronda 3)
- Hex exactos de Radix Slate dark vienen de análisis comunitario (la página es JS-rendered); verificar con el generador custom de Radix en implementación. IBM Cyan/Plex y Wong son de fuente oficial/paper (alta confianza).
- **View Transitions API** en Safari: usar siempre con `if (document.startViewTransition)`; degrada a navegación directa.
- Web Worker + OffscreenCanvas: opcional, no requerido para el MVP; medir el decode antes de invertir.
- Limitación de video: varios transcripts llegaron en formato `pb3` (auto-captions) parcialmente parseables; las citas con timestamp son verificables. Emil Kowalski y Rauno **no** están en YouTube (su obra vive en blogs, ya capturada en Ronda 1).
- `vite preview` no es servidor de producción endurecido; aceptable para demo. Si se quisiera robustez extra, un `Bun.serve` estático de ~10 líneas es la alternativa.

---

## Historial de investigación

| Ronda | Fecha | Profundidad | Foco |
|-------|-------|-------------|------|
| 1 | 2026-05-24 | alto | Geist/tokens · detección SO · reconexión WS · craft Sonner/Rauno/Emil · arquitectura de consolidación |
| 2 | 2026-05-24 | alto | anti-AI-tropes · patrones elegantes + paletas frías · layouts sin cards · estado sin border-left · hub screen · selección de framework (SolidJS) |
| 3 | 2026-05-24 | alto | layouts por superficie (dim. concretas) · animaciones compositor-only · paletas no-vibe-coding (Slate+Cyan IBM) · perf SolidJS (signals/rAF/worker) · orquestación PowerShell único |

## Fuentes consultadas (acumulado)

| # | Título | URL | Tipo | Ronda |
|---|--------|-----|------|-------|
| 1 | Geist Design System — Colors | https://vercel.com/geist/colors | Doc oficial | 1 |
| 2 | Geist Design System — Typography | https://vercel.com/geist/typography | Doc oficial | 1 |
| 3 | Geist Design System — Materials | https://vercel.com/geist/materials | Doc oficial | 1 |
| 4 | Geist — Keyboard Input | https://vercel.com/geist/keyboard-input | Doc oficial | 1 |
| 5 | vercel/geist-font | https://github.com/vercel/geist-font | Repo (fuente) | 1 |
| 6 | npm `geist` | https://www.npmjs.com/package/geist | Paquete | 1 |
| 7 | Vercel DESIGN.md (shadcn) | https://www.shadcn.io/design/vercel | Doc comunitario | 1 |
| 8 | vercel-ui tokens completos | https://vercel-ui-phi.vercel.app/docs/colors | Implementación comunitaria | 1 |
| 9 | gist tokens Geist (Anthony Shew) | https://gist.github.com/anthonyshew/9bfd709949f83b4acac9062787c071a7 | Gist (ex-Vercel) | 1 |
| 10 | Web Interface Guidelines (Rauno Freiberg) | https://interfaces.rauno.me/ | Living doc | 1 |
| 11 | Great Animations (Emil Kowalski) | https://emilkowal.ski/ui/great-animations | Blog | 1 |
| 12 | 7 Practical Animation Tips (Emil Kowalski) | https://emilkowal.ski/ui/7-practical-animation-tips | Blog | 1 |
| 13 | emilkowalski/sonner — styles.css + index.tsx | https://github.com/emilkowalski/sonner | Repo (fuente) | 1 |
| 14 | pladaria/reconnecting-websocket | https://github.com/pladaria/reconnecting-websocket | Repo (fuente) | 1 |
| 15 | zimv/WebSocketHeartBeat | https://github.com/zimv/WebSocketHeartBeat | Repo (fuente) | 1 |
| 16 | WebSocket Heartbeat (websocket.org) | https://websocket.org/guides/heartbeat/ | Guía técnica | 1 |
| 17 | User-Agent Client Hints API | https://developer.mozilla.org/en-US/docs/Web/API/User-Agent_Client_Hints_API | MDN | 1 |
| 18 | NavigatorUAData.getHighEntropyValues | https://developer.mozilla.org/en-US/docs/Web/API/NavigatorUAData/getHighEntropyValues | MDN | 1 |
| 19 | darkreader — utils/platform.ts | https://github.com/darkreader/darkreader/blob/main/src/utils/platform.ts | Código | 1 |
| 20 | TanStack Hotkeys — detectPlatform | https://github.com/TanStack/hotkeys | Código | 1 |
| 21 | PR Twenty CRM — ctrl vs ⌘ | https://github.com/twentyhq/twenty/pull/9617 | Pull request | 1 |
| 22 | Command Palette en cada app (Linuru) | https://linuru.com/documents/command-palette/ | Artículo | 1 |
| 23 | Linear Keyboard Shortcuts (Win/Mac) | https://keycombiner.com/collections/linear/ | Referencia | 1 |
| 24 | Impeccable — slop catalog (tells IA) | https://impeccable.style/slop | Guía | 2 |
| 25 | Radix Colors | https://www.radix-ui.com/colors | Doc oficial | 2 |
| 26 | IBM Carbon Design System | https://carbondesignsystem.com | Doc oficial | 2 |
| 27 | SolidJS — documentación | https://docs.solidjs.com | Doc oficial | 2 |
| 28 | VKWHM/Altair-GUI (SolidJS + WS dashboard) | https://github.com/VKWHM/Altair-GUI | Repo | 2 |
| 29 | younesZdDz/react-bbox-annotator | https://github.com/younesZdDz/react-bbox-annotator | Repo | 2 |
| 30 | Dashboard Design Patterns (arXiv 2205.00757) | https://arxiv.org/abs/2205.00757 | Paper | 3 |
| 31 | niklasfrick/spark-dashboard (GPU+WS, Tailwind4+Vite) | https://github.com/niklasfrick/spark-dashboard | Repo | 3 |
| 32 | Datature — annotator layout | https://developers.datature.io/docs/annotator | Doc | 3 |
| 33 | A closer look at CVAT (LearnOpenCV) | https://learnopencv.com/a-closer-look-at-cvat-perfecting-your-annotations/ | Tutorial | 3 |
| 34 | Linear design system (shadcn.io) | https://www.shadcn.io/design/linear | Doc comunitario | 3 |
| 35 | Animations on the Web (Benedikt Sperl) | https://www.benedikt-sperl.de/blog/2026-01-13-animations-on-the-web | Blog | 3 |
| 36 | emilkowalski/skill — animations/easing/perf | https://github.com/emilkowalski/skill | Repo | 3 |
| 37 | solidjs-community/solid-transition-group | https://github.com/solidjs-community/solid-transition-group | Repo | 3 |
| 38 | Motion — reduce bundle size | https://motion.dev/docs/react-reduce-bundle-size | Doc oficial | 3 |
| 39 | Browser Rendering Performance: Pixel Pipeline | https://allahabadi.dev/blogs/frontend/browser-rendering-performance-pixel-pipeline/ | Blog | 3 |
| 40 | WebKit — scroll-driven animations con CSS | https://webkit.org/blog/17101/a-guide-to-scroll-driven-animations-with-just-css/ | Doc oficial | 3 |
| 41 | IBM Design Language — Color | https://www.ibm.com/design/language/color/ | Doc oficial | 3 |
| 42 | Carbon v10 — data viz color palettes | https://v10.carbondesignsystem.com/data-visualization/color-palettes/ | Doc oficial | 3 |
| 43 | Radix Colors — understanding the scale | https://www.radix-ui.com/colors/docs/palette-composition/understanding-the-scale | Doc oficial | 3 |
| 44 | cordum — dashboard design language | https://github.com/cordum-io/cordum/blob/main/cordum-dashboard-design-language.md | Referencia | 3 |
| 45 | 7 Signs a UI Has Been Vibe Coded | https://www.thefountaininstitute.com/blog/signs-vibe-coded-ui | Blog | 3 |
| 46 | Wong — Color blindness (Nature Methods, 2011) | https://www.nature.com/articles/nmeth.1618 | Paper | 3 |
| 47 | Animations: use CSS or JS? (David Khourshid / Prismic) | https://youtu.be/BuNksQ5PkxM | Video | 3 |
| 48 | The Little Details of UI Design (Steve Schoger / Laracon) | https://youtu.be/EjEYTRD-W-M | Video | 3 |
| 49 | SolidJS: reads-not-writes (Ryan Carniato / GitNation) | https://youtu.be/bJyREHmgo5E | Video | 3 |
| 50 | Absolute speed: SolidJS + Web-Workers (Atila) | https://youtu.be/Ll2zt2m5Z5A | Video | 3 |
| 51 | Stop-Process (Microsoft Learn, pwsh 7.5) | https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/stop-process | Doc oficial | 3 |
| 52 | Build a frontend using Vite and Bun | https://bun.com/docs/guides/ecosystem/vite | Doc oficial | 3 |
| 53 | Building for Production (Vite) | https://vite.dev/guide/build | Doc oficial | 3 |
