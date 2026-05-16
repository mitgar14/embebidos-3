# 06 · Dashboard · pestaña "modelo"

El dashboard es un SPA vanilla JS servido como estáticos (`scripts/dashboard/`) desde un dev server local en el equipo del operador. Apunta a la API en `http://100.100.166.120:8000` (Tailscale del Nano).

## Archivos del dashboard

| Archivo | Líneas | Función |
|---|---|---|
| `index.html` | ~180 | Estructura DOM, dos panes (`live`, `modelo`), header/topbar, símbolos SVG inline |
| `style.css` | ~1150 | Estilos OKLCH, layout, componentes |
| `app.js` | ~? | Pane "live": WebSocket, render de bboxes sobre `<video>`, telemetría header |
| `modelo.js` | ~660 | Pane "modelo": fetch state, render templates, acciones (build/check/rollback/cancel) |
| `ui.js` | ~170 | Primitivas reutilizables: `showToast`, `openModal`, `withButtonLoading` (ver doc 07) |

Este documento cubre **`modelo.js`** y la interacción con la API. Los componentes UI están en doc 07. El sistema visual en doc 08.

## Modelo de datos

Todo gira alrededor del objeto de `/model/state` (ver doc 02). El módulo lo guarda en `state.lastState` para:
- Renderizar el template correspondiente
- Recordar el último estado conocido cuando el server se cae (durante un build)
- Dar contexto al mensaje del card "Servidor no responde" (último estado building → mensaje "se reinicia tras compilación")

## Templates por estado

```js
const TEMPLATES = {
  no_model: noModelTemplate,
  ready: readyTemplate,
  update_available: (s) => readyTemplate(s, { banner: true }),
  degraded: degradedTemplate,
  building: buildingTemplate,
};
```

### `noModelTemplate(s)`

CTA principal "Descargar y compilar engine" + banner secundario "Adoptar engine existente" (si `engine_binary_present=true`).

### `readyTemplate(s, opts)`

Diseño editorial (post-redesign 2026-05-16):
- Heading "Modelo activo" + badge "ADOPTADO" inline (si aplica)
- 2 bloques separados por `border-top` hairline:
  - **Identidad**: Commit (7-char), ONNX (12-char SHA), Engine (12-char SHA)
  - **Build**: Entrenado (timestamp HF), Compilado (timestamp build), Build (duration · WS · FP16)
- Actions: "Verificar actualizaciones", "Recompilar engine"

### `degradedTemplate(s)`

Banner ámbar "Usando engine anterior" + el `readyTemplate` debajo. Indica que el último build falló y se restauró el `.previous`.

### `buildingTemplate(s)`

Header con job_id en monospace + barra de progreso (no nested card) + grid mínimo (Commit, Iniciado) + button "Cancelar build" + pre con logs en vivo (SSE).

### `renderUnreachable(err)`

Card-warn con border-left ámbar cuando `/model/state` falla. Mensaje contextual según `state.lastState`:
- Si último estado fue `building` → "El servidor se reinicia automáticamente cuando termina la compilación"
- En otros casos → "El servidor está temporalmente fuera de alcance"

Incluye botón "reintentar ahora" con `withButtonLoading` + `role="status"` con `aria-live="polite"` para anuncios accesibles.

## Polling

```js
window.initModelTab = function () {
  fetchState();
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(fetchState, 3000);
};
```

Polling cada 3s. **No hay circuit breaker**: si el server está abajo, el dashboard sigue haciendo `fetch()` cada 3s. Es aceptable para MVP (el browser maneja errores rápido), pero apuntado como vacío.

## Acciones y patrón `withButtonLoading`

Cada handler de acción usa el helper `withButtonLoading(btn, label, asyncFn)` para:
1. Disabled + aria-busy + spinner inline durante la request
2. Label change descriptivo ("verificando…", "lanzando build…", "revirtiendo…", "cancelando…")
3. Restauración garantizada en `finally`
4. Idempotencia (segundo click es no-op gracias a `dataset.busy`)

Ejemplo:
```js
async function checkUpdates(btn = null) {
  await window.withButtonLoading(btn, 'verificando…', async () => {
    try {
      const r = await fetch(api('/model/check-updates'), { method: 'POST' });
      const data = await r.json();
      const ui = formatCheckUpdates(data);
      window.showToast(ui.body, ui.type, { title: ui.title, durationMs: ui.durationMs });
      fetchState();
    } catch (e) {
      window.showToast(e.message, 'error', { title: 'No se pudo consultar HF' });
    }
  });
}
```

## El patrón "verificación post-error"

`triggerBuild` aplica un patrón sustentado en research (Matheus Palma, 2026-04): *naive rollback creates a false negative*. Ante un error HTTP del POST, antes de declarar fallo, **consulta el estado actual del recurso** para ver si la operación sí se realizó.

```js
async function _confirmBuildLaunchedDespiteError() {
  try {
    const r = await fetch(api('/model/state'));
    if (!r.ok) return null;
    const data = await r.json();
    if (data.state === 'building' && data.active_job?.job_id) {
      return data.active_job.job_id;
    }
    return null;
  } catch (_) { return null; }
}
```

Cuando `r.ok` es false o hay network error en `triggerBuild`, llama esto. Si encuentra un job activo nuevo, muestra `Build lanzado (con aviso del server)` en vez de `No se pudo lanzar`. Esto **antes salvaba un bug del backend** (race con `Type=oneshot`, ver doc 01) y ahora queda como red de seguridad defensiva.

## Stream de logs SSE

Buffer + flush en `requestAnimationFrame` con cap circular de 2000 líneas. Reemplazo de la versión naive `pane.textContent += line` que era O(n²) y congelaba el browser con bursts de trtexec.

```js
const LOG_CAP = 2000;
const _logs = { pending: [], rendered: [], rafId: null, follow: true };

function _enqueueLog(line) {
  _logs.pending.push(line);
  if (_logs.rafId !== null) return;
  _logs.rafId = requestAnimationFrame(_flushLogs);
}

function _flushLogs() {
  _logs.rafId = null;
  const pane = document.getElementById('logs-stream');
  if (!pane || !_logs.pending.length) return;
  const wasAtBottom = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;
  const batch = _logs.pending;
  _logs.pending = [];
  const chunk = batch.join('\n') + '\n';
  for (const ln of batch) _logs.rendered.push(ln);
  if (_logs.rendered.length > LOG_CAP) {
    _logs.rendered.splice(0, _logs.rendered.length - LOG_CAP);
    pane.textContent = _logs.rendered.join('\n') + '\n';
  } else {
    pane.appendChild(document.createTextNode(chunk));  // O(1)
  }
  if (wasAtBottom && _logs.follow) pane.scrollTop = pane.scrollHeight;
}
```

Listener de scroll detecta cuando el usuario hace scroll manual hacia arriba → pausa auto-follow hasta que vuelva al bottom.

## Formato de timestamps

`_fmtLocalDateTime(iso)` convierte UTC ISO a hora local Bogotá (UTC-5) con formato `YYYY-MM-DD HH:MM:SS`. `_fmtRelative(iso)` da "hace 12 min" / "hace 3 h". `_timeEl(iso)` los compone con `<time datetime title>` para tooltip nativo del browser con el ISO original.

Patrón sustentado en Cloudscape (AWS) Design System.

## API contract: lo que el dashboard espera de la API

| Endpoint | Polling | Fallo aceptable |
|---|---|---|
| `GET /model/state` | 3s | sí, muestra "Servidor no responde" |
| `GET /jobs/{id}/logs` (SSE) | continuo | sí, cierra silencioso |
| `POST /model/build` | on-demand | sí, toast error |
| `POST /model/check-updates` | on-demand | sí, toast error |
| `POST /model/rollback` | on-demand | sí, toast error |
| `POST /model/adopt` | on-demand | sí, toast error |
| `DELETE /jobs/{id}` | on-demand | sí, toast error |

## Gotchas

- **El polling no usa AbortController** → si una request se cuelga >3s, llegará a tener varias fetches concurrentes. En la práctica no causa bugs porque solo modifica el DOM con el último resultado.
- **`apiBase()` deriva la URL HTTP del input `ws-url`** → consistencia entre pane live (WS) y modelo (HTTP). Si el usuario cambia el input mientras hay polling activo, la próxima request va a la URL nueva.
- **`render()` reemplaza `innerHTML` completo** → cualquier estado local del DOM se pierde (loading state de botones, scroll position). Para botones esto se manejó pasando `btn` al handler y aplicando loading state **después** del render. Para scroll de logs se preserva via el flag `_logs.follow`.
- **El polling re-renderiza** → no usar `setTimeout` largos en estados del DOM esperando que sobrevivan. Si necesitás estado persistente, llevalo a un objeto global.
