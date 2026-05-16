# 07 · UI components · toast, modal, loading

Primitivas reutilizables en `scripts/dashboard/ui.js` (170 líneas, vanilla JS, sin frameworks). Reemplazan `alert()`/`confirm()` nativos del browser (que el usuario rechazó como "porquería") y dan feedback de loading robusto.

## Por qué un módulo dedicado

`alert("localhost:8001 dice — Modelo al día. Commit b93964f · ONNX 223f1a71.")` es horrible: bloquea el thread, no respeta el dark mode del dashboard, prefija con la URL del origin. Lo reemplazamos por:

- **`showToast`**: notificación no-bloqueante esquina superior-derecha, severidad por color, auto-dismiss
- **`openModal`**: dialog confirmación accesible, retorna `Promise<boolean>`
- **`withButtonLoading`**: helper que envuelve cualquier async fn con state visual de loading sobre un botón específico

## API pública

```js
window.showToast(message, type, opts)
// type: 'info' | 'success' | 'warn' | 'error'
// opts: { title?, durationMs? } — durationMs=0 → no auto-dismiss

window.openModal(opts)
// opts: { title, body, confirmText?, cancelText?, danger? }
// retorna: Promise<boolean>

window.withButtonLoading(btn, loadingLabel, asyncFn)
// Envuelve asyncFn con disabled + aria-busy + spinner + label change.
// retorna: lo que retorne asyncFn (o undefined si btn no existe).
```

## Toast — diseño

Stack vertical top-right, `clamp(280px, 32vw, 380px)`. Cada toast:
- Border-left 3px de color por severidad — mismo lenguaje visual que `.ctr-glass/paper/plastic`
- Background tintado sutil: `color-mix(in oklch, var(--severity) 6-9%, var(--bg-elev-2))`
- Title bold + body multi-línea + botón × sutil
- Animación entrada: `translateX(8px) → 0` + opacity 220ms ease-out-quart
- Salida: 160ms ease-in

Severidades:
| Type | Color | Uso |
|---|---|---|
| `info` | accent (mostaza) | acción iniciada / contexto |
| `success` | OK (verde) | operación completada |
| `warn` | warn (mostaza cálida) | algo no-fatal a notar |
| `error` | err (rojo) | falla, requiere acción |

Coherente con la regla del usuario: NO abusar de minúsculas (titles en sentence case, mensajes normales).

## Modal — diseño

- Overlay: `color-mix(in oklch, var(--bg) 75%, transparent)` con `backdrop-filter: blur(2px)` sutil
- Dialog centrado: `clamp(320px, 44vw, 460px)`, `border-strong`, padding 20px 22px 16px
- Animación entrada: `translateY(8px) scale(0.985) → 0,1` con opacity, 220ms ease-out-quint
- Botones: `.danger` (rojo), `.primary` (mostaza), `.ghost` (cancelar)
- ESC = cancel, Enter = confirm, click overlay = cancel

## `withButtonLoading` — el patrón anti-frustración

Sustentado en investigación (Nielsen + gomakethings + thelinuxcode + Matheus Palma):

```js
async function withButtonLoading(btn, loadingLabel, fn) {
  if (!btn) return await fn();
  if (btn.dataset.busy === '1') return; // idempotencia: segundo click no-op
  btn.dataset.busy = '1';
  const originalText = btn.textContent;
  const originalAriaBusy = btn.getAttribute('aria-busy');
  btn.disabled = true;
  btn.setAttribute('aria-busy', 'true');
  btn.classList.add('is-loading');
  if (loadingLabel) btn.textContent = loadingLabel;
  try {
    return await fn();
  } finally {
    btn.disabled = false;
    if (originalAriaBusy === null) btn.removeAttribute('aria-busy');
    else btn.setAttribute('aria-busy', originalAriaBusy);
    btn.classList.remove('is-loading');
    btn.textContent = originalText;
    delete btn.dataset.busy;
  }
}
```

### 5 piezas que lo hacen robusto

1. **Flag `data-busy`** previene doble-submit ANTES de que `disabled` aplique (race con clicks rápidos)
2. **`disabled`** desactiva el botón nativo
3. **`aria-busy="true"`** anuncia a screen readers que el control está procesando
4. **Label change descriptivo** ("verificando…", no solo spinner): el usuario sabe QUÉ está pasando
5. **`finally` restaura siempre**: si la request falla, el botón vuelve a estar usable sin recargar

### CSS del spinner inline

```css
@keyframes btn-spin { to { transform: rotate(360deg); } }
button.is-loading {
  cursor: progress;
  opacity: 0.85;
  min-width: max(72px, calc(1ch * var(--label-len, 8) + 36px));
  position: relative;
  padding-left: 28px;
}
button.is-loading::before {
  content: '';
  position: absolute;
  left: 10px;
  top: 50%;
  width: 11px;
  height: 11px;
  margin-top: -6.5px;
  border: 1.5px solid currentColor;
  border-top-color: transparent;
  border-radius: 50%;
  animation: btn-spin 700ms linear infinite;
  opacity: 0.85;
}
@media (prefers-reduced-motion: reduce) {
  button.is-loading::before {
    animation: none;
    border-top-color: currentColor;
    opacity: 0.55;
  }
}
```

- `currentColor` para el spinner → toma el color del botón. Funciona en `.primary`, `.danger`, `.ghost` sin overrides.
- `min-width` previene layout shift cuando cambia el label
- `prefers-reduced-motion` rinde un spinner estático (anillo completo) en vez de animar

## Pattern aplicado

Todos los handlers del dashboard reciben `btn` y lo pasan:

| Handler | Loading label |
|---|---|
| `triggerBuild(force, btn)` | `lanzando build…` |
| `checkUpdates(btn)` | `verificando…` |
| `rollback(btn)` | `revirtiendo…` |
| `adoptEngine(btn)` | `adoptando…` |
| `cancelBuild(jobId, btn)` | `cancelando…` |
| `btn-retry-state` | `reintentando…` |

## Anti-patterns que evita

- **Spinner global flotante** que oculta el botón clicado → el usuario pierde contexto. Aquí el spinner está **en el botón mismo**.
- **`disabled` puro sin `aria-busy`** → screen readers no saben qué pasa
- **Label que cambia sin disabled** → doble-click dispara dos requests
- **Falta `finally`** → si la API falla, el botón queda perma-disabled

## Toast: helper de severity-aware messages

`modelo.js::formatCheckUpdates()` devuelve `{title, body, type, durationMs}` con la severidad apropiada según el caso, y se la pasa a `showToast`. Los duraciones varían: 5s para info trivial ("Modelo al día"), 10s para warnings importantes ("Inconsistencia detectada"), 0 (persistente) para casos críticos.

## Gotchas

- **Toast host se crea lazy**: el primer `showToast` crea `#toast-host` y lo appendea al body. Si el DOM se borra (no debería), se recrea solo.
- **Modal con body string vs HTMLElement**: el modal acepta string (lo splittea por `\n` para multi-línea) o un HTMLElement (lo appendea directo).
- **`withButtonLoading` con `btn=null`**: el helper ejecuta la fn sin loading state. Útil para handlers que pueden ser invocados desde lugares donde no hay botón (e.g., polling automático).
- **Si el handler hace `fetchState()` al final**, el `render` reemplaza el innerHTML y el botón que tenía loading state desaparece. El `finally` corre sobre el botón viejo (que ya no está en DOM) — no hace daño, solo no es visible. Para que sea correcto, los handlers cierran su loading state ANTES de que `fetchState` re-renderice.
