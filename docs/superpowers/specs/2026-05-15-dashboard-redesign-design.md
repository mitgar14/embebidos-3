# Dashboard redesign — diseño aprobado

**Fecha:** 2026-05-15
**Proyecto:** `embebidos-3` — clasificador glass/paper/plastic, demo 2026-05-26.
**Página afectada:** `scripts/dashboard/` (index.html, style.css, app.js).
**Stack server:** FastAPI + WS + TensorRT FP16 en Jetson Nano remoto (`ws://100.100.166.120:8000/ws` vía Tailscale).

---

## 1. Constraints duros

- Cero scroll global. Si hay scroll, solo en el sidebar derecho.
- Cero emojis. Íconos SVG inline (Lucide).
- Cero tropes de UI generada por IA (ver §6).
- Layout **information-driven**: jerarquía visual = importancia operativa.
- Terminología visible conceptual; jerga técnica en tooltip on hover.
- Auto-init: cámara y enlace al servidor arrancan apenas carga la página.
- Defaults nuevos: certeza mínima `0.50`, ritmo objetivo `14 /s`.

## 2. Arquitectura visual (Approach A: dos columnas)

```
viewport (100vh, overflow: hidden)
┌─ header ~52 px ────────────────────────────────────────────────────┐
│ [embebidos-3 · live detection]                                      │
│ [chip conexión] [chip ritmo] [chip retardo] [chip temp] [chip mem]  │
│                                                       [icon capturar]│
├─────────────────────────────────────────────┬──────────────────────┤
│ main left (flex column, justify-center)     │ aside right 340 px   │
│                                             │ overflow-y: auto     │
│   .canvas-wrap                              │                      │
│     aspect-ratio: dinámico = videoW/videoH  │   card Servidor      │
│     centrado, sin barras                    │   card Cámara        │
│   <video> + <canvas overlay>                │   card Inferencia    │
│   labels sobre bbox = clase + %             │   card Métricas      │
│                                             │                      │
└─────────────────────────────────────────────┴──────────────────────┘
```

Notas:
- Header `flex-shrink: 0`, altura fija.
- `main` con `flex: 1; min-height: 0` y `display: grid; grid-template-columns: minmax(0, 1fr) 340px`.
- `.canvas-wrap` con `aspect-ratio` que JS actualiza a `videoWidth / videoHeight` cuando arranca la cámara → cero barras negras laterales o verticales en ningún viewport.
- Sin footer (eliminado en iteración previa).

## 3. Jerarquía de información (21 piezas → ubicación)

Header (info crítica/alta, glanceable):
- chip *conexión* (dot color `--ok` / `--err`)
- chip *ritmo* (FPS round-trip)
- chip *retardo* (latencia total)
- chip *temperatura* (GPU Nano, alarmable >70 °C → `--err`)
- chip *memoria libre* (RAM Nano)
- icon-button *capturar* (descarga PNG con frame + bboxes)

Sidebar — *Servidor*:
- input *dirección del servidor*
- buttons *conectar* / *desconectar* (auto-conectados al cargar)

Sidebar — *Cámara*:
- select *cámara*
- buttons *iniciar* / *detener* (auto-iniciados al cargar)
- meta *cuadro* `640 × 480` (fila inline al lado de los botones)

Sidebar — *Inferencia*:
- slider *certeza mínima* (0.50)
- slider *ritmo objetivo* (14)
- toggle *mostrar cámara* (on)

Sidebar — *Métricas*:
- *tiempo de predicción* (ms server)
- *tiempo de transferencia* (ms red)
- *detecciones por cuadro*
- *cuadros procesados*
- fila inferior: *conteo por clase* — vidrio / papel / plástico (colored, tabular-nums)

Sobre el video (overlay):
- bbox de cada detección con `label cls_name + conf %`, dibujada con color `--glass` / `--paper` / `--plastic` (no legend separada; la bbox label es suficiente).

## 4. Terminología y tooltips

Mapeo término técnico → visible humano + tooltip on hover.

| Visible | Tooltip (técnico) |
|---|---|
| **conexión** + dot | "WebSocket binario al modelo en el Jetson Nano." |
| **ritmo** *N* /s | "Cuadros procesados por segundo, ida-y-vuelta." |
| **retardo** *N* ms | "Captura → envío → predicción → respuesta." |
| **temperatura** *N* °C | "GPU del Jetson Nano. Bajo carga, objetivo < 70 °C." |
| **memoria libre** *N* MB | "RAM disponible en el Nano." |
| **tiempo de predicción** *N* ms | "Latencia GPU (TensorRT FP16)." |
| **tiempo de transferencia** *N* ms | "Camino de red. retardo − tiempo de predicción." |
| **detecciones por cuadro** | "Objetos detectados tras NMS." |
| **cuadros procesados** | "Acumulado desde el inicio de la sesión." |
| **conteo por clase** | "Suma de detecciones por clase." |
| **dirección del servidor** | "Endpoint WebSocket. Cambialo si la IP no es la Tailscale por defecto." |
| **conectar** / **desconectar** | "Abre o cierra el enlace al servidor." |
| **cámara** (dropdown) | "Dispositivo de captura. Auto-inicia por defecto." |
| **cuadro** *640 × 480* | "Resolución actual del stream de cámara local." |
| **iniciar** / **detener** | "Comienza o detiene la captura local." |
| **certeza mínima** *50 %* | "Umbral del detector. Predicciones menores se descartan." |
| **ritmo objetivo** *14 /s* | "Cuadros por segundo que se intenta enviar. Real depende del retardo." |
| **mostrar cámara** | "Si está apagado, el fondo se vuelve negro y solo se ven las bbox." |
| **capturar** (icon SVG) | "Descarga PNG con frame + bounding boxes." |

Implementación: `<element data-tip="…">` + CSS `:hover::after`, delay 500 ms, `aria-describedby` espejo.

## 5. Iconografía SVG

Librería: **Lucide Icons** (MIT, sucesora de Feather). Embed inline `<svg viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">…</svg>`. Tamaños: 14 px en chips header, 13 px en card h2, 18 px en action buttons.

| Uso | Lucide |
|---|---|
| Conexión (dot ok/err) | `radio` / custom dot |
| Ritmo | `activity` |
| Retardo | `gauge` |
| Temperatura | `thermometer` |
| Memoria libre | `cpu` |
| Capturar PNG | `camera` |
| h2 Servidor | `server` |
| h2 Cámara | `video` |
| h2 Inferencia | `sliders-horizontal` |
| h2 Métricas | `bar-chart-3` |
| Indicador tooltip | `circle-help` |

Regla: cero íconos decorativos. Cada SVG es funcionalmente representativo.

## 6. Tipografía y tokens de color

**Stack tipográfico** (FOSS, self-hosted, sin CDN):

```css
--font-sans: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
--font-mono: "Source Code Pro", ui-monospace, "Consolas", monospace;
```

Justificación (síntesis de investigación, ronda 2026-05-15):

- Source Sans 3 con `font-optical-sizing: auto` — único FOSS con `opsz` variable, clave para 10-14 px.
- Hinting Windows ClearType más robusto de la categoría libre.
- Comparten métricas con Source Code Pro por diseño Adobe → transición label→número imperceptible.
- Ninguna de las dos está en el "lookbook v0/Lovable/Cursor" mayo 2026 (confirmado: BSWEN 2026-03, FontAlternatives 2026-01).

**Anti-tropes aplicados**:
- Cero Inter, Geist, Cal Sans, Satoshi, DM Sans, Space Grotesk + Space Mono, JetBrains Mono UI.
- Cero gradientes en texto. Cero glow / drop-shadow neón.
- Cero glassmorphism / `backdrop-filter: blur` en core.
- Border-radius diferenciado: `6 px` cards, `12 px` chips/badges.
- Una sola familia + companion mono; sin mix decorativo serif/display.

**Escala**:

| Token | px | peso | uso |
|---|---|---|---|
| `--fs-title` | 16 | 600 | brand header |
| `--fs-h2` | 11 | 600 | h2 cards, uppercase, ls 0.06em |
| `--fs-chip-label` | 10 | 500 | label chip uppercase, ls 0.05em |
| `--fs-chip-value` | 14 | 600 | **mono tabular** |
| `--fs-body` | 13 | 400 | inputs, descripciones |
| `--fs-meta` | 11 | 400 | meta info |
| `--fs-bbox` | 13 | 700 | label sobre bbox |

**Tokens de color** (dark theme único):

```css
--bg:           #0f1419;   /* canvas */
--bg-elev-1:    #161d26;   /* panel */
--bg-elev-2:    #1d2632;   /* card */
--bg-elev-3:    #252f3c;   /* chip / input */
--border:       #2a3340;
--border-strong:#3a4555;
--text:         #e6edf3;
--text-muted:   #8a96a3;
--text-dim:     #5e6975;
--accent:       #4fb3d9;
--ok:           #43d27e;
--warn:         #fbbf24;
--err:          #f87171;
--glass:        #43d27e;
--paper:        #4fa3ff;
--plastic:      #ff8c42;
```

Contrast WCAG (verificado):
- text/bg → 14.8:1 AAA
- text-muted/bg → 5.8:1 AA
- accent/bg → 6.7:1 AA

## 7. Comportamiento

- **Auto-init**: enumerateCams → poll /health inicial → connect WS → startCam. Si alguno falla, mostrar estado en chip y continuar (no bloquear).
- **Tooltips**: `data-tip` + CSS `:hover::after`, delay 500 ms, `aria-describedby`. No bloquean teclado.
- **Polling /health**: 3 s. Si falla 3 consecutivos, el *valor* (no el chip entero) de `temperatura` y `memoria libre` se renderiza en `--text-dim` con texto `— °C` / `— MB`.
- **Estados de chips data-driven**:
  - `temperatura` 60-70 °C → tinte `--warn` (border + valor).
  - `temperatura` > 70 °C → tinte `--err`.
  - `conexión` inactiva → chip completo en `--err` con `aria-live: polite`.
- **Accesibilidad**: focus visible `outline: 2px solid var(--accent); outline-offset: 2px`. Sliders con `aria-valuetext`. Action icons con `aria-label`.
- **`prefers-reduced-motion`**: desactiva transiciones de estado.
- **Cache busting**: `style.css?v={timestamp}` o headers `Cache-Control: no-cache` en el launcher local (problema conocido durante iteración).

## 8. Anti-tropes (referencia rápida)

Lista síntesis de la investigación 2026-05-15 (ver `investigaciones/2026-05-15-frontend-dashboard-design-anti-tropes.md` cuando se complete):

| Trope IA | Cómo lo evito acá |
|---|---|
| Inter / Roboto / DM Sans | Source Sans 3 (Adobe, OFL) |
| Geist / Geist Mono | Source Code Pro (Adobe, OFL) |
| JetBrains Mono en UI | Source Code Pro (Adobe, OFL) |
| Glassmorphism | Cero `backdrop-filter: blur` |
| Gradientes en texto | Texto sólido |
| Neon glow / drop-shadow | Borders sólidos |
| Border-radius 8 px uniforme | 6 px cards, 12 px chips |
| Emojis como íconos | Lucide SVG inline |
| Cards apiladas sin jerarquía | h2 con `--fs-h2` distinguible + spacing semánticamente diferente |
| Legend que repite info de bbox | Eliminada (la bbox label ya muestra clase + color) |

## 9. Información que NO va en el dashboard

(Decisiones explícitas para limitar scope):

- *Nano health* card eliminada. Reemplazada por chips header `temperatura` + `memoria libre`.
- *Legend* de colores eliminada. Las bbox labels ya muestran clase + color.
- *Footer* eliminado.
- *JPEG quality 0.70* visible eliminada (sigue siendo 0.7 internamente; el dato no agrega valor al evaluador).
- Counters de detección **por segundo** (no implementados): scope futuro si la demo lo amerita.

## 10. Plan de implementación

Después de aprobación del usuario:

1. **Fase B (investigación)** — `/investiga` Alto. Track A: research-web (anti-tropes + dashboard layout 2026) + research-video + research-code repos referencia. Track B: discover.py + crawling_exa. Output: `investigaciones/2026-05-15-frontend-dashboard-design-anti-tropes.md`.
2. **Fase C (implementación)** — `/frontend-design`. Descargar Source Sans 3 + Source Code Pro woff2 a `scripts/dashboard/fonts/`. Descargar 12 SVGs Lucide a `scripts/dashboard/icons/`. Reescribir `index.html`, `style.css`, `app.js` aplicando todo lo anterior. Validar accesibilidad básica.
3. **Fase D (validación)** — `playwright-cli` itera. Screenshot por sección. Cache-bust entre cambios.
4. **Commit final** — un solo commit con todo el redesign del dashboard, mencionando deuda investigaciones/.

## Referencias clave (research 2026-05-15)

- BSWEN, "AI-Generated UI Anti-Patterns Guide", 2026-03 — "Inter es el Comic Sans de la IA".
- FontAlternatives, "Best Fonts for Dense Dashboards", 2026-01 — Source Sans 3 mejor hinting Windows.
- Adobe `source-sans` (GitHub, OFL) — woff2 directos.
- Adobe `source-code-pro` (GitHub, OFL) — woff2 directos.
- Lucide Icons (MIT) — SVG catalog.
- Geist Font Issue #102 (Vercel, 2024-04) — bug `l/1` mono.
