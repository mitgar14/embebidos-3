# 08 · Sistema visual

Lenguaje visual del dashboard. Decidido durante la implementación y refinado en iteraciones con el usuario. Aplicado en `scripts/dashboard/style.css` (~1150 líneas).

## Inspiraciones y referencias

- **Impeccable `frontend-design` skill** (anti-AI-slop checklist)
- **AWS Cloudscape** (timestamps, key-value displays)
- **Linear / Vercel / Stripe** (editorial metadata sin nested cards)
- **El "editorial-v3" del proyecto Notion-Tracking del usuario** (regla memorizada: sin cards/sombras, tipografía como protagonista, separadores con líneas, sin glassmorphism, sin gradientes morados-azules)

## Sistema de color — OKLCH

Paleta tintada hacia mostaza (hue 80). Modo oscuro siempre.

```css
:root {
  --bg:           oklch(18% 0.012 80);
  --bg-elev-1:    oklch(22% 0.014 80);
  --bg-elev-2:    oklch(26% 0.016 80);
  --bg-elev-3:    oklch(30% 0.018 80);
  --border:       oklch(34% 0.020 80);
  --border-strong:oklch(45% 0.025 80);

  --text:         oklch(95% 0.010 80);
  --text-muted:   oklch(68% 0.015 80);
  --text-dim:     oklch(48% 0.012 80);

  --accent:       oklch(72% 0.13 80);          /* mostaza/ocre */
  --accent-soft:  color-mix(in oklch, var(--accent) 18%, transparent);
  --accent-text:  oklch(20% 0.020 80);

  --ok:           oklch(72% 0.17 145);
  --warn:         oklch(80% 0.16 80);
  --err:          oklch(68% 0.20 25);

  --glass:        oklch(76% 0.16 150);
  --paper:        oklch(70% 0.15 245);
  --plastic:      oklch(72% 0.17 55);
}
```

Decisiones:
- **Neutrales tintados** hacia hue 80 (mostaza) — crea cohesión sin que se note conscientemente
- **Sin pure black/white** — siempre con un toque de tint
- **Severidad por color funcional** (ok verde, warn mostaza-cálido, err rojo) sin gradientes
- **Clases vidrio/papel/plástico** con hues distintos para reconocimiento instantáneo

## Tipografía

Dos fuentes variables locales (sin Google Fonts):

```css
--font-sans: "Source Sans 3", ui-sans-serif, system-ui, sans-serif;
--font-mono: "Source Code Pro", ui-monospace, "Consolas", monospace;
```

Decisiones:
- **NO Inter** (overused, AI-default)
- **NO monospace para todo lo técnico** — solo para datos comparables (hashes, timestamps, IDs)
- **Tabular-nums activado** (`font-variant-numeric: tabular-nums lining-nums`) en valores que se alinean en columna

Tamaños:
- `--fs-title: 16px` (brand)
- `--fs-h2: 10px` (cards live tab)
- `--fs-chip-label: 10px`
- `--fs-chip-value: 14px`
- `--fs-body: 13px`
- `--fs-meta: 11px`

## Capitalización (regla del usuario)

> "No abuses de minúsculas en todo: solo en el header principal se ven bien; las demás es mejor verlas normal"

| Elemento | Caso | Decisión |
|---|---|---|
| Brand title `embebidos-3` | lowercase | sí (header principal) |
| Brand sub `live detection` | lowercase | sí (parte del brand) |
| Tab labels `live`, `modelo` | lowercase | sí (chips pequeños) |
| Chip labels (header) | lowercase | sí (chips pequeños de datos) |
| Card h2 sidebar live (`Servidor`, `Cámara`...) | Capitalize | sí (sentence case) |
| Side-card h3 sidebar modelo (`Servidor`, `HF Hub`...) | Capitalize | sí |
| Section h2 modelo (`Modelo activo`, `Compilando engine`) | Sentence case | sí |
| Botones (`Verificar actualizaciones`, `Recompilar engine`) | Sentence case | sí |
| Toast titles | Sentence case | sí |
| Modal titles | Sentence case | sí |
| Badge `ADOPTADO` | uppercase | sí (badge funcional, convención) |

Eliminado `text-transform: uppercase` de `.card h2` y `.side-card h3`. Conservado en `.brand-sub`, `.tab`, `.chip` porque son chips/labels pequeñas donde lowercase queda bien.

## Layout

### Estructura general

```
<body>
  <header class="topbar">  // 60px, sticky
    .brand · .tabs · .chips · .actions
  </header>
  <main class="layout" data-pane="live">          // grid: 1fr 340px
    <section class="stage"><video><canvas></section>
    <aside class="panel">.card × 4</aside>
  </main>
  <main class="layout-modelo" data-pane="modelo"> // grid: 1fr 340px
    <section class="modelo-main"><modelo-content></section>
    <aside class="modelo-side"><side-card × 4></aside>
  </main>
</body>
```

### Cards (live tab) vs sections (modelo tab)

- **Live tab**: usa `.card` con padding y border — apropiado para grupos de controles pequeños (server, cámara, inferencia, métricas) en sidebar densa
- **Modelo tab**: usa `.modelo-section` plana sin wrapper — el contenido es de lectura, no de controles densos. Separación por `border-top` hairline entre sub-bloques (identidad vs build info)

Patrón sustentado en Cloudscape: la jerarquía la hacen tipo y espacio, no contenedores.

## Iconografía

SVG inline en `<defs>` del HTML, referenciados con `<use href="#i-name">`. Set actual:
- `#i-activity` — pulse (chip fps)
- `#i-gauge` — speedometer (chip latency)
- `#i-thermometer` (chip temp)
- `#i-cpu` (chip RAM)
- `#i-radio` (chip status)
- `#i-camera` (action button)

Más SVG inline para la pestaña modelo (en `modelo.js`):
- Commit dot (3 puntos alineados) → label `Commit`
- Reloj con manilla → label `Entrenado`
- Key (llave) → label `Compilado`

Regla: **iconitos solo donde justifican** (commit, time, key). Para hashes y parámetros técnicos heterogéneos, omitir. Sustentado en research de doc 03.

## Tooltips nativos

`[data-tip="..."]::after` con `transition-delay: 500ms` para que aparezcan tras hover sostenido. Definidos para todos los chips, action buttons, slider labels y check labels. Sustentan el sistema de docs *in situ*.

## Componentes específicos

| Componente | Selector | Doc |
|---|---|---|
| Chip status (header) | `.chip-status[data-state="..."]` | live |
| Card collapsible | `.card[data-card="..."]` | live |
| Slider control | `.slider-row` | live |
| Modelo section | `.modelo-section` + `.meta-grid` | doc 06 |
| Build progress | `.build-progress > .build-progress-bar > .build-progress-fill` | doc 06 |
| Logs pane | `.logs-pane > .logs-stream` | doc 06 |
| Banner update / warn | `.banner-update`, `.banner-warn` | acento border-left |
| Toast | `.toast.toast-{info,success,warn,error}` | doc 07 |
| Modal | `.modal-overlay > .modal-dialog` | doc 07 |
| Button loading | `button.is-loading::before` | doc 07 |
| sr-only | `.sr-only` (aria-live regions) | doc 06 |

## Motion design

Sustentado en frontend-design skill reference:
- Solo `transform` y `opacity` (nunca layout properties)
- `ease-out-quart: cubic-bezier(0.25, 1, 0.5, 1)` — default refinado
- `ease-out-quint: cubic-bezier(0.22, 1, 0.36, 1)` — modal entrance (más dramático)
- **NO bounce/elastic** (anti-trope)
- Entradas 220ms, salidas ~75% (160ms)
- `prefers-reduced-motion: reduce` cubierto globalmente

## Cosas que NO usamos (anti-tropes)

- Glassmorphism (blur excesivo, glass cards)
- Gradient text ("hero metrics")
- Gradientes morados-azules
- Glow borders decorativos
- Rounded rectangles con drop shadows genéricos
- Bounce easing
- Sparklines decorativos
- Cards anidadas
- "Hero metric template" (número gigante + small label + supporting stats)
- Texto en gradiente
- Esquinas redondeadas uniformes en todo (variamos 4px / 6px según contexto)

## Cache busting

`<link rel="stylesheet" href="style.css?v=20260516-7" />`
`<script src="ui.js?v=20260516-7"></script>`

Bumpear el `?v=` parámetro cada vez que se hace deploy de CSS/JS. Sin esto, el browser sirve la versión cacheada.
