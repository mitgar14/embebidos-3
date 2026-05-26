---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: complete
stopped_at: Fase 6 (Cámara Local del Nano) EJECUTADA y VALIDADA en hardware el 2026-05-25. Deploy de nano_server.py + camera_capture.py con backup (nano_server.py.bak-pre06) + verificación /health + rollback automático. Pipeline GStreamer HW validado (C920 640x480), /ws/local emitió frame+detecciones (inferencia ~35ms), Dashboard mostró video local 4:3 con overlay; cámara liberada al cerrar; modo remoto intacto. CAM-01..04 cerrados. Milestone v1.0 con las 6 fases completas. NOTA DEMO: el modo local exige equipo+Nano en la MISMA red local (por el túnel Headscale DERP cae a ~1fps; en LAN va a ~14fps).
last_updated: "2026-05-25T21:30:00.000Z"
last_activity: 2026-05-25
progress:
  total_phases: 6
  completed_phases: 6
  total_plans: 9
  completed_plans: 9
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-24)

**Core value:** Dashboard live a ~14 fps con WS reconectante, fluido y profesional en la sustentación del 2026-05-26
**Current focus:** Milestone v1.0 completo (6 fases). Fase 06 (Cámara Local del Nano) validada en hardware. Pendiente: ensayar la demo del 2026-05-26 con equipo+Nano en la misma red local

## Current Position

Phase: 06 (Cámara Local del Nano) COMPLETA y validada en hardware
Plan: 3 of 3 (todos desplegados; deploy con backup+rollback, /ws/local validado, Dashboard mostró video local en LAN)
Status: Milestone v1.0 con las 6 fases completas. Demo lista (el modo local exige equipo+Nano en la misma red local)
Last activity: 2026-05-25

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 2
- Average duration: 5,5 min
- Total execution time: 0,2 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. Fundación | 2/3 completo | 11 min | 5,5 min |

**Recent Trend:**

- Last 5 plans: 01-01 (8 min), 01-02 (3 min)
- Trend: acelerando

*Updated after each plan completion*
| Phase 05 P01 | 8 | 2 tasks | 6 files |
| Phase 05 P02 | 20min | 2 tasks | 2 files |
| Phase 06 P01 | 5min | 2 tasks | 2 files |
| Phase 06 P02 | 5min | 2 tasks (T3 deploy diferida) | 1 file |
| Phase 06 P03 | 18min | 2 tasks (T3 checkpoint humano diferido) | 2 files |

## Accumulated Context

### Decisions

Decisiones canónicas en PROJECT.md Key Decisions. Resumen relevante:

- Roadmap: SolidJS 1.9 + Vite + Bun + Tailwind 4 + HashRouter; carpeta `web/`
- Paleta: Radix Slate dark + IBM Cyan 50 (`#1192e8`); tipografía IBM Plex Sans + IBM Plex Mono
- WS: clase `ReconnectingWebSocket` vanilla (~180 LOC) con backoff exponencial + jitter + heartbeat ping/pong
- Orquestación: único `web.ps1` (pwsh 7, cross-platform) con `-Action start|stop|restart|status`
- Canvas a 14 fps: signals estratificadas por frecuencia; WS→frame con throttle rAF; `createSignal` (no store) para el frame
- Clase→color en canvas: paleta Wong (vidrio=#56B4E9 sky, papel=#E69F00 ámbar, plástico=#009E73 bluish-green)
- Deadline: 2026-05-26 (2 días); prioridad absoluta = Dashboard live fluido
- Tema claro/oscuro (DES-05) promovido de v2 a v1: arquitectura de tokens semánticos dual-theme desde la Fase 1 (CSS vars que conmutan por `data-theme` en `<html>`, persistencia en `localStorage`, primer arranque por `prefers-color-scheme` con fallback oscuro). Vercel/Geist como referencia tiene toggle claro/oscuro, así que refuerza la estética buscada
- [Plan 01-01] Template solidjs/templates/ts vacío — estructura creada manualmente desde RESEARCH.md; equivalente al scaffold oficial
- [Plan 01-01] bun.lock (texto plano) en lugar de bun.lockb (binario) — Bun 1.3.6 cambia formato
- [Plan 01-01] Acento light #0072c3 (Cyan 70, WCAG 5.01:1) — NO #1192e8 (3.33:1, falla AA en fondo claro)
- [Plan 01-02] Parámetros de backoff sin ajustar: minDelay=1000, maxDelay=30000, growFactor=1.5, connectionTimeout=5000, minUptime=3000, pingInterval=20000, pongTimeout=8000 — calibrados para DERP/Headscale
- [Plan 01-02] setWsUrl lanza Error si esquema no es ws:// o wss:// (T-02-01); wsStore instanciado a nivel de módulo (no en componente)
- [Plan 01-03] bun install sin --frozen-lockfile: bun.lock (texto plano de Bun 1.3.6) no requiere el flag; más robusto para la demo
- [Plan 01-03] $IsMacOS explícito en Kill-ProcessTree (elseif branch) — cumple criterio de ≥2 ocurrencias del plan
- [HUB-02] Salud del Nano derivada del WS + sondeo /health. (Refinado en Fase 3, ver abajo.) La última inferencia se marca con cada mensaje no-pong del WS (frame/detección)
- [Fase 3] El Dashboard reusa el `ws` global reconectante (no crea uno propio): la reconexión y su indicador (DASH-04) salen del signal `wsStatus` sin trabajo extra. Captura imperativa (rAF + backpressure maxInFlight=2); el frame NO pasa por signals (solo métricas/controles son reactivos)
- [Fase 3] Protocolo WS verificado contra el dashboard previo (scripts/dashboard/app.js, fuente autoritativa): browser envía JPEG binario (el server cuenta frames por conexión) + {type:conf}; Nano responde {ok,seq,bboxes:[{x1,y1,x2,y2,cls_name,conf}],t_infer_ms}. fps/retardo/net se calculan en cliente; GPU temp y RAM vienen de GET /health
- [Fase 3] /health se sondea SOLO con el WS activo: si el WS conecta el Nano es alcanzable y /health responde sin error; con el WS caído no se sondea (ERR_CONNECTION_TIMED_OUT no es capturable por JS y ensuciaría la consola). Corrige el sondeo continuo inicial
- [Fase 3 · pulido] Máquina de estados de cámara idle/starting/live/error: durante el arranque (getUserMedia) muestra "Cargando cámara…" en vez de "Cámara detenida", para no confundir procesamiento con cámara apagada. Una sola llamada a getUserMedia (enumerateCams solo lista dispositivos tras permiso concedido)
- [Fase 3 · pulido] Recarga dura (Ctrl+Shift+R) ya no cae en "No se pudo abrir la cámara": listener pagehide libera los tracks antes de recargar + reintento automático en startCam con delays 0/400/1.000 ms (no reintenta ante NotAllowedError). No 100% testeable en headless; verificado por razonamiento + mock
- [Fase 3 · pulido] Checkbox "Mostrar cámara" eliminado (la cámara siempre visible). Métricas APILADAS (no inline): el usuario rechazó la versión comprimida; en su viewport real entra sin scroll
- [HUB-02 · pulido] Etiqueta "salud" eliminada de NanoStatus: "salud enlazando" era confuso y redundante con el dot de conexión. El hub muestra solo conexión + última inferencia
- [Fase 4] Engine del Modelo construido contra nano_server.py: /model/state (poll gateado por wsStatus), /model/build {force}, /jobs/{id}/logs (SSE, eventos 'log'/'done'), /jobs (historial), /model/rollback, /model/check-updates. Logs con buffer no reactivo + flush por rAF a un <pre> imperativo (perf). URL HTTP via nanoStore.nanoHttpBase(). Referencia de UI: scripts/dashboard/modelo.js
- [Fase 4] Labelling NO tiene backend en el Nano (es offline: auto-etiquetado en Vast.ai / server WSL2 en standby scripts/labeling/server/main.py). Decisión: construir Labelling como editor 100% client-side (cargar imágenes del disco, dibujar/editar bboxes, asignar clase vidrio/papel/plástico, export YOLO .txt + data.yaml). Autocontenido, cero red, demoable. IMPLEMENTADO (commit 55498f0): canvas con dibujo/mover/redimensionar (8 tiradores, coords de imagen), 3 clases con color Wong (teclas 1/2/3), navegación + tira de thumbnails con contador, export YOLO en .zip (data.yaml + images/ + labels/) vía JSZip y .txt de la imagen actual. CLASS_ID 0=glass/1=paper/2=plastic. Verificado en vivo (Playwright + imágenes sintéticas, .zip inspeccionado)
- [Fase 4] Bboxes del WS salen en PÍXELES del frame original (nano_server.py:243-249, clamp a ow/oh), NO normalizados; el Dashboard que asume píxeles es correcto. Descarta un falso bug del reporte del explorador (que decía 0..1)
- [Fase 3 · bug] CORREGIDO: Brave exige activación de usuario para getUserMedia AUNQUE el permiso esté 'granted' (no solo en 'prompt'); por eso un F5 simple caía en error y solo "Reintentar"/un clic funcionaba. El gating por Permissions API ('granted' → auto-inicio) NO basta en Brave. Fix definitivo: si el arranque sin gesto da NotAllowedError, armar listener global pointerdown/keydown → la 1ª interacción en cualquier parte abre la cámara (sin cazar un botón), mostrando "Haz clic para iniciar". Chrome sí permite getUserMedia sin gesto con permiso concedido (cero clics). Distinción de bloqueo real: gesto presente + NotAllowedError → permiso bloqueado
- [Fase 3 · bug] El <select> de cámara se quedaba pegado en la primera al cambiar de dispositivo: enumerateCams() hacía setDevices() con objetos nuevos en cada arranque, forzando a <For> a reconstruir las <option> y reseteando la selección visible. Fix: enumerateCams solo re-emite si la lista cambió; selectedDevice se sincroniza con el deviceId REAL del track abierto; un createEffect re-afirma el valor del select al cambiar opciones
- [Fase 3 · bug] CORREGIDO (commit 1fd1f39): Retardo y Transferencia siempre en "—" salvo en la primera carga. Causa raíz (confirmada por SSH al Nano + prueba del usuario): el Nano numera `seq` por CONEXIÓN WS (nano_server.py:1004/1015) y el cliente reiniciaba `seqOut` por cada MONTAJE del componente Dashboard; como el `ws` es global y no se reabre al navegar Hub↔Dashboard, el seq del Nano queda adelantado y `pendingFrames.get(msg.seq)` siempre da undefined. Predicción (t_infer_ms) seguía bien porque no usa seq. Fix: reemplazar el Map seq→sendTs por una cola FIFO de timestamps (el Nano procesa/responde en orden estricto vía `await future`, así que orden de respuestas == orden de envíos → shift()). De paso libera inFlight también en ok:false (queue_full/engine_unavailable), que antes podía congelar el envío durante un build. Verificado en vivo (Nano arriba + cámara sintética en Playwright): tras navegar por SPA, Retardo = Predicción + Transferencia (62 = 32 + 30 ms)
- [Diseño · acento] Énfasis monocromo (commit 70cb544): --accent = blanco #fff en oscuro e invertido a casi-negro #1c2024 en claro (el blanco sería invisible sobre fondo claro). --accent-text se pone al inverso para texto legible sobre el acento. engine.tsx: botón primario usa text-accent-text (no text-white) y el estado "compilando" se ata a var(--accent), con el badge pintado vía color-mix (acepta variables CSS, no solo hex). Los colores semáforo (verde/ámbar/gris) y de clase (Wong) NO son acento y se conservan. Verificado con Playwright en ambos temas sobre hub/dashboard/engine
- [Fase 4 · refinamientos Labelling] (commit a600e33) tres cambios pedidos por el usuario: (1) drag&drop de imágenes en cualquier parte del editor, con overlay "Soltá las imágenes para cargarlas"; addFiles unifica el input de archivos y el drop, el overlay aparece en dragenter (contador dragDepth para enter/leave anidados) y se limpia al soltar. (2) El formato YOLO pasa de un párrafo fijo a tooltip gráfico (InfoTip, role=tooltip, abre hacia arriba anclado a la derecha), igual al patrón del hub, junto al título "Exportar". (3) Fix Dark Reader del botón primario blanco que "parecía no existir": se declara color-scheme dark/light en los tokens + meta color-scheme en index.html, así Dark Reader respeta el tema propio y no doble-invierte. Verificado e2e con Playwright (overlay en dragenter, drop carga 2 imgs y limpia overlay, InfoTip opacity 1 con texto exacto, botón bg rgb(255,255,255)). NOTA DE VERIFICACIÓN: `playwright-cli goto` a una URL que solo difiere en el hash (#/labelling) NO recarga el bundle (navegación same-document); usar `playwright-cli reload` tras cada build o el test corre contra código viejo
- [Fase 4 · refinamientos Labelling 2] (commit a94d6df) cinco arreglos pedidos por el usuario: (1) BUG reactividad: al cambiar la clase de una caja seleccionada, la lista CAJAS no actualizaba texto/dot (el canvas sí). Causa: `<For>` keyed por referencia + boxesView()=[...boxes] (mismas refs) → no re-renderiza filas, y las expresiones internas no leían señal. Fix: patrón tick() en dot/etiqueta/tamaño de la fila (igual que el badge del thumbnail). Verificado: dibujar vidrio→tecla 3→fila pasa a plástico con dot verde rgb(0,158,115). (2) Instructivo de atajos como tooltip gráfico (kbd) en CLASE ACTIVA; InfoTip generalizado a children JSX + placement (top/bottom) + width; abre hacia abajo (.info-tip-down CSS). (3) Teclas A/D para navegar imágenes (además de flechas). (4) Cursor dinámico en canvas: hit-test en onPointerMove idle → tirador=ew/ns/nwse/nesw-resize, cuerpo=move, resto=crosshair (signal hoverCursor + style). (5) Tooltip de export reformateado (antes texto corrido ilegible): patrón YOLO en bloque mono + 3 clases como leyenda con índice y dot Wong. GOTCHA SolidJS: combinar `class` dinámico con `classList` borra las clases del classList al setear className; mover info-tip-down al string de `class` (solo info-tip-open queda en classList, se togglea sin wipe). Todo verificado e2e con Playwright
- [Fase 4 · Labelling persistencia + galería] (commit 596f16a) dos features pedidas: (1) PERSISTENCIA de sesión por 30 min vía IndexedDB (nuevo lib/labelStore.ts). IndexedDB y NO localStorage porque hay que guardar los File de las imágenes como Blob (excede el límite de localStorage). Dos stores: `blobs` (id->{name,blob,w,h}, pesado, se escribe al cargar imágenes) y `meta` (current->{savedAt,idx,order,boxes}, liviano, autosave debounced 600ms en cada cambio). LabImage ahora tiene `id` (crypto.randomUUID). Restauración SILENCIOSA en onMount (reconstruye File/url/img desde el blob) si Date.now()-savedAt <= 30min; si expiró, clearSession. Flag `hydrated` evita que el autosave inicial pise con vacío antes de restaurar. Botón Vaciar sesión (icono basura, con confirm) en el header. Verificado: cargar 2 imgs + dibujar caja vidrio → reload duro → restaura 2 imgs + caja intactas. (2) VISTA GALERÍA (overview, no edita): signal view editor/gallery, botón icono grilla en header (estado activo con border-accent + bg-accent-bg) alterna; grid responsive de celdas = img + overlay SVG de las cajas (vector-effect non-scaling-stroke así el trazo se mantiene ~2px sin importar el viewBox, color Wong por clase); tooltip al hover (group-hover) con nombre + desglose por clase (conteo vidrio/papel/plástico); clic en celda navega al editor en esa imagen. Decisiones de diseño confirmadas con el usuario: toggle = botón con icono (no segmented), celda = mínima + tooltip de desglose, restauración = silenciosa (sin banner). Todo verificado e2e con Playwright
- [Fase 4 · Labelling fix canvas galería] (commit 5ef213a) BUG: al volver de galería al editor la imagen no se dibujaba (canvas negro), ni las siguientes. Causa: envolver el editor en `<Show when={view()==='editor'}>` DESMONTABA el `<canvas>` al ir a galería; al volver, Show creaba un canvas NUEVO, pero `ctx` y el ResizeObserver (obtenidos una sola vez en onMount) seguían apuntando al canvas viejo destruido → redraw pintaba en un elemento desconectado. Los thumbnails sí cargaban (usan `<img src>`). Fix: NO desmontar el editor; ocultarlo con `style={{display: view()==='editor'?'flex':'none'}}` (inline gana a flex/hidden), así canvas+ctx+observer sobreviven; + un createEffect que re-mide (resizeCanvas) y redibuja al volver a 'editor' (estuvo en 0px). LECCIÓN DE VERIFICACIÓN: mi test de galería anterior solo chequeó el DOM (posición), no los píxeles del canvas, por eso no detecté el bug; ahora verifico con getImageData (nonBg pixels) que el canvas realmente dibujó. [boundary] Envolver un canvas imperativo en <Show> rompe el ctx/observer cacheados en onMount; usar display toggle, no montar/desmontar
- [Fase 4 · Labelling modal vaciar] (commit 2ab2fb9) reemplazado el confirm() nativo (se veía fuera del sistema de diseño) por un modal propio: signal confirmClear, `<Show>` con backdrop bg-bg-app/70 + panel hairline (role=dialog, aria-modal), botones Cancelar (BTN) y Vaciar (destructivo rojo #e5484d). Escape y clic en backdrop cancelan; onKey bloquea los atajos del editor mientras el modal está abierto. Verificado: no dispara diálogo nativo, Cancelar conserva, Vaciar limpia

- [Fase 5 · planeada] Integración de 2 páginas ESP32 (Guía 3D + Control de servos) encauzada por GSD. Investigación cerrada (investigaciones/2026-05-25-integracion-esp32-servos.md). Fase 5 en ROADMAP/REQUIREMENTS (HUB-03, GUIA-01..03, CTRL-01..03). CONTEXT.md escrito con las decisiones del usuario; el planner generó 3 planes (05-01 andamiaje, 05-02 Guía 3D, 05-03 Control MQTT) en 2 waves; plan-checker PASSED (0 blockers, 1 warning menor en 05-02 T1 resoluble en ejecución). Decisiones clave: la Guía sigue 100% en 3D (porte fiel del repo, solo revestido al estilo Vercel, colores de cable conservados); el Control usa mqtt.js sobre wss://broker.emqx.io:8084/mqtt SIN backend, estado dual por el LWT retenido servos/<id>/online, UI autoconfigurada por telemetría, movimientos serializados por brownout; orden del hub: Guía, Dashboard, Engine, Labelling, Control.
- [Phase ?]: three@0.184.0 y mqtt@5.15.1 instalados via bun add; @types/three como devDependency
- [Phase ?]: GuiaIcon: SVG de placa con 5 pines; ControlIcon: SVG de 2 sliders horizontales con circulo deslizante
- [Phase ?]: guia.tsx expone div#guia-container como punto de montaje del canvas Three.js del Plan 05-02
- [Phase ?]: Sin fugar contexto WebGL al navegar entre rutas en SolidJS

- [Fase 6 · planeada] Cámara local del Nano (C920 en /dev/video0) como 2a fuente del Dashboard, encauzada por GSD. Cámara verificada lista (sin config extra). Investigación cerrada (investigaciones/2026-05-25-camara-local-server-jetson.md, copiada a 06-RESEARCH.md). Fase 6 en ROADMAP/REQUIREMENTS (CAM-01..04). 3 planes en 3 waves secuenciales: 06-01 captura+worker dual (autónomo), 06-02 transporte+deploy (checkpoint humano), 06-03 frontend (checkpoint humano). plan-checker dio CONCERNS, resueltos quirúrgicamente. Decisiones clave: pipeline GStreamer `v4l2src io-mode=2 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! BGRx ! videoconvert ! BGR` (frame NATIVO 640x480, SIN forzar 416 para no aplastar el 4:3; el TRTWorker letterboxea internamente igual que en remoto y postprocess devuelve bboxes en 640x480); `cap.read()` en hilo daemon (NO callbacks GLib, mueren cada ~25s por GIL); TRTWorker bifurca por tipo (bytes=imdecode remoto intacto, np.ndarray=local salta imdecode); transporte WS binario (frame JPEG + bboxes JSON) con `imencode` SOLO en el hook del worker (no bloquea el event loop async del /ws remoto); MJPEG de respaldo como generador síncrono; arbitraje de modo único (un solo /ws/local enciende la cámara) con un único `finally` dueño de la liberación de /dev/video0; el cliente dibuja a las dims reales del frame reusando `drawDetections`.
- [Phase ?]: 06-01: TRTWorker bifurca por tipo del item (bytes remoto vs np.ndarray local) con ruta de inferencia compartida (_letterbox); modo remoto byte-por-byte equivalente
- [Phase ?]: 06-01: el resultado del modo local sale por el hook on_local_result(result, frame_original_640x480), no por future; lo conecta 06-02
- [Phase ?]: 06-01: pipeline GStreamer entrega 640x480 BGR nativo (sin redimensionar a 416 en nvvidconv); el worker letterboxea, conservando el 4:3
- [Fase 6] 06-02: endpoint nuevo /ws/local (no extender /ws); por frame manda binario JPEG 640x480 + JSON {ok,bboxes,t_infer_ms,seq} desde un snapshot único (get_latest una vez por iteración, ambos sends del mismo snap para no desemparejar). Contrato congelado, lo consume 06-03 idéntico
- [Fase 6] 06-02: el cv2.imencode vive SOLO en el hook _on_local_result (hilo del worker), nunca en el event loop; /ws/local y /camera/mjpeg solo leen el latest-frame ya codificado (LocalStreamState con lock). Mitiga la degradación del /ws remoto
- [Fase 6] 06-02: arbitraje de modo único (local_active/local_lock): la 2a conexión a /ws/local recibe local_busy. El arbitraje y el arranque de cámara van FUERA del try/finally; el finally (único dueño de la limpieza, _cam_capture.stop() + reset) solo se arma cuando ESTA conexión encendió la cámara. /camera/mjpeg NO enciende la cámara (espejo de solo lectura, 409 local_not_active si no hay modo local) -> una sola puerta de encendido de /dev/video0
- [Fase 6] 06-02: /camera/mjpeg es SÍNCRONO (Starlette lo corre en threadpool, el time.sleep no bloquea el loop); el generador yield-ea bytes multipart (boundary --frame). Verificado por AST: camera_mjpeg y su gen son síncronos, un solo finally en ws_local
- [Fase 6] 06-03: cliente WS del modo local en lib/localCamera.ts (LocalCameraClient): WS crudo a /ws/local (no el ws global), binaryType arraybuffer, empareja el frame binario con el siguiente JSON ok:true (orden estricto), decodifica con createImageBitmap fuera del hilo principal, close() idempotente. Sin canvas fuera de pantalla ni Web Worker (diferido a V2-02)
- [Fase 6] 06-03: el Dashboard dimensiona el canvas del frame local a bitmap.width/bitmap.height en el primer frame (640x480 4:3, sin aplastar; nunca un tamaño cuadrado fijo) y dibuja el overlay con drawDetections a esas dims; bitmap.close() tras drawImage. Modo remoto getUserMedia/captureLoop/sendFrame intacto, gated tras source==remota; bootRemote() extraído y reutilizado por onMount y el cambio a remota
- [Fase 6] 06-03: en local Retardo/Transferencia van en guion (sin envío de frame del cliente) y Ritmo objetivo se deshabilita (lo marca el Nano). CAM-01 cerrado (selector verificado con Playwright 10/10: conmuta, estado de error claro sin spinner infinito, remoto intacto); CAM-03/CAM-04 siguen Pending hasta el checkpoint humano con el Nano vivo

### Pending Todos

- **DIFERIDO — Deploy de 06-02 al Nano (T3) + verificación en hardware:** cuando el túnel SSH vuelva. Comandos exactos en 06-02-SUMMARY.md. Regla de oro: confirmar que NO hay build activo (curl /jobs/active == null) ANTES de copiar; NUNCA tocar scripts del Nano durante un build. Pasos: scp nano_server.py + camera_capture.py -> restart embebidos3-server.service -> validar pipeline gst-launch -> abrir /camera/mjpeg con un /ws/local activo y confirmar video 4:3 sin aplastar + liberación de /dev/video0 al cerrar.
- Fase 6 Wave 3: 06-03 frontend selector remota/local + render local (checkpoint humano). Consume el contrato de /ws/local ya congelado. La C920 ya está verificada lista en el Nano.
- CAM-03 y CAM-04 siguen Pending: se cierran cuando 06-03 exista y la verificación en hardware de 06-02 pase.
- Ensayar la demo del 2026-05-26 con ambos modos de cámara (remota getUserMedia + local C920).

### Blockers/Concerns

- **Demo clock:** La demo es en 2 días (2026-05-26). Las Fases 3 y 4 pueden correr en paralelo tras la Fase 2. Si el tiempo escasea, la Fase 4 (Engine y Labelling) puede mostrase degradada.
- **Tailscale/Headscale:** El WS del Nano está en `ws://100.64.0.2:8000/ws` vía Headscale. Si la red DERP está caída en la sustentación, el Dashboard mostrará el banner de reconexión (comportamiento esperado).

## Deferred Items

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | Command palette (⌘K) | Deferred | Init |
| v2 | Web Worker + OffscreenCanvas | Deferred | Init |

## Session Continuity

Last session: 2026-05-25T20:42:30.510Z
Stopped at: 06-02 CODIGO completo (T1 /ws/local + T2 /camera/mjpeg). T3 deploy al Nano + checkpoint humano DIFERIDOS por Nano inalcanzable (SSH timeout) y usuario ausente. Comandos de deploy listos en 06-02-SUMMARY.md. Reanudar con el deploy cuando el túnel SSH vuelva, luego 06-03.
Resume file: None
