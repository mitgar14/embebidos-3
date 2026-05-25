// web/src/routes/labelling.tsx
// Labelling (Fase 4, LBL-01..05): editor de bounding boxes 100% client-side.
// No hay backend en el Nano para el etiquetado (es un flujo offline), así que
// todo vive en el navegador: cargar imágenes del disco, dibujar/mover/
// redimensionar cajas, asignar una de las 3 clases (vidrio/papel/plástico) y
// exportar en formato YOLO (un .txt por imagen + data.yaml) empaquetado en .zip.

import { createSignal, createEffect, onMount, onCleanup, For, Show } from 'solid-js';
import { A } from '@solidjs/router';
import JSZip from 'jszip';
import { ThemeToggle } from '../components/ThemeToggle';
import { WONG, CLASS_LABEL_ES, type ClsName } from '../lib/detection';

const CLASSES: ClsName[] = ['glass', 'paper', 'plastic'];
const CLASS_ID: Record<ClsName, number> = { glass: 0, paper: 1, plastic: 2 };

// Caja en coordenadas de la IMAGEN original (píxeles). Trabajar en este espacio
// (no en el del canvas) hace el export YOLO directo y estable ante el zoom/fit.
interface Box { x: number; y: number; w: number; h: number; cls: ClsName; }
interface LabImage {
  name: string; url: string; file: File;
  w: number; h: number; img: HTMLImageElement; boxes: Box[];
}
type HandleId = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';
type Mode = 'idle' | 'draw' | 'move' | 'resize';

const HANDLE = 8;      // lado del tirador, en px de pantalla
const MIN_BOX = 6;     // caja mínima (px de imagen) para conservarla al soltar

const BTN = 'rounded-md border border-border px-3 py-1.5 text-sm text-text-primary ' +
  'hover:border-accent hover:bg-bg-surface transition-colors ' +
  'disabled:opacity-40 disabled:hover:border-border disabled:hover:bg-transparent';
const BTN_PRIMARY = 'rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-text ' +
  'hover:opacity-90 transition-opacity disabled:opacity-40 disabled:hover:opacity-40';

function IconBack() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}
function IconUpload() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M17 8l-5-5-5 5" />
      <path d="M12 3v12" />
    </svg>
  );
}
function IconDownload() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M7 10l5 5 5-5" />
      <path d="M12 15V3" />
    </svg>
  );
}
function IconInfo() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  );
}

// Tooltip informativo: elemento gráfico propio (role=tooltip), NUNCA el title
// nativo, igual al patrón del hub. Abre hacia arriba para no salirse del panel.
function InfoTip(props: { text: string }) {
  const [open, setOpen] = createSignal(false);
  return (
    <span class="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        aria-label="Formato de exportación"
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        class="flex items-center justify-center text-text-secondary hover:text-text-primary transition-colors"
      >
        <IconInfo />
      </button>
      <span
        role="tooltip"
        classList={{ 'info-tip-open': open() }}
        class="info-tip pointer-events-none absolute bottom-full right-0 mb-2 z-30 w-64
               rounded-md border border-border bg-bg-panel px-3 py-2 text-xs text-text-secondary leading-relaxed"
      >
        {props.text}
      </span>
    </span>
  );
}

export default function Labelling() {
  let canvas!: HTMLCanvasElement;
  let wrap!: HTMLDivElement;
  let fileInput!: HTMLInputElement;
  let ctx: CanvasRenderingContext2D | null = null;

  const [images, setImages]       = createSignal<LabImage[]>([]);
  const [idx, setIdx]             = createSignal(0);
  const [activeClass, setActive]  = createSignal<ClsName>('glass');
  const [selected, setSelected]   = createSignal(-1);
  const [tick, bump]              = createSignal(0, { equals: false }); // refresh manual de listas
  const [exporting, setExporting] = createSignal(false);
  const [dragOver, setDragOver]   = createSignal(false);

  const cur = () => images()[idx()];
  const refresh = () => bump(0);

  // ── Geometría imagen <-> canvas (letterbox fit) ──────────────────────────────
  let scale = 1, offX = 0, offY = 0;
  function computeFit() {
    const im = cur();
    if (!im || !canvas.width || !canvas.height) return;
    scale = Math.min(canvas.width / im.w, canvas.height / im.h);
    offX = (canvas.width - im.w * scale) / 2;
    offY = (canvas.height - im.h * scale) / 2;
  }
  const toCx = (x: number) => offX + x * scale;
  const toCy = (y: number) => offY + y * scale;
  const toIx = (cx: number) => (cx - offX) / scale;
  const toIy = (cy: number) => (cy - offY) / scale;

  function handlePts(b: Box): { id: HandleId; x: number; y: number }[] {
    const x1 = toCx(b.x), y1 = toCy(b.y), x2 = toCx(b.x + b.w), y2 = toCy(b.y + b.h);
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    return [
      { id: 'nw', x: x1, y: y1 }, { id: 'n', x: mx, y: y1 }, { id: 'ne', x: x2, y: y1 },
      { id: 'e', x: x2, y: my }, { id: 'se', x: x2, y: y2 }, { id: 's', x: mx, y: y2 },
      { id: 'sw', x: x1, y: y2 }, { id: 'w', x: x1, y: my },
    ];
  }

  // ── Redibujado imperativo ────────────────────────────────────────────────────
  function redraw() {
    if (!ctx || !canvas.width) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#0a0a0b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const im = cur();
    if (!im) return;
    computeFit();
    ctx.drawImage(im.img, offX, offY, im.w * scale, im.h * scale);

    ctx.font = '600 13px "Geist Variable", system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic';
    im.boxes.forEach((b, i) => {
      const color = WONG[b.cls];
      const x = toCx(b.x), y = toCy(b.y), w = b.w * scale, h = b.h * scale;
      ctx!.lineWidth = 2;
      ctx!.strokeStyle = color;
      ctx!.strokeRect(x, y, w, h);

      const label = CLASS_LABEL_ES[b.cls];
      const tw = ctx!.measureText(label).width + 10;
      const ly = Math.max(18, y);
      ctx!.fillStyle = color;
      ctx!.fillRect(x, ly - 18, tw, 18);
      ctx!.fillStyle = '#0b0b0c';
      ctx!.fillText(label, x + 5, ly - 5);

      if (i === selected()) {
        for (const p of handlePts(b)) {
          ctx!.fillStyle = '#ffffff';
          ctx!.fillRect(p.x - HANDLE / 2, p.y - HANDLE / 2, HANDLE, HANDLE);
          ctx!.lineWidth = 1;
          ctx!.strokeStyle = color;
          ctx!.strokeRect(p.x - HANDLE / 2, p.y - HANDLE / 2, HANDLE, HANDLE);
        }
      }
    });
  }

  // Redibuja ante cambios de estado reactivo (imagen actual, selección, ediciones).
  createEffect(() => { idx(); selected(); tick(); images(); redraw(); });

  // ── Interacción con el puntero ───────────────────────────────────────────────
  let mode: Mode = 'idle';
  let dragHandle: HandleId | null = null;
  let dragStart = { x: 0, y: 0 };       // coords de imagen
  let boxStart: Box | null = null;

  function canvasPos(e: PointerEvent) {
    const r = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - r.left) * (canvas.width / r.width),
      y: (e.clientY - r.top) * (canvas.height / r.height),
    };
  }
  function hitHandle(b: Box, cx: number, cy: number): HandleId | null {
    for (const p of handlePts(b)) {
      if (Math.abs(cx - p.x) <= HANDLE && Math.abs(cy - p.y) <= HANDLE) return p.id;
    }
    return null;
  }
  const inside = (b: Box, ix: number, iy: number) =>
    ix >= b.x && ix <= b.x + b.w && iy >= b.y && iy <= b.y + b.h;

  function onPointerDown(e: PointerEvent) {
    const im = cur();
    if (!im) return;
    canvas.setPointerCapture(e.pointerId);
    const c = canvasPos(e);
    const ix = toIx(c.x), iy = toIy(c.y);

    const sel = selected();
    if (sel >= 0 && im.boxes[sel]) {
      const hid = hitHandle(im.boxes[sel], c.x, c.y);
      if (hid) { mode = 'resize'; dragHandle = hid; boxStart = { ...im.boxes[sel] }; dragStart = { x: ix, y: iy }; return; }
    }
    for (let i = im.boxes.length - 1; i >= 0; i--) {
      if (inside(im.boxes[i], ix, iy)) {
        setSelected(i); mode = 'move'; boxStart = { ...im.boxes[i] }; dragStart = { x: ix, y: iy };
        return;
      }
    }
    // dibujar una caja nueva con la clase activa
    im.boxes.push({ x: ix, y: iy, w: 0, h: 0, cls: activeClass() });
    setSelected(im.boxes.length - 1);
    mode = 'draw'; dragStart = { x: ix, y: iy }; boxStart = null;
    redraw();
  }

  function onPointerMove(e: PointerEvent) {
    if (mode === 'idle') return;
    const im = cur();
    if (!im) return;
    const b = im.boxes[selected()];
    if (!b) return;
    const c = canvasPos(e);
    const ix = Math.max(0, Math.min(toIx(c.x), im.w));
    const iy = Math.max(0, Math.min(toIy(c.y), im.h));

    if (mode === 'draw') {
      b.x = Math.min(dragStart.x, ix); b.y = Math.min(dragStart.y, iy);
      b.w = Math.abs(ix - dragStart.x); b.h = Math.abs(iy - dragStart.y);
    } else if (mode === 'move' && boxStart) {
      b.x = Math.max(0, Math.min(boxStart.x + (ix - dragStart.x), im.w - b.w));
      b.y = Math.max(0, Math.min(boxStart.y + (iy - dragStart.y), im.h - b.h));
    } else if (mode === 'resize' && boxStart && dragHandle) {
      let x1 = boxStart.x, y1 = boxStart.y, x2 = boxStart.x + boxStart.w, y2 = boxStart.y + boxStart.h;
      if (dragHandle.includes('w')) x1 = ix;
      if (dragHandle.includes('e')) x2 = ix;
      if (dragHandle.includes('n')) y1 = iy;
      if (dragHandle.includes('s')) y2 = iy;
      b.x = Math.min(x1, x2); b.y = Math.min(y1, y2);
      b.w = Math.abs(x2 - x1); b.h = Math.abs(y2 - y1);
    }
    redraw();
  }

  function onPointerUp() {
    const im = cur();
    if (im && mode === 'draw') {
      const s = selected();
      const b = im.boxes[s];
      if (b && (b.w < MIN_BOX || b.h < MIN_BOX)) { im.boxes.splice(s, 1); setSelected(-1); }
    }
    mode = 'idle'; dragHandle = null; boxStart = null;
    refresh();
  }

  function deleteBox(i: number) {
    const im = cur();
    if (!im || !im.boxes[i]) return;
    im.boxes.splice(i, 1);
    setSelected(-1);
    refresh();
  }
  function assignClass(c: ClsName) {
    setActive(c);
    const im = cur();
    const s = selected();
    if (im && s >= 0 && im.boxes[s]) { im.boxes[s].cls = c; refresh(); }
  }

  function go(delta: number) {
    const n = images().length;
    if (!n) return;
    const ni = Math.min(n - 1, Math.max(0, idx() + delta));
    if (ni !== idx()) { setIdx(ni); setSelected(-1); }
  }
  function selectImage(i: number) { setIdx(i); setSelected(-1); }

  // ── Teclado: borrar, navegar, asignar clase ──────────────────────────────────
  function onKey(e: KeyboardEvent) {
    if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (selected() >= 0) { deleteBox(selected()); e.preventDefault(); }
    } else if (e.key === 'ArrowRight') { go(1); }
    else if (e.key === 'ArrowLeft') { go(-1); }
    else if (e.key === 'Escape') { setSelected(-1); }
    else if (e.key >= '1' && e.key <= '3') { assignClass(CLASSES[+e.key - 1]); }
  }

  // ── Carga de imágenes ────────────────────────────────────────────────────────
  function loadImage(url: string): Promise<HTMLImageElement> {
    return new Promise((res, rej) => {
      const im = new Image();
      im.onload = () => res(im);
      im.onerror = rej;
      im.src = url;
    });
  }
  async function addFiles(list: FileList | File[]) {
    const files = Array.from(list).filter((f) => f.type.startsWith('image/'));
    if (!files.length) return;
    const loaded: LabImage[] = [];
    for (const file of files) {
      const url = URL.createObjectURL(file);
      try {
        const img = await loadImage(url);
        loaded.push({ name: file.name, url, file, w: img.naturalWidth, h: img.naturalHeight, img, boxes: [] });
      } catch { URL.revokeObjectURL(url); }
    }
    if (!loaded.length) return;
    const wasEmpty = images().length === 0;
    setImages([...images(), ...loaded]);
    if (wasEmpty) setIdx(0);
    requestAnimationFrame(() => { resizeCanvas(); redraw(); });
  }
  function onFiles(e: Event) {
    const input = e.target as HTMLInputElement;
    void addFiles(input.files ?? []);
    input.value = '';                       // permite recargar los mismos nombres
  }

  // Arrastrar y soltar imágenes en cualquier parte del editor.
  let dragDepth = 0;
  function onDragEnter(e: DragEvent) {
    e.preventDefault(); dragDepth++; setDragOver(true);
  }
  function onDragOver(e: DragEvent) {
    e.preventDefault();                       // imprescindible para habilitar el drop
    if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  }
  function onDragLeave(e: DragEvent) {
    e.preventDefault(); dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) setDragOver(false);
  }
  function onDrop(e: DragEvent) {
    e.preventDefault(); dragDepth = 0; setDragOver(false);
    if (e.dataTransfer?.files?.length) void addFiles(e.dataTransfer.files);
  }

  // ── Export YOLO ──────────────────────────────────────────────────────────────
  const baseName = (n: string) => { const i = n.lastIndexOf('.'); return i > 0 ? n.slice(0, i) : n; };
  function yoloText(im: LabImage): string {
    return im.boxes.map((b) => {
      const cx = (b.x + b.w / 2) / im.w, cy = (b.y + b.h / 2) / im.h;
      return `${CLASS_ID[b.cls]} ${cx.toFixed(6)} ${cy.toFixed(6)} ${(b.w / im.w).toFixed(6)} ${(b.h / im.h).toFixed(6)}`;
    }).join('\n');
  }
  const dataYaml = () =>
    ['path: .', 'train: images', 'val: images', '', 'nc: 3', "names: ['glass', 'paper', 'plastic']", ''].join('\n');

  function download(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  }
  function exportCurrentTxt() {
    const im = cur();
    if (!im) return;
    download(new Blob([yoloText(im)], { type: 'text/plain' }), baseName(im.name) + '.txt');
  }
  async function exportZip() {
    const ims = images();
    if (!ims.length) return;
    setExporting(true);
    try {
      const zip = new JSZip();
      zip.file('data.yaml', dataYaml());
      const imgDir = zip.folder('images')!;
      const lblDir = zip.folder('labels')!;
      for (const im of ims) {
        lblDir.file(baseName(im.name) + '.txt', yoloText(im));
        imgDir.file(im.name, im.file);
      }
      download(await zip.generateAsync({ type: 'blob' }), 'tiny-trash-dataset.zip');
    } finally {
      setExporting(false);
    }
  }

  // ── Métricas derivadas (reactivas vía tick) ──────────────────────────────────
  const boxesView = () => { tick(); return [...(cur()?.boxes ?? [])]; };
  const totalBoxes = () => { tick(); return images().reduce((s, im) => s + im.boxes.length, 0); };
  const annotated  = () => { tick(); return images().filter((im) => im.boxes.length > 0).length; };

  function resizeCanvas() {
    if (!canvas || !wrap) return;
    const r = wrap.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(r.width));
    canvas.height = Math.max(1, Math.floor(r.height));
  }

  onMount(() => {
    ctx = canvas.getContext('2d');
    resizeCanvas();
    redraw();
    window.addEventListener('keydown', onKey);
    const ro = new ResizeObserver(() => { resizeCanvas(); redraw(); });
    ro.observe(wrap);
    onCleanup(() => {
      window.removeEventListener('keydown', onKey);
      ro.disconnect();
      images().forEach((im) => URL.revokeObjectURL(im.url));
    });
  });

  return (
    <div
      class="h-screen overflow-hidden bg-bg-app flex flex-col relative"
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <header class="flex items-center justify-between border-b border-border px-6 h-14 gap-4 shrink-0">
        <div class="flex items-center gap-3 min-w-0">
          <A href="/" aria-label="Volver al inicio"
            class="flex items-center text-text-secondary hover:text-text-primary transition-colors">
            <IconBack />
          </A>
          <span class="font-semibold text-text-primary">Labelling</span>
        </div>
        <div class="flex items-center gap-4">
          <Show when={images().length}>
            <span class="font-mono text-xs text-text-secondary tabular">
              {annotated()} / {images().length} anotadas · {totalBoxes()} cajas
            </span>
          </Show>
          <ThemeToggle />
        </div>
      </header>

      <div class="flex-1 flex min-h-0">
        {/* Lienzo + tira de thumbnails */}
        <section class="flex-1 min-w-0 flex flex-col">
          <div ref={wrap} class="flex-1 min-h-0 relative">
            <canvas
              ref={canvas}
              class="absolute inset-0 w-full h-full touch-none select-none"
              classList={{ 'cursor-crosshair': !!cur() }}
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
            />
            <Show when={!cur()}>
              <div class="absolute inset-0 flex flex-col items-center justify-center gap-4 text-center px-6 text-text-secondary">
                <IconUpload />
                <p class="max-w-xs text-sm">
                  Cargá imágenes del dataset (o arrastralas aquí) para anotarlas. Todo ocurre en tu navegador, sin subir nada.
                </p>
                <button class={BTN_PRIMARY} onClick={() => fileInput.click()}>Cargar imágenes</button>
              </div>
            </Show>
          </div>

          <Show when={images().length}>
            <div class="border-t border-border bg-bg-panel px-4 py-2 flex items-center gap-3 shrink-0">
              <button class={BTN} onClick={() => go(-1)} disabled={idx() === 0} aria-label="Anterior">‹</button>
              <span class="font-mono text-xs text-text-secondary tabular whitespace-nowrap">
                {idx() + 1} / {images().length}
              </span>
              <button class={BTN} onClick={() => go(1)} disabled={idx() >= images().length - 1} aria-label="Siguiente">›</button>
              <div class="flex-1 min-w-0 overflow-x-auto flex gap-2 py-1">
                <For each={images()}>
                  {(im, i) => (
                    <button
                      onClick={() => selectImage(i())}
                      class="relative shrink-0 rounded overflow-hidden border-2 transition-colors"
                      classList={{ 'border-accent': i() === idx(), 'border-transparent hover:border-border': i() !== idx() }}
                      aria-label={im.name}
                    >
                      <img src={im.url} class="h-12 w-16 object-cover block" alt="" />
                      <Show when={(tick(), im.boxes.length > 0)}>
                        <span class="absolute top-0 right-0 bg-bg-app/85 text-text-primary text-[10px] leading-none px-1 py-0.5 rounded-bl">
                          {(tick(), im.boxes.length)}
                        </span>
                      </Show>
                    </button>
                  )}
                </For>
              </div>
            </div>
          </Show>
        </section>

        {/* Panel de control */}
        <aside class="w-80 border-l border-border bg-bg-panel flex flex-col shrink-0 overflow-y-auto">
          <div class="border-b border-border px-4 py-4">
            <button class={`${BTN} w-full flex items-center justify-center gap-2`} onClick={() => fileInput.click()}>
              <IconUpload /> Añadir imágenes
            </button>
          </div>

          <div class="border-b border-border px-4 py-4">
            <h2 class="mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary">Clase activa</h2>
            <div class="flex flex-col gap-2">
              <For each={CLASSES}>
                {(c) => (
                  <button
                    onClick={() => assignClass(c)}
                    class="flex items-center gap-2.5 px-2.5 py-1.5 rounded-md border text-sm transition-colors"
                    classList={{
                      'border-accent text-text-primary bg-bg-surface': activeClass() === c,
                      'border-border text-text-secondary hover:border-accent': activeClass() !== c,
                    }}
                  >
                    <span class="w-3 h-3 rounded-sm shrink-0" style={{ 'background-color': WONG[c] }} />
                    {CLASS_LABEL_ES[c]}
                  </button>
                )}
              </For>
            </div>
            <p class="mt-3 text-xs text-text-secondary leading-relaxed">
              Arrastrá sobre la imagen para dibujar. Tecla 1/2/3 cambia de clase; Supr borra la caja seleccionada.
            </p>
          </div>

          <div class="flex-1 min-h-0 px-4 py-4">
            <h2 class="mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary">
              Cajas {cur() ? `(${boxesView().length})` : ''}
            </h2>
            <Show
              when={cur() && boxesView().length}
              fallback={<p class="text-xs text-text-secondary">{cur() ? 'Sin cajas en esta imagen.' : 'Cargá una imagen.'}</p>}
            >
              <div class="flex flex-col gap-1">
                <For each={boxesView()}>
                  {(b, i) => (
                    <div
                      onClick={() => setSelected(i())}
                      class="group flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer transition-colors"
                      classList={{ 'bg-bg-surface': i() === selected(), 'hover:bg-bg-surface/60': i() !== selected() }}
                    >
                      <span class="w-2.5 h-2.5 rounded-sm shrink-0" style={{ 'background-color': WONG[b.cls] }} />
                      <span class="flex-1 text-sm text-text-primary">{CLASS_LABEL_ES[b.cls]}</span>
                      <span class="font-mono text-[11px] text-text-secondary tabular">
                        {Math.round(b.w)}×{Math.round(b.h)}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); deleteBox(i()); }}
                        class="text-text-secondary hover:text-text-primary px-1 leading-none"
                        aria-label="Eliminar caja"
                      >
                        ×
                      </button>
                    </div>
                  )}
                </For>
              </div>
            </Show>
          </div>

          <div class="border-t border-border px-4 py-4 flex flex-col gap-2">
            <div class="flex items-center justify-between mb-1">
              <h2 class="text-[11px] font-medium uppercase tracking-wider text-text-secondary">Exportar</h2>
              <InfoTip text="Formato YOLO: class x_center y_center w h normalizado, con data.yaml (clases 0 vidrio, 1 papel, 2 plástico)." />
            </div>
            <button class={BTN_PRIMARY} onClick={exportZip} disabled={!images().length || exporting()}>
              {exporting() ? 'Empaquetando…' : 'Descargar dataset (.zip)'}
            </button>
            <button class={`${BTN} flex items-center justify-center gap-2`} onClick={exportCurrentTxt} disabled={!cur()}>
              <IconDownload /> Imagen actual (.txt)
            </button>
          </div>
        </aside>
      </div>

      <input ref={fileInput} type="file" accept="image/*" multiple class="hidden" onChange={onFiles} />

      {/* Overlay de arrastre: aparece mientras se arrastran archivos sobre la ventana */}
      <Show when={dragOver()}>
        <div class="absolute inset-0 z-40 flex items-center justify-center bg-bg-app/80 pointer-events-none">
          <div class="rounded-lg border-2 border-dashed border-accent px-8 py-6 text-sm font-medium text-text-primary">
            Soltá las imágenes para cargarlas
          </div>
        </div>
      </Show>
    </div>
  );
}
