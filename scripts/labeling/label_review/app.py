"""Web UI local para refinar bboxes auto-generados (asistencia).

Flujo:
  1. Pre-pone bboxes de GroundingDINO con NMS+filtro area como ASISTENCIA.
  2. UI galeria single-image con prev/next.
  3. Por bbox: drag centro = mover, drag esquinas = redimensionar, click = seleccionar.
  4. Dibujar nueva bbox: drag sobre area vacia con la clase activa.
  5. Atajos: 1/2/3 cambia clase del seleccionado (o clase activa para nuevas),
     Del/Backspace elimina seleccionado, Esc deselecciona, flechas <- -> navegan.
  6. Auto-save al disco en cada cambio.
  7. Boton "Exportar YOLO" genera batch1-clean/.

Uso:
  uv run python scripts/labeling/label_review/app.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "datasets/waste-3class-batch1-auto/batch1"
STAGING = ROOT / "datasets/waste-3class-batch1-auto/staging.json"
FINAL = ROOT / "datasets/waste-3class-batch1-auto/batch1-clean"

CLASS_NAMES = ["plastic", "paper", "glass"]
MAX_BOX_AREA_RATIO = 0.35
NMS_IOU = 0.6


def iou_box(b1: tuple, b2: tuple) -> float:
    """IoU entre dos (cls, xc, yc, w, h) — ignora cls."""
    _, xc1, yc1, w1, h1 = b1
    _, xc2, yc2, w2, h2 = b2
    x1a, y1a, x2a, y2a = xc1 - w1 / 2, yc1 - h1 / 2, xc1 + w1 / 2, yc1 + h1 / 2
    x1b, y1b, x2b, y2b = xc2 - w2 / 2, yc2 - h2 / 2, xc2 + w2 / 2, yc2 + h2 / 2
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (x2a - x1a) * (y2a - y1a) + (x2b - x1b) * (y2b - y1b) - inter
    return inter / union if union > 0 else 0.0


def preprocess() -> dict:
    images = []
    for split in ["train", "valid"]:
        for img_path in sorted((DATASET / split / "images").glob("*.jpg")):
            label_path = DATASET / split / "labels" / (img_path.stem + ".txt")
            bboxes: list[dict] = []
            if label_path.exists() and label_path.stat().st_size > 0:
                raw = []
                for line in label_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    if w * h > MAX_BOX_AREA_RATIO:
                        continue
                    raw.append((cls, xc, yc, w, h))
                # NMS por area ASC, ignora clase
                order = sorted(range(len(raw)), key=lambda i: raw[i][3] * raw[i][4])
                kept: list[int] = []
                for i in order:
                    if all(iou_box(raw[i], raw[j]) < NMS_IOU for j in kept):
                        kept.append(i)
                bboxes = [
                    {"x": raw[i][1], "y": raw[i][2], "w": raw[i][3], "h": raw[i][4], "cls": raw[i][0]}
                    for i in kept
                ]
            images.append({
                "stem": img_path.stem,
                "split": split,
                "filename": img_path.name,
                "bboxes": bboxes,
            })
    return {"images": images, "class_names": CLASS_NAMES}


if not STAGING.exists():
    STAGING.write_text(json.dumps(preprocess(), indent=2))
state = json.loads(STAGING.read_text())
total = sum(len(im["bboxes"]) for im in state["images"])
print(f"[READY] {len(state['images'])} imagenes, {total} bboxes pre-cargadas como asistencia")

app = FastAPI(title="Embebidos-3 label review")
app.mount("/img", StaticFiles(directory=str(DATASET)), name="img")


class Bbox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    cls: int


class ImgBboxes(BaseModel):
    bboxes: list[Bbox]


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


@app.get("/api/state")
async def get_state():
    return state


@app.post("/api/image/{idx}")
async def save_image(idx: int, payload: ImgBboxes):
    if idx < 0 or idx >= len(state["images"]):
        raise HTTPException(404, "Image index out of range")
    state["images"][idx]["bboxes"] = [b.dict() for b in payload.bboxes]
    STAGING.write_text(json.dumps(state, indent=2))
    return {"ok": True, "saved": len(payload.bboxes)}


@app.post("/api/reset")
async def reset():
    global state
    STAGING.unlink(missing_ok=True)
    state = preprocess()
    STAGING.write_text(json.dumps(state, indent=2))
    return {"ok": True, "images": len(state["images"])}


@app.post("/api/export")
async def export():
    if FINAL.exists():
        shutil.rmtree(FINAL)
    for split in ["train", "valid"]:
        (FINAL / split / "images").mkdir(parents=True)
        (FINAL / split / "labels").mkdir(parents=True)

    stats = {"images": 0, "with_boxes": 0, "boxes": 0, "by_class": {0: 0, 1: 0, 2: 0}}
    for img in state["images"]:
        src = DATASET / img["split"] / "images" / img["filename"]
        dst = FINAL / img["split"] / "images" / img["filename"]
        shutil.copy2(src, dst)
        lines = []
        for b in img["bboxes"]:
            if not (0 <= b["cls"] < 3):
                continue
            lines.append(f"{b['cls']} {b['x']:.6f} {b['y']:.6f} {b['w']:.6f} {b['h']:.6f}")
            stats["by_class"][b["cls"]] += 1
        (FINAL / img["split"] / "labels" / (img["stem"] + ".txt")).write_text("\n".join(lines))
        stats["images"] += 1
        if lines:
            stats["with_boxes"] += 1
        stats["boxes"] += len(lines)

    (FINAL / "data.yaml").write_text(
        "names:\n- plastic\n- paper\n- glass\nnc: 3\ntrain: train/images\nval: valid/images\n"
    )
    return {"ok": True, "path": str(FINAL.relative_to(ROOT)), "stats": stats}


HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>embebidos-3 / label review</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; background: #1a1a1a;
         color: #eee; overflow: hidden; display: flex; flex-direction: column; }

  header { background: #2a2a2a; padding: 10px 16px; border-bottom: 1px solid #444;
           display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  header h1 { font-size: 14px; margin: 0; color: #aaa; font-weight: 500; }
  .nav { display: flex; align-items: center; gap: 6px; }
  .nav button { background: #444; color: white; border: none; padding: 6px 12px;
                cursor: pointer; border-radius: 4px; font-size: 13px; }
  .nav button:hover { background: #555; }
  .nav button:disabled { opacity: 0.4; cursor: not-allowed; }
  .counter { font-family: monospace; min-width: 64px; text-align: center; font-size: 13px; }
  .filename { font-family: monospace; color: #888; font-size: 12px; flex: 1; }

  .cls-buttons { display: flex; gap: 6px; }
  .cls-btn { padding: 6px 12px; cursor: pointer; border: 2px solid #555;
             background: #333; color: white; border-radius: 4px; font-size: 13px;
             min-width: 90px; text-align: center; }
  .cls-btn:hover { background: #3a3a3a; }
  .cls-btn.active { font-weight: bold; }
  .cls-btn.plastic.active { background: #ff4040; border-color: #ff4040; }
  .cls-btn.paper.active   { background: #40c840; border-color: #40c840; }
  .cls-btn.glass.active   { background: #0080ff; border-color: #0080ff; }
  .cls-btn kbd { font-size: 10px; opacity: 0.7; margin-left: 4px;
                 background: rgba(0,0,0,0.3); padding: 1px 4px; border-radius: 2px; }

  .export-btn { background: #4caf50; color: white; border: none; padding: 6px 14px;
                cursor: pointer; border-radius: 4px; font-weight: 600; font-size: 13px; }
  .export-btn:hover { background: #43a047; }
  .danger-btn { background: #c62828; color: white; border: none; padding: 6px 12px;
                cursor: pointer; border-radius: 4px; font-size: 12px; }

  main { flex: 1; display: flex; justify-content: center; align-items: center;
         padding: 16px; overflow: hidden; min-height: 0; }
  .canvas-wrapper { position: relative; max-width: 100%; max-height: 100%;
                    user-select: none; line-height: 0; }
  .canvas-wrapper img { max-width: 100%;
                        max-height: calc(100vh - 130px); display: block; pointer-events: none; }
  .overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%;
             cursor: crosshair; }

  .bbox { position: absolute; border: 2px solid; cursor: move;
          background: rgba(255,255,255,0.04); }
  .bbox.plastic { border-color: #ff4040; background: rgba(255,64,64,0.10); }
  .bbox.paper   { border-color: #40c840; background: rgba(64,200,64,0.10); }
  .bbox.glass   { border-color: #0080ff; background: rgba(0,128,255,0.10); }
  .bbox.selected { box-shadow: 0 0 0 2px white, 0 0 8px rgba(255,255,255,0.4); z-index: 5; }
  .bbox .label { position: absolute; top: -22px; left: -2px; padding: 1px 6px;
                 font-size: 11px; font-weight: 700; color: white; border-radius: 3px;
                 white-space: nowrap; pointer-events: none; }
  .bbox.plastic .label { background: #ff4040; }
  .bbox.paper   .label { background: #40c840; }
  .bbox.glass   .label { background: #0080ff; }

  .handle { position: absolute; width: 12px; height: 12px; background: white;
            border: 2px solid black; border-radius: 2px; }
  .handle.nw { top: -7px;    left: -7px;    cursor: nwse-resize; }
  .handle.ne { top: -7px;    right: -7px;   cursor: nesw-resize; }
  .handle.sw { bottom: -7px; left: -7px;    cursor: nesw-resize; }
  .handle.se { bottom: -7px; right: -7px;   cursor: nwse-resize; }

  .preview { position: absolute; border: 2px dashed yellow;
             background: rgba(255,255,0,0.1); pointer-events: none; }

  footer { background: #2a2a2a; padding: 8px 16px; border-top: 1px solid #444;
           display: flex; gap: 16px; align-items: center; font-size: 12px;
           flex-wrap: wrap; }
  .counts { font-family: monospace; }
  .progress-bar { flex: 1; height: 6px; background: #444; border-radius: 3px;
                  overflow: hidden; max-width: 280px; }
  .progress-bar > div { height: 100%;
                        background: linear-gradient(90deg, #4caf50, #8bc34a); transition: width .3s; }
  .help { color: #888; font-size: 11px; }
  .help kbd { background: #444; padding: 1px 5px; border-radius: 2px; color: #fff; font-size: 10px; }
  .saving { color: #ffc107; font-size: 11px; min-width: 60px; }
  .saving.ok { color: #4caf50; }
</style>
</head>
<body>

<header>
  <h1>embebidos-3 / label review</h1>
  <div class="nav">
    <button id="prev" title="Imagen anterior">◀</button>
    <span class="counter" id="counter">--/--</span>
    <button id="next" title="Imagen siguiente">▶</button>
  </div>
  <span class="filename" id="filename">cargando...</span>

  <div class="cls-buttons">
    <div class="cls-btn plastic active" data-cls="0">plastic<kbd>1</kbd></div>
    <div class="cls-btn paper"           data-cls="1">paper<kbd>2</kbd></div>
    <div class="cls-btn glass"           data-cls="2">glass<kbd>3</kbd></div>
  </div>

  <button class="export-btn" id="export">Exportar YOLO</button>
  <button class="danger-btn" id="reset" title="Vuelve a generar staging desde GroundingDINO">Reset</button>
</header>

<main>
  <div class="canvas-wrapper" id="wrapper">
    <img id="image" src="" alt="">
    <div class="overlay" id="overlay"></div>
    <div class="preview" id="preview" style="display:none; left:0; top:0; width:0; height:0;"></div>
  </div>
</main>

<footer>
  <span class="counts" id="counts">0 cajas</span>
  <div class="progress-bar"><div id="pbar" style="width:0"></div></div>
  <span class="saving" id="saving"></span>
  <span class="help">
    <kbd>drag</kbd> sobre vacio para dibujar &middot;
    <kbd>drag</kbd> centro mueve &middot;
    <kbd>drag</kbd> esquinas redimensiona &middot;
    <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> cambia clase del seleccionado &middot;
    <kbd>Del</kbd> elimina &middot;
    <kbd>←</kbd><kbd>→</kbd> navegar
  </span>
</footer>

<script>
const CLASS_NAMES = ['plastic', 'paper', 'glass'];
const MIN_BBOX_SIZE = 0.01;

let state = null;
let currentIdx = 0;
let activeClass = 0;
let selectedIdx = null;
let dragMode = null;      // 'draw' | 'move' | 'resize'
let dragData = null;
let saveTimer = null;

const wrapper = document.getElementById('wrapper');
const image = document.getElementById('image');
const overlay = document.getElementById('overlay');
const counter = document.getElementById('counter');
const filenameEl = document.getElementById('filename');
const counts = document.getElementById('counts');
const pbar = document.getElementById('pbar');
const savingEl = document.getElementById('saving');

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

async function load() {
  state = await (await fetch('/api/state')).json();
  showImage(0);
}

function showImage(idx) {
  if (idx < 0 || idx >= state.images.length) return;
  currentIdx = idx;
  selectedIdx = null;
  const im = state.images[idx];
  image.src = `/img/${im.split}/images/${im.filename}`;
  counter.textContent = `${idx + 1}/${state.images.length}`;
  filenameEl.textContent = `[${im.split}] ${im.stem}`;
  document.getElementById('prev').disabled = idx === 0;
  document.getElementById('next').disabled = idx === state.images.length - 1;
  renderBboxes();
  updateCounts();
}

function renderBboxes() {
  overlay.innerHTML = '';
  const boxes = state.images[currentIdx].bboxes;
  boxes.forEach((b, i) => {
    const div = document.createElement('div');
    div.className = `bbox ${CLASS_NAMES[b.cls]}${i === selectedIdx ? ' selected' : ''}`;
    div.dataset.idx = i;
    div.style.left   = ((b.x - b.w / 2) * 100) + '%';
    div.style.top    = ((b.y - b.h / 2) * 100) + '%';
    div.style.width  = (b.w * 100) + '%';
    div.style.height = (b.h * 100) + '%';
    const label = document.createElement('span');
    label.className = 'label';
    label.textContent = `${i + 1}. ${CLASS_NAMES[b.cls]}`;
    div.appendChild(label);
    if (i === selectedIdx) {
      ['nw', 'ne', 'sw', 'se'].forEach(h => {
        const handle = document.createElement('div');
        handle.className = `handle ${h}`;
        handle.dataset.handle = h;
        div.appendChild(handle);
      });
    }
    overlay.appendChild(div);
  });
}

function updateCounts() {
  const im = state.images[currentIdx];
  const c = {0: 0, 1: 0, 2: 0};
  im.bboxes.forEach(b => { if (c[b.cls] !== undefined) c[b.cls]++; });
  counts.textContent = `Imagen: ${im.bboxes.length} cajas (P:${c[0]} Pa:${c[1]} G:${c[2]})`;
  const withBoxes = state.images.filter(im => im.bboxes.length > 0).length;
  pbar.style.width = (withBoxes / state.images.length * 100) + '%';
}

function eventToNorm(e) {
  const r = wrapper.getBoundingClientRect();
  return {
    x: clamp((e.clientX - r.left) / r.width, 0, 1),
    y: clamp((e.clientY - r.top) / r.height, 0, 1),
  };
}

overlay.addEventListener('mousedown', (e) => {
  const pos = eventToNorm(e);
  if (e.target.classList.contains('handle')) {
    const boxEl = e.target.closest('.bbox');
    const idx = parseInt(boxEl.dataset.idx);
    selectedIdx = idx;
    dragMode = 'resize';
    dragData = {
      handle: e.target.dataset.handle,
      idx,
      original: { ...state.images[currentIdx].bboxes[idx] },
    };
    renderBboxes();
  } else if (e.target.classList.contains('bbox') || e.target.closest('.bbox')) {
    const boxEl = e.target.closest('.bbox');
    const idx = parseInt(boxEl.dataset.idx);
    selectedIdx = idx;
    dragMode = 'move';
    dragData = {
      startX: pos.x, startY: pos.y,
      idx,
      original: { ...state.images[currentIdx].bboxes[idx] },
    };
    renderBboxes();
  } else {
    // Modo DRAW: el preview es un elemento permanente FUERA del overlay,
    // asi renderBboxes() nunca puede borrarlo accidentalmente.
    selectedIdx = null;
    dragMode = 'draw';
    dragData = { startX: pos.x, startY: pos.y };
    const preview = document.getElementById('preview');
    preview.style.display = 'block';
    preview.style.left = (pos.x * 100) + '%';
    preview.style.top = (pos.y * 100) + '%';
    preview.style.width = '0%';
    preview.style.height = '0%';
    renderBboxes();
  }
  e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
  if (!dragMode) return;
  const pos = eventToNorm(e);
  const boxes = state.images[currentIdx].bboxes;

  if (dragMode === 'draw') {
    const preview = document.getElementById('preview');
    const left = Math.min(dragData.startX, pos.x);
    const top = Math.min(dragData.startY, pos.y);
    const w = Math.abs(pos.x - dragData.startX);
    const h = Math.abs(pos.y - dragData.startY);
    preview.style.left = (left * 100) + '%';
    preview.style.top = (top * 100) + '%';
    preview.style.width = (w * 100) + '%';
    preview.style.height = (h * 100) + '%';
  } else if (dragMode === 'move') {
    const dx = pos.x - dragData.startX;
    const dy = pos.y - dragData.startY;
    const o = dragData.original;
    const b = boxes[dragData.idx];
    b.x = clamp(o.x + dx, o.w / 2, 1 - o.w / 2);
    b.y = clamp(o.y + dy, o.h / 2, 1 - o.h / 2);
    renderBboxes();
  } else if (dragMode === 'resize') {
    const o = dragData.original;
    let x1 = o.x - o.w / 2, y1 = o.y - o.h / 2;
    let x2 = o.x + o.w / 2, y2 = o.y + o.h / 2;
    if (dragData.handle.includes('w')) x1 = pos.x;
    if (dragData.handle.includes('e')) x2 = pos.x;
    if (dragData.handle.includes('n')) y1 = pos.y;
    if (dragData.handle.includes('s')) y2 = pos.y;
    if (x1 > x2) [x1, x2] = [x2, x1];
    if (y1 > y2) [y1, y2] = [y2, y1];
    const b = boxes[dragData.idx];
    b.x = (x1 + x2) / 2;
    b.y = (y1 + y2) / 2;
    b.w = Math.max(MIN_BBOX_SIZE, x2 - x1);
    b.h = Math.max(MIN_BBOX_SIZE, y2 - y1);
    renderBboxes();
  }
});

document.addEventListener('mouseup', (e) => {
  if (!dragMode) return;
  const pos = eventToNorm(e);
  if (dragMode === 'draw') {
    document.getElementById('preview').style.display = 'none';
    const w = Math.abs(pos.x - dragData.startX);
    const h = Math.abs(pos.y - dragData.startY);
    if (w > MIN_BBOX_SIZE && h > MIN_BBOX_SIZE) {
      const xc = (dragData.startX + pos.x) / 2;
      const yc = (dragData.startY + pos.y) / 2;
      state.images[currentIdx].bboxes.push({ x: xc, y: yc, w, h, cls: activeClass });
      selectedIdx = state.images[currentIdx].bboxes.length - 1;
    }
  }
  dragMode = null;
  dragData = null;
  scheduleSave();
  renderBboxes();
  updateCounts();
});

function scheduleSave() {
  savingEl.textContent = 'guardando...';
  savingEl.classList.remove('ok');
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    await fetch(`/api/image/${currentIdx}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ bboxes: state.images[currentIdx].bboxes }),
    });
    savingEl.textContent = 'guardado';
    savingEl.classList.add('ok');
    setTimeout(() => { savingEl.textContent = ''; }, 1500);
  }, 200);
}

function setActiveClass(cls) {
  activeClass = cls;
  document.querySelectorAll('.cls-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.cls) === cls);
  });
  if (selectedIdx !== null) {
    state.images[currentIdx].bboxes[selectedIdx].cls = cls;
    renderBboxes();
    updateCounts();
    scheduleSave();
  }
}

document.querySelectorAll('.cls-btn').forEach(btn => {
  btn.addEventListener('click', () => setActiveClass(parseInt(btn.dataset.cls)));
});

document.getElementById('prev').addEventListener('click', () => showImage(currentIdx - 1));
document.getElementById('next').addEventListener('click', () => showImage(currentIdx + 1));

document.getElementById('export').addEventListener('click', async () => {
  const res = await fetch('/api/export', {method: 'POST'});
  const data = await res.json();
  const s = data.stats;
  alert(`Exportado a ${data.path}\\n\\n` +
        `Imagenes: ${s.images}\\n` +
        `Con bboxes: ${s.with_boxes}\\n` +
        `Total bboxes: ${s.boxes}\\n` +
        `  plastic: ${s.by_class[0]}\\n` +
        `  paper: ${s.by_class[1]}\\n` +
        `  glass: ${s.by_class[2]}`);
});

document.getElementById('reset').addEventListener('click', async () => {
  if (!confirm('Reset descarta TODO el progreso y vuelve a generar staging desde GroundingDINO. Confirmar?')) return;
  await fetch('/api/reset', {method: 'POST'});
  await load();
});

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft')  { showImage(currentIdx - 1); e.preventDefault(); }
  else if (e.key === 'ArrowRight') { showImage(currentIdx + 1); e.preventDefault(); }
  else if (e.key === '1') setActiveClass(0);
  else if (e.key === '2') setActiveClass(1);
  else if (e.key === '3') setActiveClass(2);
  else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (selectedIdx !== null) {
      state.images[currentIdx].bboxes.splice(selectedIdx, 1);
      selectedIdx = null;
      renderBboxes();
      updateCounts();
      scheduleSave();
      e.preventDefault();
    }
  } else if (e.key === 'Escape') {
    if (dragMode === 'draw') {
      document.getElementById('preview').style.display = 'none';
    }
    dragMode = null; dragData = null;
    selectedIdx = null;
    renderBboxes();
  }
});

load();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("\n  Abre http://127.0.0.1:8000 en tu navegador\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
