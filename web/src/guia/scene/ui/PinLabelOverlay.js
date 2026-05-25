import * as THREE from 'three';

/**
 * Overlay 2D para etiquetas de pines:
 *  - Cada pin tiene un <div> posicionado en pantalla
 *  - Una <line> SVG va del pin (proyectado) a la etiqueta
 *  - Los labels en un mismo lado se apilan auto-evitando solape
 *
 * Lados soportados: 'left' | 'right' | 'top' | 'bottom'
 */
export class PinLabelOverlay {
  constructor(canvas, camera) {
    this.canvas = canvas;
    this.camera = camera;
    this.container = document.getElementById('pin-labels-overlay');
    this.svg = document.getElementById('leader-lines');

    this.entries = [];
    this._v = new THREE.Vector3();
    this._showAll = false;

    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  _resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    this.svg.style.width = w + 'px';
    this.svg.style.height = h + 'px';
  }

  /**
   * @param {Pin|VirtualPin} pin   debe exponer .label, .type, opcional .sub, y .getAnchorWorld()
   * @param {{side: 'left'|'right'|'top'|'bottom', used: boolean}} opts
   */
  register(pin, opts = {}) {
    const side = opts.side || 'right';
    const used = opts.used !== false;

    const div = document.createElement('div');
    div.className = `pin-label pl-${side}` + (used ? '' : ' aux');
    div.dataset.type = pin.type || 'IO';
    div.innerHTML =
      `<span class="pl-main">${pin.label}</span>` +
      (pin.sub ? `<span class="pl-sub">${pin.sub}</span>` : '');
    this.container.appendChild(div);

    const ns = 'http://www.w3.org/2000/svg';
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('class', 'leader' + (used ? '' : ' aux'));
    line.dataset.type = pin.type || 'IO';
    this.svg.appendChild(line);

    const entry = { pin, div, line, side, used, hidden: !used && !this._showAll };
    this.entries.push(entry);
    this._applyVisibility(entry);
    return entry;
  }

  setShowAll(v) {
    this._showAll = v;
    this.entries.forEach(e => {
      e.hidden = !e.used && !v;
      this._applyVisibility(e);
    });
  }

  _applyVisibility(e) {
    e.div.style.display = e.hidden ? 'none' : '';
    e.line.style.display = e.hidden ? 'none' : '';
  }

  highlight(pin, on) {
    const e = this.entries.find(x => x.pin === pin);
    if (e) e.div.classList.toggle('hi', on);
  }

  clearHighlights() {
    this.entries.forEach(e => e.div.classList.remove('hi'));
  }

  /** Llamar cada frame en el render loop. */
  update() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;

    const buckets = { left: [], right: [], top: [], bottom: [] };

    for (const e of this.entries) {
      if (e.hidden) continue;
      const worldPos = e.pin.getAnchorWorld();
      this._v.copy(worldPos).project(this.camera);
      const inFront = this._v.z < 1;
      const sx = (this._v.x + 1) * 0.5 * w;
      const sy = (1 - this._v.y) * 0.5 * h;

      if (!inFront) {
        e.div.style.visibility = 'hidden';
        e.line.style.visibility = 'hidden';
        continue;
      }
      e.div.style.visibility = '';
      e.line.style.visibility = '';

      e.pinX = sx; e.pinY = sy;
      buckets[e.side].push(e);
    }

    this._layoutVertical(buckets.left,  'left',  w, h);
    this._layoutVertical(buckets.right, 'right', w, h);
    this._layoutHorizontal(buckets.top,    'top',    w, h);
    this._layoutHorizontal(buckets.bottom, 'bottom', w, h);
  }

  _layoutVertical(entries, side, w, h) {
    if (!entries.length) return;
    entries.sort((a, b) => a.pinY - b.pinY);

    const labelH = 24;
    const gap = 4;
    const margin = 14;
    const colX = side === 'left' ? margin : w - margin;

    for (const e of entries) {
      const r = e.div.getBoundingClientRect();
      e.labelW = r.width || 70;
      e.labelH = r.height || labelH;
      e.idealY = e.pinY - e.labelH / 2;
    }
    let last = -Infinity;
    for (const e of entries) {
      e.y = Math.max(e.idealY, last + gap);
      last = e.y + e.labelH;
    }
    const totalBottom = last;
    const overflow = totalBottom - (h - margin);
    if (overflow > 0) {
      for (let i = entries.length - 1; i >= 0; i--) {
        entries[i].y -= overflow;
      }
    }

    for (const e of entries) {
      const x = side === 'left' ? colX : colX - e.labelW;
      e.div.style.transform = `translate(${x}px, ${e.y}px)`;

      const lineEndX = side === 'left' ? x + e.labelW : x;
      const lineEndY = e.y + e.labelH / 2;
      e.line.setAttribute('x1', e.pinX);
      e.line.setAttribute('y1', e.pinY);
      e.line.setAttribute('x2', lineEndX);
      e.line.setAttribute('y2', lineEndY);
    }
  }

  _layoutHorizontal(entries, side, w, h) {
    if (!entries.length) return;
    entries.sort((a, b) => a.pinX - b.pinX);

    const gap = 6;
    const margin = 14;
    const rowY = side === 'top' ? margin : h - margin - 24;

    for (const e of entries) {
      const r = e.div.getBoundingClientRect();
      e.labelW = r.width || 50;
      e.labelH = r.height || 22;
      e.idealX = e.pinX - e.labelW / 2;
    }
    let last = -Infinity;
    for (const e of entries) {
      e.x = Math.max(e.idealX, last + gap);
      last = e.x + e.labelW;
    }
    const overflowR = last - (w - margin);
    if (overflowR > 0) {
      for (let i = entries.length - 1; i >= 0; i--) {
        entries[i].x -= overflowR;
      }
    }

    for (const e of entries) {
      const y = side === 'top' ? rowY : rowY;
      e.div.style.transform = `translate(${e.x}px, ${y}px)`;

      const lineEndX = e.x + e.labelW / 2;
      const lineEndY = side === 'top' ? y + e.labelH : y;
      e.line.setAttribute('x1', e.pinX);
      e.line.setAttribute('y1', e.pinY);
      e.line.setAttribute('x2', lineEndX);
      e.line.setAttribute('y2', lineEndY);
    }
  }
}

/**
 * Pin virtual: cualquier punto 3D que quiera tener una etiqueta sin ser un Pin metálico.
 * Útil para los 16 canales servo y bornes del terminal de tornillos.
 */
export class VirtualPin {
  constructor({ id, label, type, sub, parent, localPos }) {
    this.id = id;
    this.label = label;
    this.type = type;
    this.sub = sub;
    this._parent = parent;
    this._local = localPos.clone();
  }
  getAnchorWorld() {
    return this._parent.localToWorld(this._local.clone());
  }
}
