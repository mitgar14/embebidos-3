import * as THREE from 'three';

/**
 * Etiqueta flotante para un pin: sprite con canvas texture.
 * Siempre mira a la cámara, escala fija en world units.
 *
 * Tipos reconocidos para color-coding:
 *   GND, VCC/POWER, SDA, SCL, IO, IN, ADC, UART, EN, OE, VPLUS
 */
const TYPE_THEME = {
  GND:    { fg: '#e6edf7', border: '#444' },
  VCC:    { fg: '#ffd6d6', border: '#d62828' },
  POWER:  { fg: '#ffd6d6', border: '#d62828' },
  VPLUS:  { fg: '#ffb3b3', border: '#b71c1c' },
  SDA:    { fg: '#cfe3ff', border: '#0077ff' },
  SCL:    { fg: '#fff3b0', border: '#f1c40f' },
  IO:     { fg: '#d6dceb', border: '#3a4a6e' },
  IN:     { fg: '#d6dceb', border: '#5a6f99' },
  ADC:    { fg: '#c8f7e7', border: '#1abc9c' },
  UART:   { fg: '#e8d6ff', border: '#8e44ad' },
  EN:     { fg: '#ffd6f1', border: '#c2185b' },
  OE:     { fg: '#ffffff', border: '#888' },
  CH:     { fg: '#dbe7ff', border: '#2d3f6d' }
};

export function createPinLabel(text, type = 'IO', opts = {}) {
  const theme = TYPE_THEME[type] || TYPE_THEME.IO;
  const fontSize = opts.fontSize ?? 44;
  const padding  = opts.padding  ?? 10;
  const subText  = opts.sub;          // ej: "GPIO21" debajo del label principal

  // 1. Medir
  const m = document.createElement('canvas').getContext('2d');
  m.font = `bold ${fontSize}px 'Inter', system-ui, sans-serif`;
  const wMain = m.measureText(text).width;
  let wSub = 0;
  if (subText) {
    m.font = `${Math.floor(fontSize * 0.55)}px 'Inter', system-ui, sans-serif`;
    wSub = m.measureText(subText).width;
  }
  const wText = Math.max(wMain, wSub);
  const h = subText ? fontSize + fontSize * 0.6 + padding * 2 + 4 : fontSize + padding * 2;
  const w = Math.ceil(wText + padding * 2 + 8);

  // 2. Dibujar
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');

  // fondo redondeado
  roundRect(ctx, 0, 0, w, h, 10);
  ctx.fillStyle = 'rgba(11,18,32,0.92)';
  ctx.fill();
  ctx.strokeStyle = theme.border;
  ctx.lineWidth = 3;
  ctx.stroke();

  // texto principal
  ctx.fillStyle = theme.fg;
  ctx.font = `bold ${fontSize}px 'Inter', system-ui, sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  const mainY = subText ? padding + fontSize * 0.55 : h / 2;
  ctx.fillText(text, w / 2, mainY);

  // sub-texto opcional (GPIO number, función)
  if (subText) {
    ctx.fillStyle = 'rgba(154,167,194,0.95)';
    ctx.font = `${Math.floor(fontSize * 0.55)}px 'Inter', system-ui, sans-serif`;
    ctx.fillText(subText, w / 2, mainY + fontSize * 0.85);
  }

  // 3. Sprite
  const tex = new THREE.CanvasTexture(canvas);
  tex.minFilter = THREE.LinearFilter;
  tex.anisotropy = 4;
  tex.colorSpace = THREE.SRGBColorSpace;

  const mat = new THREE.SpriteMaterial({
    map: tex,
    transparent: true,
    depthTest: true,
    depthWrite: false
  });
  const sprite = new THREE.Sprite(mat);

  const scale = opts.scale ?? 0.07;
  sprite.scale.set(canvas.width * scale, canvas.height * scale, 1);
  sprite.userData.isPinLabel = true;
  sprite.renderOrder = 10;
  return sprite;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
