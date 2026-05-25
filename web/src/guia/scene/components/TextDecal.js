import * as THREE from 'three';

/**
 * Genera un plano horizontal con texto serigráfico para "pegar" sobre el PCB.
 * Equivalente a la serigrafía blanca que ves en las placas reales.
 */
export function makeTextDecal(text, opts = {}) {
  const {
    width = 6, height = 2,
    color = '#ffffff',
    bg = null,
    fontSize = 56,
    fontWeight = 'bold',
    rotation = 0,    // rotación 2D en radianes (0 = horizontal, PI/2 = vertical)
    align = 'center'
  } = opts;

  const c = document.createElement('canvas');
  const aspect = width / height;
  c.width = 512;
  c.height = Math.round(512 / aspect);
  const ctx = c.getContext('2d');
  // Reescalar fontSize al alto del canvas para que sea consistente
  const fs = Math.round((fontSize / 64) * c.height);

  if (bg) {
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, c.width, c.height);
  }
  ctx.translate(c.width / 2, c.height / 2);
  if (rotation) ctx.rotate(rotation);
  ctx.fillStyle = color;
  ctx.font = `${fontWeight} ${fs}px 'Inter', system-ui, sans-serif`;
  ctx.textAlign = align;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 0, 0);

  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  tex.anisotropy = 4;
  tex.colorSpace = THREE.SRGBColorSpace;

  const mat = new THREE.MeshBasicMaterial({
    map: tex, transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide
  });
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(width, height), mat);
  plane.rotation.x = -Math.PI / 2;   // colocado plano sobre el PCB
  plane.renderOrder = 5;
  return plane;
}
