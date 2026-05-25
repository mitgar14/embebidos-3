import * as THREE from 'three';

/**
 * Cable curvo entre dos puntos del mundo.
 * Usa CatmullRom con un punto medio elevado para dar curvatura natural.
 */
export class Wire {
  /**
   * @param {THREE.Vector3} from
   * @param {THREE.Vector3} to
   * @param {number} colorHex
   * @param {{ radius?: number, arc?: number }} opts
   */
  constructor(from, to, colorHex, opts = {}) {
    const radius = opts.radius ?? 0.45;
    const arc = opts.arc ?? 14;
    this.from = from.clone();
    this.to = to.clone();
    this.colorHex = colorHex;

    const curve = this._buildCurve(this.from, this.to, arc);
    const geometry = new THREE.TubeGeometry(curve, 64, radius, 12, false);
    const material = new THREE.MeshStandardMaterial({
      color: colorHex,
      roughness: 0.4,
      metalness: 0.2,
      transparent: true,
      opacity: 0
    });
    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.castShadow = true;
    this.curve = curve;

    // Animación: comienza invisible
    this._progress = 0;
    this._targetProgress = 0;
  }

  _buildCurve(a, b, arc) {
    const mid = a.clone().lerp(b, 0.5);
    const dist = a.distanceTo(b);
    mid.y += Math.max(arc, dist * 0.25);
    // Pequeño wobble lateral según diferencia X
    const wobble = (b.x - a.x) * 0.05;
    mid.z += wobble;
    return new THREE.CatmullRomCurve3([a, mid, b]);
  }

  show(instant = false) {
    this._targetProgress = 1;
    if (instant) {
      this._progress = 1;
      this.mesh.material.opacity = 1;
    }
  }

  hide(instant = false) {
    this._targetProgress = 0;
    if (instant) {
      this._progress = 0;
      this.mesh.material.opacity = 0;
    }
  }

  update(dt) {
    const speed = 4; // unidades por segundo
    if (this._progress < this._targetProgress) {
      this._progress = Math.min(this._targetProgress, this._progress + dt * speed);
    } else if (this._progress > this._targetProgress) {
      this._progress = Math.max(this._targetProgress, this._progress - dt * speed);
    }
    this.mesh.material.opacity = this._progress;
  }

  dispose() {
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
  }
}
