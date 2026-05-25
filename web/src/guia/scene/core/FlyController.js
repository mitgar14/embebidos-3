import * as THREE from 'three';

/**
 * Cámara FPS / vuelo libre:
 *  - WASD: mover horizontal (relativo a la dirección de la cámara)
 *  - Q/E:  bajar / subir
 *  - Shift: x3 velocidad (boost)
 *  - Drag con botón izquierdo del ratón sobre el canvas: yaw (X) + pitch (Y)
 *  - Rueda del ratón: ajustar velocidad de movimiento
 *
 * No usa Pointer Lock (queda activo solo el drag-look) para no secuestrar
 * el cursor cuando el usuario interactúa con la UI HTML del overlay.
 */
export class FlyController {
  constructor(camera, domElement) {
    this.camera = camera;
    this.dom = domElement;
    this.enabled = false;

    // Estado de teclas
    this.keys = { w: false, a: false, s: false, d: false, q: false, e: false, shift: false };
    // Mouse look
    this._dragging = false;
    this._lastX = 0; this._lastY = 0;
    this._yaw = 0;   // alrededor de Y
    this._pitch = 0; // alrededor de X local

    // Parámetros
    this.moveSpeed = 60;       // unidades/seg base
    this.lookSpeed = 0.0025;   // rad por pixel
    this.boostFactor = 3;

    // Bind handlers
    this._onKeyDown   = this._onKeyDown.bind(this);
    this._onKeyUp     = this._onKeyUp.bind(this);
    this._onMouseDown = this._onMouseDown.bind(this);
    this._onMouseMove = this._onMouseMove.bind(this);
    this._onMouseUp   = this._onMouseUp.bind(this);
    this._onWheel     = this._onWheel.bind(this);
  }

  enable() {
    if (this.enabled) return;
    this.enabled = true;
    const euler = new THREE.Euler().setFromQuaternion(this.camera.quaternion, 'YXZ');
    this._yaw = euler.y;
    this._pitch = euler.x;

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup',   this._onKeyUp);
    this.dom.addEventListener('mousedown', this._onMouseDown);
    window.addEventListener('mousemove',   this._onMouseMove);
    window.addEventListener('mouseup',     this._onMouseUp);
    this.dom.addEventListener('wheel', this._onWheel, { passive: false });
  }

  disable() {
    if (!this.enabled) return;
    this.enabled = false;
    this._dragging = false;
    this.keys = { w:false, a:false, s:false, d:false, q:false, e:false, shift:false };

    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup',   this._onKeyUp);
    this.dom.removeEventListener('mousedown', this._onMouseDown);
    window.removeEventListener('mousemove',   this._onMouseMove);
    window.removeEventListener('mouseup',     this._onMouseUp);
    this.dom.removeEventListener('wheel', this._onWheel);
  }

  _onKeyDown(e) {
    if (e.target && e.target.tagName === 'INPUT') return;
    const k = e.key.toLowerCase();
    if (k in this.keys) { this.keys[k] = true; e.preventDefault(); }
    if (e.key === 'Shift') this.keys.shift = true;
  }
  _onKeyUp(e) {
    const k = e.key.toLowerCase();
    if (k in this.keys) this.keys[k] = false;
    if (e.key === 'Shift') this.keys.shift = false;
  }

  _onMouseDown(e) {
    if (e.button !== 0) return;
    this._dragging = true;
    this._lastX = e.clientX;
    this._lastY = e.clientY;
    this.dom.style.cursor = 'grabbing';
  }
  _onMouseMove(e) {
    if (!this._dragging) return;
    const dx = e.clientX - this._lastX;
    const dy = e.clientY - this._lastY;
    this._lastX = e.clientX;
    this._lastY = e.clientY;
    this._yaw   -= dx * this.lookSpeed;
    this._pitch -= dy * this.lookSpeed;
    const lim = Math.PI / 2 - 0.001;
    this._pitch = Math.max(-lim, Math.min(lim, this._pitch));
  }
  _onMouseUp() {
    this._dragging = false;
    this.dom.style.cursor = '';
  }
  _onWheel(e) {
    e.preventDefault();
    const delta = -Math.sign(e.deltaY);
    this.moveSpeed = Math.max(8, Math.min(400, this.moveSpeed * (1 + delta * 0.15)));
  }

  update(dt) {
    if (!this.enabled) return;

    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(this._pitch, this._yaw, 0, 'YXZ'));
    this.camera.quaternion.copy(q);

    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
    const right   = new THREE.Vector3(1, 0, 0).applyQuaternion(this.camera.quaternion);
    const up      = new THREE.Vector3(0, 1, 0);

    const speed = this.moveSpeed * (this.keys.shift ? this.boostFactor : 1) * dt;
    const move = new THREE.Vector3();
    if (this.keys.w) move.add(forward);
    if (this.keys.s) move.sub(forward);
    if (this.keys.d) move.add(right);
    if (this.keys.a) move.sub(right);
    if (this.keys.e) move.add(up);
    if (this.keys.q) move.sub(up);
    if (move.lengthSq() > 0) {
      move.normalize().multiplyScalar(speed);
      this.camera.position.add(move);
    }
  }
}
