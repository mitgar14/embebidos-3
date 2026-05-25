import * as THREE from 'three';

/**
 * Raycaster sobre los meshes de pin. Emite eventos onHover/onLeave/onClick.
 * Le pasas un array de objetos {pin, mesh} con cada pin del mundo.
 */
export class InteractionManager {
  constructor(canvas, camera) {
    this.canvas = canvas;
    this.camera = camera;
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2(-10, -10);
    this.targets = [];
    this.currentHover = null;

    this.handlers = { hover: null, leave: null, click: null };

    canvas.addEventListener('pointermove', e => this._onMove(e));
    canvas.addEventListener('pointerleave', () => {
      this.pointer.set(-10, -10);
      this._emitLeave();
    });
    canvas.addEventListener('click', e => this._onClick(e));
  }

  registerPin(pin) {
    this.targets.push(pin.metalMesh, pin.socketMesh);
  }

  on(event, handler) { this.handlers[event] = handler; }

  _onMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  _onClick(e) {
    const pin = this._raycast();
    if (pin && this.handlers.click) this.handlers.click(pin, e);
  }

  _raycast() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects(this.targets, false);
    return hits.length ? hits[0].object.userData.pin : null;
  }

  _emitLeave() {
    if (this.currentHover && this.handlers.leave) {
      this.handlers.leave(this.currentHover);
    }
    this.currentHover = null;
  }

  update() {
    const hit = this._raycast();
    if (hit !== this.currentHover) {
      if (this.currentHover && this.handlers.leave) {
        this.handlers.leave(this.currentHover);
      }
      this.currentHover = hit;
      if (hit && this.handlers.hover) this.handlers.hover(hit);
      this.canvas.style.cursor = hit ? 'pointer' : 'grab';
    }
  }
}
