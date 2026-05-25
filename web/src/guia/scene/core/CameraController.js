import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import * as THREE from 'three';
import { FlyController } from './FlyController.js';

/**
 * Gestor de cámara con DOS modos:
 *  - 'orbit' (default): OrbitControls relajado, sin tope superior/inferior,
 *    distancia min/max amplias, pan en espacio de pantalla.
 *  - 'fly':   FlyController FPS con WASD + drag.
 *
 * Expone .flyTo(target, position) y .reset() que solo aplican en modo orbit.
 * Cuando se activa 'fly' se desactiva OrbitControls y viceversa.
 */
export class CameraController {
  constructor(camera, domElement) {
    this.camera = camera;
    this.dom = domElement;

    // Modo orbit (default), sin restricciones de polar
    this.orbit = new OrbitControls(camera, domElement);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = 0.08;
    this.orbit.minDistance = 5;
    this.orbit.maxDistance = 1500;
    this.orbit.minPolarAngle = 0;
    this.orbit.maxPolarAngle = Math.PI;  // poder mirar desde abajo
    this.orbit.screenSpacePanning = true;
    this.orbit.panSpeed = 1.2;
    this.orbit.zoomSpeed = 1.1;
    this.orbit.target.set(5, 5, 15);

    // Modo vuelo libre
    this.fly = new FlyController(camera, domElement);

    this.mode = 'orbit';
    this._homePos = camera.position.clone();
    this._homeTarget = this.orbit.target.clone();
    this._tween = null;
  }

  setMode(mode) {
    if (mode === this.mode) return;
    if (mode === 'fly') {
      this.orbit.enabled = false;
      this.fly.enable();
    } else {
      this.fly.disable();
      this.orbit.enabled = true;
      // Reconstruir el target del orbit a partir de la posición/orientación
      // actuales para que la cámara no salte al volver al modo órbita.
      const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
      this.orbit.target.copy(this.camera.position).addScaledVector(forward, 60);
    }
    this.mode = mode;
  }
  toggle() { this.setMode(this.mode === 'fly' ? 'orbit' : 'fly'); }

  reset() {
    if (this.mode === 'fly') this.setMode('orbit');
    this.flyTo(this._homeTarget, this._homePos, 0.9);
  }

  /** Tween cinemático (solo en modo orbit). */
  flyTo(target, position, duration = 0.8) {
    if (this.mode !== 'orbit') return;
    this._tween = {
      t: 0, duration,
      fromPos: this.camera.position.clone(),
      toPos: position.clone(),
      fromTarget: this.orbit.target.clone(),
      toTarget: target.clone()
    };
  }

  update(dt) {
    if (this._tween && this.mode === 'orbit') {
      const t = this._tween;
      t.t = Math.min(t.duration, t.t + dt);
      const k = this._easeInOut(t.t / t.duration);
      this.camera.position.lerpVectors(t.fromPos, t.toPos, k);
      this.orbit.target.lerpVectors(t.fromTarget, t.toTarget, k);
      if (t.t >= t.duration) this._tween = null;
    }
    if (this.mode === 'orbit') this.orbit.update();
    else                       this.fly.update(dt);
  }

  dispose() {
    this.orbit.dispose();
    this.fly.disable();
  }

  // Compatibilidad con el código antiguo que llamaba `this.controls.target`
  get controls() { return this.orbit; }

  _easeInOut(x) {
    return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
  }
}
