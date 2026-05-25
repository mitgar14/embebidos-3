import * as THREE from 'three';
import { Wire } from './Wire.js';

/**
 * Cable trifilar entre un servo y un canal del PCA9685.
 * Genera 3 Wires (señal/V+/GND) en colores estándar:
 *   blanco/naranja = señal, rojo = V+, marrón = GND.
 */
export class ServoCable {
  constructor(servo, pcaChannelAnchors) {
    this.servo = servo;
    const exit = servo.getCableExitWorld();
    // Tres puntos de salida cercanos al cuerpo del servo
    const sigStart = exit.clone().add(new THREE.Vector3(0, 1.2, 0));
    const vStart   = exit.clone();
    const gStart   = exit.clone().add(new THREE.Vector3(0, -1.2, 0));

    const sigEnd = pcaChannelAnchors.signal;
    const vEnd   = pcaChannelAnchors.vplus;
    const gEnd   = pcaChannelAnchors.gnd;

    this.signal = new Wire(sigStart, sigEnd, 0xe0a020, { radius: 0.28, arc: 8 });
    this.vplus  = new Wire(vStart,   vEnd,   0xb71c1c, { radius: 0.28, arc: 8 });
    this.gnd    = new Wire(gStart,   gEnd,   0x222222, { radius: 0.28, arc: 8 });
  }

  addTo(scene) {
    scene.add(this.signal.mesh);
    scene.add(this.vplus.mesh);
    scene.add(this.gnd.mesh);
  }

  show(instant = false) {
    this.signal.show(instant);
    this.vplus.show(instant);
    this.gnd.show(instant);
  }

  hide(instant = false) {
    this.signal.hide(instant);
    this.vplus.hide(instant);
    this.gnd.hide(instant);
  }

  update(dt) {
    this.signal.update(dt);
    this.vplus.update(dt);
    this.gnd.update(dt);
  }
}
