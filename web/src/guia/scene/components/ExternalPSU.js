import * as THREE from 'three';
import { Pin } from './Pin.js';

/** Fuente externa estilizada para alimentar V+ del PCA9685. */
export class ExternalPSU {
  constructor() {
    this.group = new THREE.Group();
    this.pins = {};

    const body = new THREE.Mesh(
      new THREE.BoxGeometry(22, 12, 14),
      new THREE.MeshStandardMaterial({ color: 0x222831, roughness: 0.7 })
    );
    body.position.y = 6;
    body.castShadow = true;
    this.group.add(body);

    // Pantalla LED roja
    const screen = new THREE.Mesh(
      new THREE.BoxGeometry(10, 4, 0.5),
      new THREE.MeshStandardMaterial({
        color: 0x1a0000, emissive: 0xff2222, emissiveIntensity: 0.6
      })
    );
    screen.position.set(0, 8, 7.1);
    this.group.add(screen);

    const label = new THREE.Mesh(
      new THREE.BoxGeometry(7, 0.6, 0.05),
      new THREE.MeshBasicMaterial({ color: 0xff5555 })
    );
    label.position.set(0, 8, 7.4);
    this.group.add(label);

    // Bornas + y -
    const plusPin = new Pin({
      id: 'PSU+', label: 'V+ 5V',
      type: 'VPLUS',
      localPosition: new THREE.Vector3(-6, 0.8, -7)
    });
    const minusPin = new Pin({
      id: 'PSU-', label: 'GND',
      type: 'GND',
      localPosition: new THREE.Vector3(6, 0.8, -7)
    });
    this.pins['PSU+'] = plusPin;
    this.pins['PSU-'] = minusPin;
    this.group.add(plusPin.group, minusPin.group);
  }

  getPin(id) { return this.pins[id]; }
}
