import * as THREE from 'three';
import { Pin } from './Pin.js';
import { ESP32_PINS } from '../../connections.js';

/**
 * Modelo 3D estilizado del ESP32-WROOM-32 devkit V1 (30 pines).
 * USB en +Z, antena PCB en -Z.
 * 15 pines por lado, espaciado 2.54 mm (unidades arbitrarias).
 */
const PIN_SPACING = 2.54;
const PIN_ROWS = 15;
const PIN_BLOCK_LEN = (PIN_ROWS - 1) * PIN_SPACING;     // 35.56
const PCB_W = 28;
const PCB_L = 55;

export class ESP32 {
  constructor() {
    this.group = new THREE.Group();
    this.pins = {};

    this._buildPCB();
    this._buildModule();
    this._buildUSB();
    this._buildButtons();
    this._buildLEDs();
    this._buildPins();
  }

  _buildPCB() {
    const pcb = new THREE.Mesh(
      new THREE.BoxGeometry(PCB_W, 1.6, PCB_L),
      new THREE.MeshStandardMaterial({
        color: 0x0a1f3a, roughness: 0.55, metalness: 0.1
      })
    );
    pcb.castShadow = true;
    pcb.receiveShadow = true;
    this.group.add(pcb);

    // Tira blanca de serigrafía bajo el módulo
    const silk = new THREE.Mesh(
      new THREE.BoxGeometry(PCB_W - 2, 0.05, PCB_L - 2),
      new THREE.MeshBasicMaterial({ color: 0x16315c, transparent: true, opacity: 0.35 })
    );
    silk.position.y = 0.82;
    this.group.add(silk);
  }

  _buildModule() {
    // Lata metálica del módulo WROOM-32
    const can = new THREE.Mesh(
      new THREE.BoxGeometry(18, 3, 25),
      new THREE.MeshStandardMaterial({
        color: 0xb8b8b8, metalness: 0.9, roughness: 0.35
      })
    );
    can.position.set(0, 2.3, -8);
    can.castShadow = true;
    this.group.add(can);

    // Texto "ESP32" simulado (línea blanca grabada)
    const stamp = new THREE.Mesh(
      new THREE.BoxGeometry(8, 0.04, 1.4),
      new THREE.MeshBasicMaterial({ color: 0xeeeeee })
    );
    stamp.position.set(0, 3.85, -8);
    this.group.add(stamp);

    // Antena PCB en el extremo opuesto al USB
    const antenna = new THREE.Mesh(
      new THREE.BoxGeometry(14, 0.08, 6),
      new THREE.MeshStandardMaterial({ color: 0xd9c47a, metalness: 0.9 })
    );
    antenna.position.set(0, 0.86, -23);
    this.group.add(antenna);
  }

  _buildUSB() {
    const usb = new THREE.Mesh(
      new THREE.BoxGeometry(8, 3, 6),
      new THREE.MeshStandardMaterial({
        color: 0xc0c0c0, metalness: 0.95, roughness: 0.2
      })
    );
    usb.position.set(0, 2.3, 26);
    this.group.add(usb);
  }

  _buildButtons() {
    const btnMat = new THREE.MeshStandardMaterial({ color: 0x222222 });
    const boot = new THREE.Mesh(new THREE.BoxGeometry(2.5, 1, 2.5), btnMat);
    boot.position.set(-10, 1.3, 18);
    const rst = boot.clone();
    rst.position.x = 10;
    this.group.add(boot, rst);
  }

  _buildLEDs() {
    const ledMat = new THREE.MeshStandardMaterial({
      color: 0xff4444, emissive: 0xaa1111, emissiveIntensity: 0.6
    });
    const pwrLed = new THREE.Mesh(new THREE.CircleGeometry(0.5, 12), ledMat);
    pwrLed.rotation.x = -Math.PI / 2;
    pwrLed.position.set(8, 0.86, 16);
    this.group.add(pwrLed);
  }

  _buildPins() {
    const startZ = -PIN_BLOCK_LEN / 2;
    for (const [id, def] of Object.entries(ESP32_PINS)) {
      const x = def.side === 'left' ? -12.5 : 12.5;
      const z = startZ + def.row * PIN_SPACING;
      const pin = new Pin({
        id,
        label: def.label,
        sub: def.sub,
        type: def.type,
        localPosition: new THREE.Vector3(x, 0.8, z)
      });
      this.pins[id] = pin;
      this.group.add(pin.group);
    }
  }

  getPin(id) { return this.pins[id]; }
}
