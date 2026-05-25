import * as THREE from 'three';
import { Pin } from './Pin.js';
import { PCA_PINS } from '../../connections.js';
import { VirtualPin } from '../ui/PinLabelOverlay.js';

/**
 * Breakout Adafruit-like PCA9685.
 * Headers, IC, cristal, terminal de tornillos y LEDs.
 * Las etiquetas se gestionan en el overlay 2D (no sprites 3D).
 */
const PCB_W = 62;
const PCB_L = 28;

export class PCA9685 {
  constructor() {
    this.group = new THREE.Group();
    this.pins = {};
    this.channels = [];   // VirtualPins de CH0..CH15
    this.terminals = [];  // VirtualPins de V+ / GND del terminal de tornillos

    this._buildPCB();
    this._buildIC();
    this._buildServoHeaders();
    this._buildScrewTerminal();
    this._buildLogicPins();
  }

  _buildPCB() {
    const pcb = new THREE.Mesh(
      new THREE.BoxGeometry(PCB_W, 1.6, PCB_L),
      new THREE.MeshStandardMaterial({
        color: 0x0d4d2a, roughness: 0.55, metalness: 0.1
      })
    );
    pcb.castShadow = true;
    pcb.receiveShadow = true;
    this.group.add(pcb);

    const silk = new THREE.Mesh(
      new THREE.BoxGeometry(PCB_W - 2, 0.05, PCB_L - 2),
      new THREE.MeshBasicMaterial({ color: 0x176b3a, transparent: true, opacity: 0.45 })
    );
    silk.position.y = 0.82;
    this.group.add(silk);
  }

  _buildIC() {
    const ic = new THREE.Mesh(
      new THREE.BoxGeometry(11, 1.4, 11),
      new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.95 })
    );
    ic.position.set(0, 1.5, 0);
    this.group.add(ic);

    const dot = new THREE.Mesh(
      new THREE.CylinderGeometry(0.4, 0.4, 0.12, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    dot.position.set(-4.2, 2.26, -4.2);
    this.group.add(dot);

    const xtal = new THREE.Mesh(
      new THREE.BoxGeometry(5, 1.2, 2),
      new THREE.MeshStandardMaterial({ color: 0xbababa, metalness: 0.85 })
    );
    xtal.position.set(10, 1.4, 5);
    this.group.add(xtal);

    [-22, 22].forEach((x, i) => {
      const led = new THREE.Mesh(
        new THREE.CircleGeometry(0.5, 12),
        new THREE.MeshStandardMaterial({
          color: i === 0 ? 0xff4444 : 0x44ff66,
          emissive: i === 0 ? 0x661111 : 0x116622,
          emissiveIntensity: 0.7
        })
      );
      led.rotation.x = -Math.PI / 2;
      led.position.set(x, 0.86, 2);
      this.group.add(led);
    });
  }

  _buildServoHeaders() {
    const headerMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.9 });
    const pinMat = new THREE.MeshStandardMaterial({ color: 0xd9c47a, metalness: 0.9 });

    for (let i = 0; i < 16; i++) {
      const x = -28.5 + i * 3.8;
      const header = new THREE.Mesh(new THREE.BoxGeometry(3, 4, 8), headerMat);
      header.position.set(x, 2.6, 10);
      header.castShadow = true;
      this.group.add(header);

      for (let p = 0; p < 3; p++) {
        const pinTop = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.4, 0.4), pinMat);
        pinTop.position.set(x, 4.6, 7.2 + p * 1.4);
        this.group.add(pinTop);
      }

      this.channels.push(new VirtualPin({
        id: `CH${i}`,
        label: `CH${i}`,
        type: 'CH',
        parent: this.group,
        localPos: new THREE.Vector3(x, 5.2, 10)
      }));
    }
  }

  _buildScrewTerminal() {
    const block = new THREE.Mesh(
      new THREE.BoxGeometry(11, 6, 7),
      new THREE.MeshStandardMaterial({ color: 0x1a7a2a, roughness: 0.75 })
    );
    block.position.set(-22, 3.5, -9);
    block.castShadow = true;
    this.group.add(block);

    [-25, -19].forEach((x, i) => {
      const sleeve = new THREE.Mesh(
        new THREE.CylinderGeometry(1.6, 1.6, 0.6, 16),
        new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.9 })
      );
      sleeve.rotation.x = Math.PI / 2;
      sleeve.position.set(x, 4.5, -8.4);
      this.group.add(sleeve);

      const screw = new THREE.Mesh(
        new THREE.CylinderGeometry(1.1, 1.1, 0.4, 16),
        new THREE.MeshStandardMaterial({ color: 0x808080, metalness: 0.95 })
      );
      screw.rotation.x = Math.PI / 2;
      screw.position.set(x, 4.5, -8.1);
      this.group.add(screw);

      const cross1 = new THREE.Mesh(
        new THREE.BoxGeometry(1.5, 0.06, 0.25),
        new THREE.MeshBasicMaterial({ color: 0x1a1a1a })
      );
      cross1.position.set(x, 4.74, -8.05);
      this.group.add(cross1);
      const cross2 = cross1.clone();
      cross2.rotation.y = Math.PI / 2;
      this.group.add(cross2);

      this.terminals.push(new VirtualPin({
        id: i === 0 ? 'TERM_V+' : 'TERM_GND',
        label: i === 0 ? 'V+' : 'GND',
        type: i === 0 ? 'VPLUS' : 'GND',
        sub: i === 0 ? 'Potencia servos' : 'Común',
        parent: this.group,
        localPos: new THREE.Vector3(x, 5.5, -9)
      }));
    });
  }

  _buildLogicPins() {
    const rowSpacing = 2.54;
    const totalLen = (Object.keys(PCA_PINS).length - 1) * rowSpacing;
    const startX = -totalLen / 2;
    const z = -10;

    for (const [id, def] of Object.entries(PCA_PINS)) {
      const x = startX + def.row * rowSpacing;
      const pin = new Pin({
        id,
        label: def.label,
        type: def.type,
        localPosition: new THREE.Vector3(x, 0.8, z)
      });
      this.pins[id] = pin;
      this.group.add(pin.group);
    }
  }

  getPin(id) {
    if (this.pins[id]) return this.pins[id];
    // Soporte para terminales V+/GND como destino de cables
    const term = this.terminals.find(t => t.id === id);
    if (term) return { getAnchorWorld: () => term.getAnchorWorld(), highlight: () => {} };
    return undefined;
  }

  /** Punto en mundo del header de un canal (0..15). Devuelve {signal, vplus, gnd} */
  getChannelAnchors(ch) {
    const x = -28.5 + ch * 3.8;
    const baseY = 5.2;
    const gnd  = this.group.localToWorld(new THREE.Vector3(x, baseY, 7.2));
    const vp   = this.group.localToWorld(new THREE.Vector3(x, baseY, 8.6));
    const sig  = this.group.localToWorld(new THREE.Vector3(x, baseY, 10.0));
    return { signal: sig, vplus: vp, gnd };
  }
}
