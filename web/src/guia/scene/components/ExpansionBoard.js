import * as THREE from 'three';
import { Pin } from './Pin.js';
import { ESP32 } from './ESP32.js';
import { VirtualPin } from '../ui/PinLabelOverlay.js';
import { makeTextDecal } from './TextDecal.js';
import { EXP_TRIPLET_GPIOS } from '../../connections.js';

/**
 * ESP32 Expansion Board (estilo Electrodemy / DOIT).
 * Iteración 2: franjas brillantes (MeshBasicMaterial), serigrafía blanca,
 * regulador, segundo micro-USB, labels JUMP/V/USB5V/DC.
 */
const PCB_W = 80;
const PCB_L = 130;
const PCB_THICK = 2;
const TOP_Y = PCB_THICK / 2;     // cara superior del PCB

const TRIPLET_X = 32;
const ESP32_Y_LIFT = 3.2;
const PIN_SPACING = 2.54;
const ROWS = 13;
const ROW_BLOCK_LEN = (ROWS - 1) * PIN_SPACING;

// Helpers de material
const stripeMat = (hex) => new THREE.MeshBasicMaterial({ color: hex });
const COLOR_S = 0xfacb12;   // amarillo señal
const COLOR_V = 0xff2424;   // rojo voltaje
const COLOR_G = 0x4a4a4a;   // gris GND
const COLOR_3V3 = 0xb71c1c; // rojo oscuro
const COLOR_GOLD = 0xd9c47a;

export class ExpansionBoard {
  constructor() {
    this.group = new THREE.Group();
    this.pins = {};
    this.tripletPins = {};

    this._buildPCB();
    this._buildMountingHoles();
    this._buildSocketHeaders();
    this._buildTripletColumns('left');
    this._buildTripletColumns('right');
    this._buildPowerBlock();
    this._buildI2CBlock();
    this._buildJumper();
    this._buildPowerInput();
    this._buildCapacitors();
    this._buildRegulator();
    this._buildAuxSilk();

    this.esp32 = new ESP32();
    this.esp32.group.position.set(0, TOP_Y + ESP32_Y_LIFT, 0);
    this.group.add(this.esp32.group);
  }

  /* PCB */
  _buildPCB() {
    const pcb = new THREE.Mesh(
      new THREE.BoxGeometry(PCB_W, PCB_THICK, PCB_L),
      new THREE.MeshStandardMaterial({ color: 0x080808, roughness: 0.92 })
    );
    pcb.castShadow = true;
    pcb.receiveShadow = true;
    this.group.add(pcb);

    // Marco blanco fino alrededor del slot del ESP32 (silkscreen)
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(30, 0.1, 60)),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })
    );
    frame.position.set(0, TOP_Y + 0.06, 0);
    this.group.add(frame);
  }

  _buildMountingHoles() {
    const off = 5;
    [[-PCB_W/2+off,-PCB_L/2+off],[PCB_W/2-off,-PCB_L/2+off],
     [-PCB_W/2+off, PCB_L/2-off],[PCB_W/2-off, PCB_L/2-off]
    ].forEach(([x, z]) => {
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(1.4, 2.6, 22),
        new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide })
      );
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(x, TOP_Y + 0.06, z);
      this.group.add(ring);
    });
  }

  /* Riel hembra donde encajan los pines del ESP32 */
  _buildSocketHeaders() {
    const mat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.95 });
    [-12.5, 12.5].forEach(x => {
      const rail = new THREE.Mesh(
        new THREE.BoxGeometry(3.2, 2.8, 40),
        mat
      );
      rail.position.set(x, TOP_Y + 1.4, 0);
      this.group.add(rail);

      // 15 agujeros visibles arriba
      for (let i = 0; i < 15; i++) {
        const z = -((15-1)/2)*PIN_SPACING + i * PIN_SPACING;
        const hole = new THREE.Mesh(
          new THREE.CylinderGeometry(0.4, 0.4, 0.2, 10),
          new THREE.MeshBasicMaterial({ color: 0x000000 })
        );
        hole.position.set(x, TOP_Y + 2.85, z);
        this.group.add(hole);
      }
    });
  }

  /* Tripletes S/V/G */
  _buildTripletColumns(side) {
    const sign = side === 'left' ? -1 : 1;
    const centerX = sign * TRIPLET_X;
    const startZ = -ROW_BLOCK_LEN / 2;
    const gpios = EXP_TRIPLET_GPIOS[side];
    const colOrder = side === 'left' ? ['S','V','G'] : ['G','V','S'];
    const stripeLen = ROW_BLOCK_LEN + 5;
    const stripeColor = { S: COLOR_S, V: COLOR_V, G: COLOR_G };

    // 3 franjas brillantes
    colOrder.forEach((kind, k) => {
      const xOff = (k - 1) * PIN_SPACING;
      const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(2.4, 0.12, stripeLen),
        stripeMat(stripeColor[kind])
      );
      stripe.position.set(centerX + xOff, TOP_Y + 0.07, 0);
      this.group.add(stripe);
    });

    // Header label en la cabeza de cada columna: "S V G" / "G V S"
    colOrder.forEach((kind, k) => {
      const xOff = (k - 1) * PIN_SPACING;
      const decal = makeTextDecal(kind, {
        width: 2.4, height: 2.4,
        color: '#ffffff', fontSize: 90
      });
      decal.position.set(centerX + xOff, TOP_Y + 0.5, -ROW_BLOCK_LEN/2 - 4);
      this.group.add(decal);
    });

    // Headers físicos + 3 pines metálicos por fila
    const headerMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.95 });
    const pinMat = new THREE.MeshStandardMaterial({ color: COLOR_GOLD, metalness: 0.9 });

    for (let i = 0; i < ROWS; i++) {
      const z = startZ + i * PIN_SPACING;
      const gpio = gpios[i];

      const header = new THREE.Mesh(
        new THREE.BoxGeometry(8.2, 1.4, 2.2),
        headerMat
      );
      header.position.set(centerX, TOP_Y + 0.7, z);
      this.group.add(header);

      for (let k = 0; k < 3; k++) {
        const kind = colOrder[k];
        const xOff = (k - 1) * PIN_SPACING;
        const metal = new THREE.Mesh(
          new THREE.CylinderGeometry(0.4, 0.4, 1.2, 12),
          pinMat
        );
        metal.position.set(centerX + xOff, TOP_Y + 1.6, z);
        this.group.add(metal);

        // Pin clickeable para D5·S (paso OE)
        if (gpio === 'D5' && side === 'right' && kind === 'S') {
          const pin = new Pin({
            id: 'D5_S',
            label: 'D5 · S',
            sub: 'Señal triplete',
            type: 'IO',
            localPosition: new THREE.Vector3(centerX + xOff, TOP_Y, z)
          });
          this.pins['D5_S'] = pin;
          this.group.add(pin.group);
        }
      }

      // Etiqueta serigráfica del GPIO al lado externo de la columna
      const labelX = sign * (TRIPLET_X + 5);
      const decal = makeTextDecal(gpio, {
        width: 5, height: 2,
        color: '#ffffff', fontSize: 68, fontWeight: '900'
      });
      decal.position.set(labelX, TOP_Y + 0.5, z);
      this.group.add(decal);

      // VirtualPin para que el overlay 2D también pueda mostrarlo (opcional)
      this.tripletPins[`${gpio}_label`] = new VirtualPin({
        id: `EXP_T_${side}_${i}`,
        label: gpio,
        type: 'IO',
        parent: this.group,
        localPos: new THREE.Vector3(labelX, TOP_Y + 1, z)
      });
    }
  }

  /* Bloque de potencia (esquina inferior-izquierda) */
  _buildPowerBlock() {
    const baseX = -25;
    const baseZ = PCB_L/2 - 18;

    const shape = new THREE.Shape();
    const w = 12, h = 18, c = 4;
    shape.moveTo(-w/2,     -h/2 + c);
    shape.lineTo(-w/2 + c, -h/2);
    shape.lineTo( w/2,     -h/2);
    shape.lineTo( w/2,      h/2);
    shape.lineTo(-w/2,      h/2);
    shape.closePath();
    const pad = new THREE.Mesh(
      new THREE.ShapeGeometry(shape),
      new THREE.MeshBasicMaterial({ color: 0x141414 })
    );
    pad.rotation.x = -Math.PI / 2;
    pad.position.set(baseX, TOP_Y + 0.04, baseZ);
    this.group.add(pad);

    // 3 columnas: GND (gris), 5V (rojo), 3V3 (rojo oscuro)
    const cols = [
      { id: 'PWR_GND', label: 'GND', type: 'GND',   color: COLOR_G,   sub: null },
      { id: 'PWR_5V',  label: '5V',  type: 'POWER', color: COLOR_V,   sub: '5 V salida' },
      { id: 'PWR_3V3', label: '3V3', type: 'POWER', color: COLOR_3V3, sub: '3.3 V salida' }
    ];
    const headerMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.95 });
    const pinMat = new THREE.MeshStandardMaterial({ color: COLOR_GOLD, metalness: 0.9 });

    cols.forEach((col, c) => {
      const colX = baseX - 4 + c * PIN_SPACING * 1.7;

      const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(2.4, 0.12, 10),
        stripeMat(col.color)
      );
      stripe.position.set(colX, TOP_Y + 0.07, baseZ);
      this.group.add(stripe);

      for (let r = 0; r < 3; r++) {
        const z = baseZ - 3 + r * PIN_SPACING;
        const header = new THREE.Mesh(
          new THREE.BoxGeometry(2.2, 1.4, 2.2),
          headerMat
        );
        header.position.set(colX, TOP_Y + 0.7, z);
        this.group.add(header);

        const metal = new THREE.Mesh(
          new THREE.CylinderGeometry(0.4, 0.4, 1.2, 12),
          pinMat
        );
        metal.position.set(colX, TOP_Y + 1.6, z);
        this.group.add(metal);
      }

      // Label serigráfico debajo de la columna
      const decal = makeTextDecal(col.label, {
        width: 4.5, height: 2.4,
        color: '#ffffff', fontSize: 70
      });
      decal.position.set(colX, TOP_Y + 0.5, baseZ + 6);
      this.group.add(decal);

      // Pin clickeable (al pin del medio)
      const pin = new Pin({
        id: col.id, label: col.label, sub: col.sub, type: col.type,
        localPosition: new THREE.Vector3(colX, TOP_Y, baseZ - 3 + PIN_SPACING)
      });
      this.pins[col.id] = pin;
      this.group.add(pin.group);
    });

    // "GND" vertical en la esquina izquierda
    const gndVert = makeTextDecal('GND', {
      width: 5, height: 2, color: '#ffffff', fontSize: 48, rotation: -Math.PI/2
    });
    gndVert.position.set(baseX - 8, TOP_Y + 0.5, baseZ);
    this.group.add(gndVert);
  }

  /* Bloque I2C (esquina inferior-derecha) */
  _buildI2CBlock() {
    const baseX = 26;
    const baseZ = PCB_L/2 - 22;

    const pad = new THREE.Mesh(
      new THREE.BoxGeometry(20, 0.06, 24),
      new THREE.MeshBasicMaterial({ color: 0x141414 })
    );
    pad.position.set(baseX, TOP_Y + 0.04, baseZ);
    this.group.add(pad);

    const rows = [
      { id: 'I2C_D22', label: 'D22', type: 'SCL', color: COLOR_S, sub: 'I²C SCL' },
      { id: 'I2C_D21', label: 'D21', type: 'SDA', color: COLOR_S, sub: 'I²C SDA' },
      { id: 'I2C_VCC', label: 'VCC', type: 'VCC', color: COLOR_V, sub: 'jumper 3V3/5V' },
      { id: 'I2C_GND', label: 'GND', type: 'GND', color: COLOR_G }
    ];
    const headerMat = new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.95 });
    const pinMat = new THREE.MeshStandardMaterial({ color: COLOR_GOLD, metalness: 0.9 });
    const rowGap = PIN_SPACING * 1.7;

    rows.forEach((row, r) => {
      const z = baseZ - 7.5 + r * rowGap;

      const stripe = new THREE.Mesh(
        new THREE.BoxGeometry(13, 0.12, 2.6),
        stripeMat(row.color)
      );
      stripe.position.set(baseX - 1, TOP_Y + 0.07, z);
      this.group.add(stripe);

      for (let k = 0; k < 3; k++) {
        const x = baseX - 3.5 + k * PIN_SPACING;
        const header = new THREE.Mesh(
          new THREE.BoxGeometry(2.2, 1.4, 2.2),
          headerMat
        );
        header.position.set(x, TOP_Y + 0.7, z);
        this.group.add(header);

        const metal = new THREE.Mesh(
          new THREE.CylinderGeometry(0.4, 0.4, 1.2, 12),
          pinMat
        );
        metal.position.set(x, TOP_Y + 1.6, z);
        this.group.add(metal);
      }

      const decal = makeTextDecal(row.label, {
        width: 6, height: 2.4,
        color: '#ffffff', fontSize: 80
      });
      decal.position.set(baseX + 6.5, TOP_Y + 0.5, z);
      this.group.add(decal);

      const pin = new Pin({
        id: row.id, label: row.label, type: row.type, sub: row.sub,
        localPosition: new THREE.Vector3(baseX - 1, TOP_Y, z)
      });
      this.pins[row.id] = pin;
      this.group.add(pin.group);
    });
  }

  /* Jumper 5V / 3V3 selector */
  _buildJumper() {
    const baseX = -8;
    const baseZ = PCB_L/2 - 38;

    const block = new THREE.Mesh(
      new THREE.BoxGeometry(10, 0.06, 3.2),
      new THREE.MeshBasicMaterial({ color: 0x141414 })
    );
    block.position.set(baseX, TOP_Y + 0.04, baseZ);
    this.group.add(block);

    [-3, 0, 3].forEach(dx => {
      const p = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 1.2, 1.4),
        new THREE.MeshStandardMaterial({ color: COLOR_GOLD, metalness: 0.9 })
      );
      p.position.set(baseX + dx, TOP_Y + 0.7, baseZ);
      this.group.add(p);
    });

    const cap = new THREE.Mesh(
      new THREE.BoxGeometry(3.4, 1.6, 2.6),
      stripeMat(0xf7c800)
    );
    cap.position.set(baseX + 1.5, TOP_Y + 1.5, baseZ);
    this.group.add(cap);

    this.group.add(this._silk('3.3V', baseX - 5.5, baseZ - 0.2, { width: 3, height: 1.2, fontSize: 38 }));
    this.group.add(this._silk('V',    baseX,      baseZ + 2.7, { width: 1.2, height: 1.2, fontSize: 38 }));
    this.group.add(this._silk('5V',   baseX + 5.4,baseZ - 0.2, { width: 2.6, height: 1.2, fontSize: 38 }));
    this.group.add(this._silk('JUMP', baseX,      baseZ + 4.5, { width: 4.5, height: 1.4, fontSize: 42 }));
  }

  /* DC jack, micro-USBs */
  _buildPowerInput() {
    const jackBody = new THREE.Mesh(
      new THREE.BoxGeometry(11, 9, 13),
      new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.7 })
    );
    jackBody.position.set(31, TOP_Y + 4.5, PCB_L/2 - 6);
    this.group.add(jackBody);

    const jackHole = new THREE.Mesh(
      new THREE.CylinderGeometry(2.4, 2.4, 6, 24),
      new THREE.MeshStandardMaterial({ color: 0x0a0a0a })
    );
    jackHole.rotation.x = Math.PI / 2;
    jackHole.position.set(31, TOP_Y + 4.5, PCB_L/2 - 2);
    this.group.add(jackHole);

    const jackPin = new THREE.Mesh(
      new THREE.CylinderGeometry(0.8, 0.8, 7, 16),
      new THREE.MeshBasicMaterial({ color: 0xfacb12 })
    );
    jackPin.rotation.x = Math.PI / 2;
    jackPin.position.set(31, TOP_Y + 4.5, PCB_L/2 - 1.5);
    this.group.add(jackPin);

    const dcLabel = makeTextDecal('DC 6.5-16V', {
      width: 9, height: 1.5, color: '#ffffff', fontSize: 40, rotation: -Math.PI/2
    });
    dcLabel.position.set(24, TOP_Y + 0.5, PCB_L/2 - 9);
    this.group.add(dcLabel);

    const usb1 = new THREE.Mesh(
      new THREE.BoxGeometry(8, 3.2, 6),
      new THREE.MeshStandardMaterial({ color: 0xb0b0b0, metalness: 0.95 })
    );
    usb1.position.set(-12, TOP_Y + 1.6, PCB_L/2 - 5);
    this.group.add(usb1);
    const usb1Label = makeTextDecal('USB5V', {
      width: 5, height: 1.4, color: '#ffffff', fontSize: 40
    });
    usb1Label.position.set(-12, TOP_Y + 0.5, PCB_L/2 - 11);
    this.group.add(usb1Label);

    const usb2 = new THREE.Mesh(
      new THREE.BoxGeometry(8, 3.2, 6),
      new THREE.MeshStandardMaterial({ color: 0xb0b0b0, metalness: 0.95 })
    );
    usb2.position.set(10, TOP_Y + 1.6, PCB_L/2 - 5);
    this.group.add(usb2);
  }

  /* Capacitores electrolíticos */
  _buildCapacitors() {
    [[-3, 32], [12, 32]].forEach(([x, z]) => {
      const can = new THREE.Mesh(
        new THREE.CylinderGeometry(3.2, 3.2, 7, 24),
        new THREE.MeshStandardMaterial({ color: 0x0a0f2a, roughness: 0.4, metalness: 0.5 })
      );
      can.position.set(x, TOP_Y + 3.5, z);
      this.group.add(can);

      const top = new THREE.Mesh(
        new THREE.CylinderGeometry(2.9, 2.9, 0.1, 24),
        new THREE.MeshBasicMaterial({ color: 0x080a18 })
      );
      top.position.set(x, TOP_Y + 7.05, z);
      this.group.add(top);

      const m1 = new THREE.Mesh(
        new THREE.BoxGeometry(4, 0.05, 0.4),
        new THREE.MeshBasicMaterial({ color: 0x444444 })
      );
      m1.position.set(x, TOP_Y + 7.08, z);
      this.group.add(m1);
      const m2 = m1.clone();
      m2.rotation.y = Math.PI / 2;
      this.group.add(m2);
    });
  }

  /* Regulador 1117c (TO-220) */
  _buildRegulator() {
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(6, 4.5, 3),
      new THREE.MeshStandardMaterial({ color: 0x111111, roughness: 0.9 })
    );
    body.position.set(0, TOP_Y + 2.3, 38);
    this.group.add(body);

    for (let i = -1; i <= 1; i++) {
      const leg = new THREE.Mesh(
        new THREE.BoxGeometry(0.6, 1.2, 0.4),
        new THREE.MeshStandardMaterial({ color: COLOR_GOLD, metalness: 0.9 })
      );
      leg.position.set(i * 1.6, TOP_Y + 0.6, 40);
      this.group.add(leg);
    }

    this.group.add(this._silk('1117c', 0, 41.5, { width: 4, height: 1.2, fontSize: 40 }));
  }

  /* Etiquetas auxiliares */
  _buildAuxSilk() {
    const tag = makeTextDecal('ESP32 EXPANSION', {
      width: 16, height: 1.5, color: '#ffffff', fontSize: 38
    });
    tag.position.set(0, TOP_Y + 0.5, -55);
    this.group.add(tag);

    const design = makeTextDecal('Design by Electrodemy', {
      width: 14, height: 1, color: '#ffffff', fontSize: 28, rotation: -Math.PI/2
    });
    design.position.set(-36, TOP_Y + 0.5, PCB_L/2 - 10);
    this.group.add(design);
  }

  _silk(text, x, z, { width=4, height=1.4, fontSize=44 } = {}) {
    const d = makeTextDecal(text, { width, height, color: '#ffffff', fontSize });
    d.position.set(x, TOP_Y + 0.5, z);
    return d;
  }

  getPin(id) { return this.pins[id]; }
}
