import * as THREE from 'three';

/**
 * Tower Pro MicroServo 9g SG90.
 * Proporciones reales aprox (escaladas):
 *   cuerpo  22.5 (alto) x 12 (profundo) x 22.5 (ancho)
 *   wings   32.2 long  x 12 deep  x 2.5 thick
 *   gear    Ø6 hub blanco saliendo por la parte superior
 *   horn    cruz de 4 brazos blanca
 *   cable   3 hilos (naranja, rojo, marrón) saliendo del costado
 */
export class Servo {
  constructor(channel, opts = {}) {
    this.channel = channel;
    this.angleDeg = 90;
    this.targetDeg = 90;
    this.group = new THREE.Group();

    const bodyColor = opts.color ?? 0x1761b8;

    // Cuerpo principal (caja vertical)
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(9, 11, 5.5),
      new THREE.MeshStandardMaterial({ color: bodyColor, roughness: 0.55 })
    );
    body.position.y = 5.5;
    body.castShadow = true;
    this.group.add(body);

    // Tapa superior más clara (parte del engranaje, donde sale el horn)
    const topCap = new THREE.Mesh(
      new THREE.BoxGeometry(9, 1.5, 5.5),
      new THREE.MeshStandardMaterial({ color: 0x1f7adb, roughness: 0.5 })
    );
    topCap.position.y = 11.5;
    this.group.add(topCap);

    // Alerones de montaje (extensiones laterales planas)
    const flangeMat = new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.6 });
    [-1, 1].forEach(dx => {
      const flange = new THREE.Mesh(
        new THREE.BoxGeometry(3.5, 1, 5.5),
        flangeMat
      );
      flange.position.set(dx * 6.2, 6.5, 0);
      flange.castShadow = true;
      this.group.add(flange);

      // Agujero de tornillo (decorativo)
      const hole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.5, 0.5, 1.05, 12),
        new THREE.MeshBasicMaterial({ color: 0x0a0a0a })
      );
      hole.position.set(dx * 6.8, 6.5, 0);
      this.group.add(hole);
    });

    // Engranaje superior visible (cilindro blanco grande, descentrado)
    const gearOuter = new THREE.Mesh(
      new THREE.CylinderGeometry(2.4, 2.4, 1, 24),
      new THREE.MeshStandardMaterial({ color: 0xeaeaea, roughness: 0.45 })
    );
    gearOuter.position.set(-2, 12.7, 0);
    gearOuter.castShadow = true;
    this.group.add(gearOuter);

    // Engranaje pequeño adyacente (visible al lado del grande)
    const gearSmall = new THREE.Mesh(
      new THREE.CylinderGeometry(1.2, 1.2, 0.7, 16),
      new THREE.MeshStandardMaterial({ color: 0xeaeaea, roughness: 0.5 })
    );
    gearSmall.position.set(2.5, 12.55, 0);
    this.group.add(gearSmall);

    // Pivot del horn (centrado en el engranaje grande)
    this.hornPivot = new THREE.Group();
    this.hornPivot.position.set(-2, 13.3, 0);
    this.group.add(this.hornPivot);

    // Horn tipo CRUZ (4 brazos de SG90)
    const hornMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 });
    const armGeom = new THREE.BoxGeometry(11, 0.5, 1.4);
    const arm1 = new THREE.Mesh(armGeom, hornMat);
    arm1.castShadow = true;
    this.hornPivot.add(arm1);
    const arm2 = new THREE.Mesh(armGeom, hornMat);
    arm2.rotation.y = Math.PI / 2;
    arm2.castShadow = true;
    this.hornPivot.add(arm2);

    // Hub central del horn
    const hub = new THREE.Mesh(
      new THREE.CylinderGeometry(1.0, 1.0, 0.8, 16),
      hornMat
    );
    hub.position.y = -0.1;
    this.hornPivot.add(hub);

    // Pequeños agujeros decorativos en los brazos
    [-4.5, -3, 3, 4.5].forEach(dx => {
      const hole = new THREE.Mesh(
        new THREE.CylinderGeometry(0.18, 0.18, 0.55, 8),
        new THREE.MeshBasicMaterial({ color: 0x303030 })
      );
      hole.position.set(dx, 0.05, 0);
      this.hornPivot.add(hole);
    });

    // Cable trifilar saliendo del lateral
    this.cableExit = new THREE.Vector3(0, 5.5, 3.1);
    this._buildCableStub();

    // Etiqueta "SG90" estilo serigrafía blanca sobre el cuerpo
    this._buildBodyDecal();

    // Sprite del canal (CH0/1/2/3)
    this._buildChannelSprite();
  }

  _buildCableStub() {
    const colors = [0xe07a1a, 0xc62828, 0x4a2c1a]; // naranja, rojo, marrón
    colors.forEach((c, i) => {
      const stub = new THREE.Mesh(
        new THREE.CylinderGeometry(0.35, 0.35, 4, 10),
        new THREE.MeshStandardMaterial({ color: c, roughness: 0.7 })
      );
      stub.rotation.x = Math.PI / 2;
      stub.position.set((i - 1) * 0.9, 5.5, 4.5);
      this.group.add(stub);
    });
  }

  _buildBodyDecal() {
    const c = document.createElement('canvas');
    c.width = 256; c.height = 96;
    const ctx = c.getContext('2d');
    ctx.fillStyle = 'rgba(255,255,255,0)';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 56px Inter, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('SG90', c.width/2, 28);
    ctx.font = '36px Inter, sans-serif';
    ctx.fillText('9g', c.width/2, 70);
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const decalMat = new THREE.MeshBasicMaterial({
      map: tex, transparent: true, depthWrite: false
    });
    const decal = new THREE.Mesh(new THREE.PlaneGeometry(5, 1.9), decalMat);
    decal.position.set(0, 6, 2.81);
    this.group.add(decal);
  }

  _buildChannelSprite() {
    const c = document.createElement('canvas');
    c.width = 256; c.height = 128;
    const ctx = c.getContext('2d');
    ctx.fillStyle = 'rgba(0,0,0,0)';
    ctx.fillRect(0,0,c.width,c.height);
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = 'rgba(0,0,0,0.65)';
    ctx.lineWidth = 6;
    ctx.font = 'bold 96px Inter, sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.strokeText(`CH${this.channel}`, c.width/2, c.height/2);
    ctx.fillText(`CH${this.channel}`, c.width/2, c.height/2);

    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true }));
    sprite.scale.set(8, 4, 1);
    sprite.position.set(0, 18, 0);
    this.group.add(sprite);
  }

  setAngle(deg) { this.targetDeg = Math.max(0, Math.min(180, deg)); }
  setAngleInstant(deg) {
    this.angleDeg = Math.max(0, Math.min(180, deg));
    this.targetDeg = this.angleDeg;
    this._applyRotation();
  }

  update(dt) {
    if (Math.abs(this.targetDeg - this.angleDeg) < 0.1) return;
    const speed = 240; // deg/s típico SG90
    const dir = Math.sign(this.targetDeg - this.angleDeg);
    this.angleDeg += dir * Math.min(speed * dt, Math.abs(this.targetDeg - this.angleDeg));
    this._applyRotation();
  }

  _applyRotation() {
    const rad = (this.angleDeg - 90) * Math.PI / 180;
    this.hornPivot.rotation.y = rad;
  }

  getCableExitWorld() {
    return this.group.localToWorld(this.cableExit.clone());
  }
}
