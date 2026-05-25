import * as THREE from 'three';

/**
 * Un pin físico: cilindro metálico vertical + zócalo negro.
 * Sin etiqueta 3D (las etiquetas se gestionan en `PinLabelOverlay` 2D).
 */
export class Pin {
  constructor({ id, label, type, sub, localPosition }) {
    this.id = id;
    this.label = label;
    this.type = type;
    this.sub = sub;

    const socket = new THREE.Mesh(
      new THREE.BoxGeometry(2.2, 0.6, 2.2),
      new THREE.MeshStandardMaterial({ color: 0x0a0a0a, roughness: 0.9 })
    );
    socket.position.y = 0.3;

    const metal = new THREE.Mesh(
      new THREE.CylinderGeometry(0.45, 0.45, 2.2, 16),
      new THREE.MeshStandardMaterial({
        color: 0xd9c47a, metalness: 0.95, roughness: 0.25
      })
    );
    metal.position.y = 1.7;
    metal.material.emissive = new THREE.Color(0x000000);

    this.group = new THREE.Group();
    this.group.add(socket);
    this.group.add(metal);
    this.group.position.copy(localPosition);

    this.metalMesh = metal;
    this.socketMesh = socket;
    metal.userData.pin = this;
    socket.userData.pin = this;
  }

  getAnchorWorld() {
    const v = new THREE.Vector3(0, 2.8, 0);
    this.group.localToWorld(v);
    return v;
  }

  highlight(on, color = 0x44ff88) {
    this.metalMesh.material.emissive.setHex(on ? color : 0x000000);
    this.metalMesh.material.emissiveIntensity = on ? 0.9 : 0;
  }
}
