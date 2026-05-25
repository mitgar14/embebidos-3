import * as THREE from 'three';

/**
 * Configura escena, cámara, renderer e iluminación.
 * No conoce nada del dominio (placas, cables). Solo Three.js.
 *
 * Adaptado para SolidJS: usa ResizeObserver en lugar de window.addEventListener('resize')
 * para que funcione cuando el canvas no ocupa toda la ventana.
 * Expone this.observer para que onCleanup pueda desconectarlo.
 */
export class SceneManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = null;
    this.scene.fog = new THREE.Fog(0x05080f, 200, 600);

    this.camera = new THREE.PerspectiveCamera(
      50,
      canvas.clientWidth / canvas.clientHeight,
      0.1,
      2000
    );
    this.camera.position.set(0, 175, 175);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

    this._setupLights();
    this._setupFloor();

    // ResizeObserver en lugar de window resize para correcta convivencia con SolidJS
    this.observer = new ResizeObserver(() => this._onResize());
    this.observer.observe(canvas);
  }

  _setupLights() {
    const hemi = new THREE.HemisphereLight(0x88aaff, 0x1a1a2a, 0.55);
    this.scene.add(hemi);

    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(60, 120, 80);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -150;
    key.shadow.camera.right = 150;
    key.shadow.camera.top = 150;
    key.shadow.camera.bottom = -150;
    key.shadow.bias = -0.0005;
    this.scene.add(key);

    const rim = new THREE.DirectionalLight(0x6fa8ff, 0.6);
    rim.position.set(-80, 40, -60);
    this.scene.add(rim);
  }

  _setupFloor() {
    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(220, 64),
      new THREE.MeshStandardMaterial({
        color: 0x0a0f1c, roughness: 1, metalness: 0
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -1.2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    const grid = new THREE.GridHelper(300, 30, 0x1d2a4a, 0x121826);
    grid.position.y = -1.15;
    this.scene.add(grid);
  }

  _onResize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }
}
