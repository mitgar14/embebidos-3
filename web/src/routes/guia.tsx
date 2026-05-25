// web/src/routes/guia.tsx
// Guía de conexión 3D (Fase 5, Plan 02).
// Monta la escena Three.js del repo esp32-pca9685-servo-guide dentro del sitio Tiny Trash.
// Ciclo de vida SolidJS: onMount inicializa la escena, onCleanup detiene el loop y
// dispone recursos para no fugar contexto WebGL al navegar entre rutas.

import { createEffect, onMount, onCleanup, type JSX } from 'solid-js';
import { A } from '@solidjs/router';
import * as THREE from 'three';
import { ThemeToggle } from '../components/ThemeToggle';
import { theme } from '../lib/theme';

// Importaciones de la escena portada
import { SceneManager }       from '../guia/scene/core/SceneManager';
import { CameraController }   from '../guia/scene/core/CameraController';
import { InteractionManager } from '../guia/scene/core/InteractionManager';

import { ExpansionBoard } from '../guia/scene/components/ExpansionBoard';
import { PCA9685 }        from '../guia/scene/components/PCA9685';
import { ExternalPSU }    from '../guia/scene/components/ExternalPSU';
import { Wire }           from '../guia/scene/components/Wire';
import { Servo }          from '../guia/scene/components/Servo';
import { ServoCable }     from '../guia/scene/components/ServoCable';

import { StepController }  from '../guia/scene/ui/StepController';
import { InfoPanel }       from '../guia/scene/ui/InfoPanel';
import { Tooltip }         from '../guia/scene/ui/Tooltip';
import { PinLabelOverlay } from '../guia/scene/ui/PinLabelOverlay';
import { WelcomeModal }    from '../guia/scene/ui/WelcomeModal';
import { PinDetail }       from '../guia/scene/ui/PinDetail';

import { STEPS, WIRE_COLORS, BOARD_INFO } from '../guia/connections';

// Colores de fondo de la escena según tema
const SCENE_BG_DARK  = '#111113';
const SCENE_BG_LIGHT = '#f8f9fa';
const SCENE_FOG_DARK  = 0x05080f;
const SCENE_FOG_LIGHT = 0xf8f9fa;

const BTN =
  'rounded-md border border-border px-3 py-1.5 text-sm text-text-primary ' +
  'hover:border-accent hover:bg-bg-surface transition-colors';

function IconBack(): JSX.Element {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}

export default function Guia() {
  let canvasRef!: HTMLCanvasElement;
  let stageRef!: HTMLDivElement;

  onMount(() => {
    // Referencias a los contenedores DOM que ya están montados en el JSX
    const canvas = canvasRef;

    /* ─── 1. Escena + cámara + interacción ─────────────────────── */
    const sm = new SceneManager(canvas);
    const cam = new CameraController(sm.camera, sm.renderer.domElement);
    const interaction = new InteractionManager(canvas, sm.camera);

    /* ─── 2. Placas ────────────────────────────────────────────── */
    const exp = new ExpansionBoard();
    exp.group.position.set(-55, 0, 0);
    sm.scene.add(exp.group);
    const esp32 = exp.esp32;

    const pca = new PCA9685();
    pca.group.position.set(55, 0, 0);
    pca.group.rotation.y = Math.PI;
    sm.scene.add(pca.group);

    const psu = new ExternalPSU();
    const usesPSU = STEPS.some((s: any) => s.from.board === 'external' || s.to.board === 'external');
    if (usesPSU) {
      psu.group.position.set(60, 0, 70);
      sm.scene.add(psu.group);
    }

    const boards: Record<string, any> = { exp, esp32, pca, external: psu };

    // Registrar pines clickeables
    [exp, pca, psu].forEach((b: any) =>
      Object.values(b.pins).forEach((p: any) => interaction.registerPin(p))
    );
    Object.values(esp32.pins).forEach((p: any) => interaction.registerPin(p));

    /* ─── 2b. Servos en CH0–CH3 ─────────────────────────────── */
    const servos: any[] = [];
    const servoCables: any[] = [];
    const servoColors = [0x1761b8, 0x1976d2, 0x1565c0, 0x0d47a1];

    const anchorCenter = (
      pca.getChannelAnchors(0).signal.x + pca.getChannelAnchors(3).signal.x
    ) / 2;
    const SERVO_SPACING = 14;
    const SERVO_BASE_X = anchorCenter - (3 * SERVO_SPACING) / 2;
    const headerZ = pca.getChannelAnchors(0).signal.z;
    const SERVO_Z = headerZ + 32;

    for (let i = 0; i < 4; i++) {
      const servo = new Servo(i, { color: servoColors[i] });
      servo.group.position.set(SERVO_BASE_X + i * SERVO_SPACING, 0, SERVO_Z);
      servo.group.rotation.y = Math.PI;
      sm.scene.add(servo.group);
      servos.push(servo);

      const cable = new ServoCable(servo, pca.getChannelAnchors(i));
      cable.addTo(sm.scene);
      cable.show(true);
      servoCables.push(cable);
    }

    /* ─── 3. Cables ─────────────────────────────────────────── */
    const wires = STEPS.map((step: any) => {
      const fromBoard = boards[step.from.board];
      const toBoard   = boards[step.to.board];
      const a = fromBoard.getPin(step.from.pin).getAnchorWorld();
      const b = toBoard.getPin(step.to.pin).getAnchorWorld();
      const wire = new Wire(a, b, (WIRE_COLORS as any)[step.color]);
      sm.scene.add(wire.mesh);
      return { step, wire };
    });

    /* ─── 4. Overlay 2D de etiquetas ──────────────────────── */
    const labels = new PinLabelOverlay(canvas, sm.camera);

    const usedKey = new Set<string>();
    for (const s of STEPS) {
      usedKey.add(`${(s as any).from.board}.${(s as any).from.pin}`);
      usedKey.add(`${(s as any).to.board}.${(s as any).to.pin}`);
    }
    const isUsed = (boardId: string, pinId: string) => usedKey.has(`${boardId}.${pinId}`);

    Object.values(exp.pins).forEach((p: any) => {
      labels.register(p, {
        side: p.id.startsWith('PWR') ? 'bottom' : 'right',
        used: isUsed('exp', p.id)
      });
    });
    Object.values(exp.tripletPins).forEach((vp: any) => {
      const isLeft = vp._local.x < 0;
      labels.register(vp, { side: isLeft ? 'left' : 'right', used: false });
    });
    for (const id of Object.keys(pca.pins)) {
      labels.register(pca.getPin(id), { side: 'right', used: isUsed('pca', id) });
    }
    pca.channels.forEach((ch: any) => labels.register(ch, { side: 'top', used: false }));
    pca.terminals.forEach((t: any) => labels.register(t, { side: 'right', used: false }));
    labels.register(psu.getPin('PSU+'), { side: 'right', used: true });
    labels.register(psu.getPin('PSU-'), { side: 'right', used: true });

    /* ─── 5. UI ────────────────────────────────────────────── */
    const panelEl = document.getElementById('guia-panel')!;
    const panel = new InfoPanel(panelEl);
    const tooltipEl = document.getElementById('guia-tooltip')!;
    const tooltip = new Tooltip(tooltipEl);
    const stepCtrl = new StepController(STEPS);

    document.getElementById('btn-prev')!.onclick = () => stepCtrl.prev();
    document.getElementById('btn-next')!.onclick = () => stepCtrl.next();
    document.getElementById('btn-all')!.onclick  = () => stepCtrl.showOverview();
    document.getElementById('btn-reset')!.onclick = () => cam.reset();

    let showAllLabels = false;
    const btnLabels = document.getElementById('btn-labels')!;
    function setShowAllLabels(v: boolean) {
      showAllLabels = v;
      labels.setShowAll(v);
      btnLabels.textContent = v ? 'Todos los pines' : 'Solo usados';
      btnLabels.classList.toggle('on', v);
    }
    btnLabels.onclick = () => setShowAllLabels(!showAllLabels);
    setShowAllLabels(false);

    const btnFly = document.getElementById('btn-fly')!;
    const flyHud = document.getElementById('fly-hud')!;
    function setFly(on: boolean) {
      cam.setMode(on ? 'fly' : 'orbit');
      btnFly.classList.toggle('on', on);
      flyHud.classList.toggle('visible', on);
      btnFly.textContent = on ? 'Salir vuelo' : 'Vuelo libre';
    }
    btnFly.onclick = () => setFly(cam.mode !== 'fly');

    const kbdHandler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight') stepCtrl.next();
      else if (e.key === 'ArrowLeft') stepCtrl.prev();
      else if (e.key === 'a' && cam.mode !== 'fly') stepCtrl.showOverview();
      else if (e.key === 'r') cam.reset();
      else if (e.key === 'l' && cam.mode !== 'fly') setShowAllLabels(!showAllLabels);
      else if (e.key === 'f' || e.key === 'F') setFly(cam.mode !== 'fly');
    };
    document.addEventListener('keydown', kbdHandler);

    const stepList = document.getElementById('guia-step-list')!;
    STEPS.forEach((s: any, i: number) => {
      const li = document.createElement('li');
      li.dataset.index = String(i);
      li.innerHTML = `
        <span class="chip" style="background:#${(WIRE_COLORS as any)[s.color].toString(16).padStart(6,'0')}"></span>
        <span class="step-name">${s.title}</span>
        ${s.required ? '' : '<span class="opt">opc</span>'}
      `;
      li.onclick = () => stepCtrl.goTo(i);
      stepList.appendChild(li);
    });

    document.getElementById('esp32-name')!.textContent  = BOARD_INFO.exp.name;
    document.getElementById('esp32-notes')!.textContent = BOARD_INFO.exp.notes;
    document.getElementById('pca-name')!.textContent    = BOARD_INFO.pca.name;
    document.getElementById('pca-notes')!.textContent   = BOARD_INFO.pca.notes;

    /* ─── 6. Reacción a cambios de paso ─────────────────────── */
    function highlightStep(state: any) {
      [exp, pca, psu].forEach((b: any) => Object.values(b.pins).forEach((p: any) => p.highlight(false)));
      Object.values(esp32.pins).forEach((p: any) => p.highlight(false));
      labels.clearHighlights();

      if (state.mode === 'overview') {
        wires.forEach((w: any) => w.wire.show());
        panel.showOverview(STEPS.length);
        cam.reset();
      } else {
        wires.forEach((w: any, idx: number) => {
          if (idx <= state.index) w.wire.show();
          else w.wire.hide();
        });
        const current = wires[state.index];
        stepCtrl.markVisited();

        const fromBoard = boards[current.step.from.board];
        const toBoard   = boards[current.step.to.board];
        const pinA = fromBoard.getPin(current.step.from.pin);
        const pinB = toBoard.getPin(current.step.to.pin);
        const color = (WIRE_COLORS as any)[current.step.color];
        pinA.highlight(true, color);
        pinB.highlight(true, color);
        labels.highlight(pinA, true);
        labels.highlight(pinB, true);

        panel.renderStep(current.step, state.index, STEPS.length);

        const a = pinA.getAnchorWorld();
        const b = pinB.getAnchorWorld();
        const target = a.clone().add(b).multiplyScalar(0.5);
        const offset = new THREE.Vector3(0, 65, 90);
        cam.flyTo(target, target.clone().add(offset));
      }

      [...stepList.children].forEach((li: any, i: number) => {
        li.classList.toggle('active', state.mode === 'step' && i === state.index);
        li.classList.toggle('done', state.visited.has((STEPS as any)[i].id));
      });
    }

    stepCtrl.onChange(highlightStep);

    /* ─── 7. Hover/click sobre pines ─────────────────────── */
    interaction.on('hover',  (pin: any) => { pin.highlight(true, 0xffffff); labels.highlight(pin, true); tooltip.show(pin); });
    interaction.on('leave',  (pin: any) => { pin.highlight(false); labels.highlight(pin, false); tooltip.hide(); refreshHighlight(); });
    const pinDetail = new PinDetail();
    interaction.on('click',  (pin: any) => pinDetail.show(pin));

    function refreshHighlight() { highlightStep(stepCtrl.snapshot()); }

    /* ─── 8. Test de servos (sweep automático) ───────────── */
    let servoTestActive = false;
    let servoTestT = 0;

    function startServoTest() {
      servoTestActive = true;
      servoTestT = 0;
      const btn = document.getElementById('btn-servos')!;
      btn.classList.add('on');
      btn.textContent = 'Detener servos';
    }
    function stopServoTest() {
      servoTestActive = false;
      servos.forEach(s => s.setAngle(90));
      const btn = document.getElementById('btn-servos')!;
      btn.classList.remove('on');
      btn.textContent = 'Probar servos';
    }
    document.getElementById('btn-servos')!.onclick = () =>
      servoTestActive ? stopServoTest() : startServoTest();

    function updateServos(dt: number) {
      if (servoTestActive) {
        servoTestT += dt;
        servos.forEach((s, i) => {
          const phase = i * (Math.PI / 4);
          const a = 90 + 80 * Math.sin(servoTestT * 1.8 + phase);
          s.setAngle(a);
        });
      }
      servos.forEach(s => s.update(dt));
    }

    /* ─── 9. Loop ──────────────────────────────────────── */
    const clock = new THREE.Clock();
    sm.renderer.setAnimationLoop(() => {
      const dt = clock.getDelta();
      cam.update(dt);
      interaction.update();
      wires.forEach((w: any) => w.wire.update(dt));
      servoCables.forEach((c: any) => c.update(dt));
      updateServos(dt);
      labels.update();
      sm.render();
    });

    document.getElementById('guia-loader')?.remove();

    /* ─── 10. Welcome modal ───────────────────────────── */
    const welcome = new WelcomeModal();
    welcome.open(false);
    document.getElementById('btn-help')!.onclick = () => welcome.open(true);
    const helpKey = (e: KeyboardEvent) => {
      if (e.key === '?' || (e.shiftKey && e.key === '/')) welcome.open(true);
    };
    document.addEventListener('keydown', helpKey);

    /* ─── Iniciar en vista general ────────────────────── */
    stepCtrl.showOverview();

    /* ─── Tema: createEffect dentro de onMount ────────── */
    createEffect(() => {
      const t = theme();
      sm.scene.background = new THREE.Color(t === 'dark' ? SCENE_BG_DARK : SCENE_BG_LIGHT);
      sm.scene.fog.color.set(t === 'dark' ? SCENE_FOG_DARK : SCENE_FOG_LIGHT);
    });

    /* ─── onCleanup ────────────────────────────────────── */
    onCleanup(() => {
      // Detener loop
      sm.renderer.setAnimationLoop(null);

      // Dispose de la cámara (OrbitControls + FlyController)
      cam.dispose?.();

      // Desconectar ResizeObserver del canvas
      sm.observer?.disconnect();

      // Remover listeners de teclado
      document.removeEventListener('keydown', kbdHandler);
      document.removeEventListener('keydown', helpKey);

      // Dispose de geometrías y materiales de la escena
      sm.scene.traverse((obj: any) => {
        obj.geometry?.dispose();
        if (Array.isArray(obj.material)) {
          obj.material.forEach((m: any) => m.dispose());
        } else {
          obj.material?.dispose();
        }
      });

      // Dispose del renderer WebGL
      sm.renderer.dispose();

      // Limpiar DOM de elementos creados por PinDetail y WelcomeModal
      pinDetail?.root?.remove();
      welcome?.overlay?.remove();
    });
  });

  return (
    <div class="h-screen overflow-hidden bg-bg-app flex flex-col">
      {/* Header del sitio */}
      <header class="flex items-center justify-between border-b border-border px-6 h-14 shrink-0">
        <div class="flex items-center gap-3">
          <A href="#/" class={BTN} aria-label="Volver al hub">
            <IconBack />
          </A>
          <span class="font-semibold text-text-primary">Guía de conexión</span>
        </div>
        <ThemeToggle />
      </header>

      {/* Cuerpo: 3 columnas */}
      <div class="flex flex-1 min-h-0">

        {/* Sidebar izquierdo */}
        <aside class="w-64 shrink-0 border-r border-border overflow-y-auto bg-bg-panel flex flex-col">
          <div class="px-4 pt-4 pb-3 border-b border-border">
            <p class="text-sm font-semibold text-text-primary">ESP32 · PCA9685</p>
            <p class="text-xs text-text-secondary mt-0.5">Guía 3D · I²C + PWM 16 canales</p>
          </div>

          {/* Placas */}
          <div class="px-4 pt-3 space-y-2">
            <div class="flex gap-2 items-start">
              <span class="mt-1 w-2 h-2 rounded-full bg-blue-500 shrink-0" />
              <div>
                <p class="text-xs font-medium text-text-primary" id="esp32-name">ESP32 Expansion Board</p>
                <p class="text-[11px] text-text-secondary leading-relaxed mt-0.5" id="esp32-notes" />
              </div>
            </div>
            <div class="flex gap-2 items-start">
              <span class="mt-1 w-2 h-2 rounded-full bg-green-600 shrink-0" />
              <div>
                <p class="text-xs font-medium text-text-primary" id="pca-name">PCA9685</p>
                <p class="text-[11px] text-text-secondary leading-relaxed mt-0.5" id="pca-notes" />
              </div>
            </div>
          </div>

          {/* BOM */}
          <div class="px-4 pt-4">
            <p class="text-[11px] font-medium uppercase tracking-wider text-text-secondary mb-2">Materiales</p>
            <ul class="space-y-1.5 text-xs text-text-secondary">
              <li class="flex gap-2"><span class="text-text-primary font-medium">1</span> ESP32-WROOM-32 DevKit V1</li>
              <li class="flex gap-2"><span class="text-text-primary font-medium">1</span> Expansion Board (Electrodemy)</li>
              <li class="flex gap-2"><span class="text-text-primary font-medium">1</span> PCA9685 16-ch PWM I²C</li>
              <li class="flex gap-2"><span class="text-text-primary font-medium">4</span> Tower Pro SG90 — canales 0–3</li>
              <li class="flex gap-2"><span class="text-text-primary font-medium">6+</span> Cables Dupont H-H ~10 cm</li>
              <li class="flex gap-2"><span class="text-text-primary font-medium">1</span> Cable Micro-USB (datos)</li>
            </ul>
          </div>

          {/* Lista de pasos */}
          <div class="px-4 pt-4 flex-1">
            <p class="text-[11px] font-medium uppercase tracking-wider text-text-secondary mb-2">Pasos</p>
            <ol id="guia-step-list" class="space-y-1 list-none p-0 m-0" />
          </div>

          {/* Atajos de teclado */}
          <div class="px-4 py-3 border-t border-border">
            <p class="text-[11px] text-text-secondary leading-relaxed">
              <kbd class="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 rounded border border-border bg-bg-surface font-mono text-[10px] text-text-primary">←</kbd>{' '}
              <kbd class="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 rounded border border-border bg-bg-surface font-mono text-[10px] text-text-primary">→</kbd>{' '}
              pasos · <kbd class="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 rounded border border-border bg-bg-surface font-mono text-[10px] text-text-primary">A</kbd>{' '}
              todos · <kbd class="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 rounded border border-border bg-bg-surface font-mono text-[10px] text-text-primary">L</kbd>{' '}
              etiquetas · <kbd class="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 rounded border border-border bg-bg-surface font-mono text-[10px] text-text-primary">F</kbd>{' '}
              vuelo · <kbd class="inline-flex items-center justify-center min-w-[1.25rem] h-5 px-1 rounded border border-border bg-bg-surface font-mono text-[10px] text-text-primary">R</kbd>{' '}
              reset
            </p>
          </div>
        </aside>

        {/* Canvas principal */}
        <main class="flex-1 relative overflow-hidden" ref={stageRef!}>
          <canvas ref={canvasRef!} style="width:100%;height:100%;display:block" />

          {/* SVG de líneas guía + overlay de etiquetas (gestionados por PinLabelOverlay) */}
          <svg id="leader-lines" xmlns="http://www.w3.org/2000/svg"
            style="position:absolute;top:0;left:0;pointer-events:none;overflow:visible" />
          <div id="pin-labels-overlay"
            style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none" />

          {/* Loader (eliminado por JS al arrancar la escena) */}
          <div id="guia-loader"
            class="absolute inset-0 flex items-center justify-center bg-bg-app z-20">
            <svg class="animate-spin text-text-secondary" width="24" height="24" viewBox="0 0 24 24"
              fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
          </div>

          {/* Controles flotantes */}
          <div class="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2 z-10">
            <button id="btn-prev"   aria-label="Paso anterior" class={BTN}>◀</button>
            <button id="btn-next"   aria-label="Siguiente paso" class={BTN}>▶</button>
            <button id="btn-all"    aria-label="Mostrar todas las conexiones" class={BTN}>Todas</button>
            <button id="btn-labels" aria-label="Mostrar u ocultar etiquetas" class={`${BTN} guia-toggle`}>Etiquetas</button>
            <button id="btn-servos" aria-label="Sweep automático en CH0–CH3" class={`${BTN} guia-toggle`}>Probar servos</button>
            <button id="btn-fly"    aria-label="Cámara libre WASD" class={`${BTN} guia-toggle`}>Vuelo libre</button>
            <button id="btn-reset"  aria-label="Vista inicial" class={BTN}>Recentrar</button>
            <button id="btn-help"   aria-label="Abrir guía de uso" class={BTN}>?</button>
          </div>

          {/* Tooltip de pin (gestionado por Tooltip.js) */}
          <div id="guia-tooltip" class="tooltip" style="position:fixed;pointer-events:none;z-index:50;display:none">
            <div class="tt-title" />
            <div class="tt-type" />
          </div>

          {/* HUD de cámara libre */}
          <div id="fly-hud" class="fly-hud" style="display:none">
            <p class="fh-title">Vuelo libre</p>
            <p class="fh-row">
              <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd> mover ·{' '}
              <kbd>Q</kbd>/<kbd>E</kbd> bajar/subir
            </p>
            <p class="fh-row"><kbd>Shift</kbd> boost · arrastrar para mirar · rueda velocidad</p>
            <p class="fh-row"><kbd>F</kbd> volver a órbita</p>
          </div>
        </main>

        {/* Panel derecho: detalle del paso */}
        <aside id="guia-panel" class="w-72 shrink-0 border-l border-border overflow-y-auto bg-bg-panel">
          <div class="p-4 border-b border-border flex items-center gap-3">
            <span id="step-badge" class="badge badge-required text-xs px-2 py-0.5 rounded font-medium">Obligatorio</span>
            <span id="step-counter" class="text-xs text-text-secondary font-mono">1 / 6</span>
          </div>

          <div class="px-4 py-3 border-b border-border flex items-center gap-2">
            <span class="text-xs text-text-secondary">Color del cable:</span>
            <span id="step-color" class="inline-block w-4 h-4 rounded-full border border-border" />
          </div>

          <div class="p-4">
            <h2 id="step-title" class="text-sm font-semibold text-text-primary mb-2">Cargando…</h2>
            <p id="step-description" class="text-xs text-text-secondary leading-relaxed mb-3" />
            <pre class="rounded-md border border-border bg-bg-surface px-3 py-2 overflow-x-auto">
              <code id="step-code" class="font-mono text-xs text-text-secondary" />
            </pre>
          </div>

          <details class="px-4 pb-3 border-t border-border">
            <summary class="py-2 text-xs text-text-secondary cursor-pointer select-none">
              Comprobación rápida con I²C scanner
            </summary>
            <pre class="mt-2 rounded-md border border-border bg-bg-surface px-3 py-2 overflow-x-auto"><code class="font-mono text-xs text-text-secondary">{`#include <Wire.h>
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Serial.println("Escaneando I2C...");
  for (byte a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0)
      Serial.printf("Dispositivo en 0x%02X\\n", a);
  }
}
void loop() {}`}</code></pre>
            <p class="mt-2 text-[11px] text-text-secondary leading-relaxed">
              Si todo está bien deberías ver <code class="font-mono">0x40</code> (PCA9685 por defecto).
            </p>
          </details>

          <details class="px-4 pb-4 border-t border-border">
            <summary class="py-2 text-xs text-text-secondary cursor-pointer select-none">
              Ejemplo: mover un servo en el canal 0
            </summary>
            <pre class="mt-2 rounded-md border border-border bg-bg-surface px-3 py-2 overflow-x-auto"><code class="font-mono text-xs text-text-secondary">{`#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

void setup() {
  Wire.begin(21, 22);
  pwm.begin();
  pwm.setPWMFreq(50);
}
void loop() {
  pwm.setPWM(0, 0, 150);   // ~0 grados
  delay(1000);
  pwm.setPWM(0, 0, 600);   // ~180 grados
  delay(1000);
}`}</code></pre>
          </details>
        </aside>

      </div>
    </div>
  );
}
