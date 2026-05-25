// web/src/routes/control.tsx
// Control de servos ESP32 (Fase 5, Plan 03, CTRL-01..03).
// Estado dual: panel de control autoconfigurado por telemetría MQTT (ESP32 online)
// o instructivo de flasheo (ESP32 offline/ausente). Sin backend FastAPI.

import { createSignal, createEffect, For, Show, onCleanup, type JSX } from 'solid-js';
import { A } from '@solidjs/router';
import { ThemeToggle } from '../components/ThemeToggle';
import { StatusDot } from '../components/StatusDot';
import {
  mqttStatus,
  esp32Online,
  numServos,
  channels,
  angles,
  presets,
  rssi,
  uptimeSec,
  publish,
  requestState,
  mqttClient,
} from '../stores/servoStore';
import {
  getDeviceId,
  setDeviceId,
  topicCmd,
  topicState,
  topicOnline,
  cmdMove,
  cmdSavePreset,
  cmdLoadPreset,
} from '../lib/servoProtocol';

// ─── Constantes de botón ──────────────────────────────────────────────────────

const BTN =
  'rounded-md border border-border px-3 py-1.5 text-sm text-text-primary ' +
  'hover:border-accent hover:bg-bg-surface transition-colors ' +
  'disabled:opacity-40 disabled:hover:border-border disabled:hover:bg-transparent';
const BTN_PRIMARY =
  'rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-text ' +
  'hover:opacity-90 transition-opacity disabled:opacity-40 disabled:hover:opacity-40';

// ─── Iconos SVG inline ────────────────────────────────────────────────────────

function IconBack() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M15 18l-6-6 6-6" />
    </svg>
  );
}

function IconRefresh() {
  return (
    <svg
      width="15"
      height="15"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

// ─── Tooltip informativo (mismo patrón que labelling.tsx) ─────────────────────

function InfoTip(props: {
  children: JSX.Element;
  label?: string;
  placement?: 'top' | 'bottom';
  width?: string;
}) {
  const [open, setOpen] = createSignal(false);
  const down = () => props.placement === 'bottom';
  return (
    <span
      class="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        aria-label={props.label ?? 'Más información'}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        class="flex items-center justify-center text-text-secondary hover:text-text-primary transition-colors"
      >
        <svg
          width="13"
          height="13"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4" />
          <path d="M12 8h.01" />
        </svg>
      </button>
      <span
        role="tooltip"
        classList={{ 'info-tip-open': open() }}
        class={`info-tip ${
          down() ? 'info-tip-down top-full mt-2' : 'bottom-full mb-2'
        } pointer-events-none absolute right-0 z-30 ${
          props.width ?? 'w-72'
        } rounded-md border border-border bg-bg-panel px-3 py-2.5 text-xs text-text-secondary leading-relaxed`}
      >
        {props.children}
      </span>
    </span>
  );
}

// ─── Gauge SVG semicircular (arco de 180 grados) ─────────────────────────────
// pathLength=100 simplifica el cálculo: dashoffset = 100 - (angle/180)*100.
// El gradiente usa --accent en vez de los colores del repo original.

function ServoGauge(props: { angle: number; size?: number }) {
  const sz = () => props.size ?? 100;
  const r = () => sz() * 0.38;
  const cx = () => sz() / 2;
  const cy = () => sz() / 2 + r() * 0.25;

  // Arco de 180 grados: d construido con el radio y el centro.
  // El arco va de izquierda a derecha (sweep 1) pasando por arriba.
  const arcPath = () => {
    const R = r();
    const C = cx();
    const Y = cy();
    return `M ${C - R} ${Y} A ${R} ${R} 0 0 1 ${C + R} ${Y}`;
  };

  // stroke-dashoffset: 0 = lleno (180 grados); 100 = vacío (0 grados).
  const offset = () => 100 - (Math.max(0, Math.min(180, props.angle)) / 180) * 100;

  return (
    <svg
      width={sz()}
      height={sz() * 0.6}
      viewBox={`0 0 ${sz()} ${sz() * 0.6}`}
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stop-color="var(--accent)" stop-opacity="0.4" />
          <stop offset="100%" stop-color="var(--accent)" />
        </linearGradient>
      </defs>
      {/* Arco de fondo */}
      <path
        d={arcPath()}
        fill="none"
        stroke="var(--border-color)"
        stroke-width={sz() * 0.08}
        stroke-linecap="round"
        pathLength={100}
      />
      {/* Arco de progreso */}
      <path
        d={arcPath()}
        fill="none"
        stroke="url(#gauge-grad)"
        stroke-width={sz() * 0.08}
        stroke-linecap="round"
        pathLength={100}
        stroke-dasharray={100}
        stroke-dashoffset={offset()}
        style={{ transition: 'stroke-dashoffset 120ms ease-out' }}
      />
      {/* Ángulo central */}
      <text
        x={cx()}
        y={cy() - r() * 0.08}
        text-anchor="middle"
        dominant-baseline="auto"
        style={{
          fill: 'var(--text-primary)',
          'font-size': `${sz() * 0.22}px`,
          'font-family': 'var(--font-mono)',
          'font-weight': '600',
        }}
      >
        {Math.round(props.angle)}
      </text>
    </svg>
  );
}

// ─── Utilidades ───────────────────────────────────────────────────────────────

/** Retardo para la serialización de movimientos masivos (brownout protection). */
function delay(ms: number): Promise<void> {
  return new Promise((res) => setTimeout(res, ms));
}

/** Throttle simple: devuelve un handler que no dispara más rápido que `ms`. */
function makeThrottle<T extends unknown[]>(
  fn: (...args: T) => void,
  ms: number,
): (...args: T) => void {
  let last = 0;
  return (...args: T) => {
    const now = Date.now();
    if (now - last < ms) return;
    last = now;
    fn(...args);
  };
}

// ─── Componente de tarjeta de servo ──────────────────────────────────────────

function ServoCard(props: {
  index: number;
  pendingServo: () => number | null;
  setPendingServo: (i: number | null) => void;
}) {
  const i = props.index;
  const angle = () => angles()[i] ?? 90;
  const channel = () => channels()[i] ?? i;

  // Ángulo local del slider (se actualiza optimistamente antes de la telemetría).
  const [localAngle, setLocalAngle] = createSignal(angle());

  // Sincronizar localAngle con la telemetría cuando llega del broker.
  createEffect(() => {
    setLocalAngle(angle());
  });

  // Throttle de 35 ms para el slider (brownout protection).
  const sendMove = makeThrottle((a: number) => {
    // Bloquear si hay un movimiento de un servo DIFERENTE pendiente.
    if (props.pendingServo() !== null && props.pendingServo() !== i) return;
    props.setPendingServo(i);
    publish(cmdMove(i, a));
    // Limpiar bloqueo tras 100 ms.
    setTimeout(() => props.setPendingServo(null), 100);
  }, 35);

  function onSlider(e: Event) {
    const a = Number((e.currentTarget as HTMLInputElement).value);
    setLocalAngle(a);
    sendMove(a);
  }

  function onNumInput(e: Event) {
    const raw = Number((e.currentTarget as HTMLInputElement).value);
    const a = Math.max(0, Math.min(180, raw));
    setLocalAngle(a);
    sendMove(a);
  }

  const preset0 = () => presets()[i]?.[0] ?? 90;
  const preset1 = () => presets()[i]?.[1] ?? 90;

  return (
    <div class="border border-border rounded-md p-4 bg-bg-panel flex flex-col gap-3">
      <div class="flex items-baseline justify-between">
        <span class="font-semibold text-text-primary text-sm">Servo {i + 1}</span>
        <span class="text-xs text-text-secondary">Canal {channel()}</span>
      </div>

      {/* Gauge semicircular */}
      <div class="flex justify-center">
        <ServoGauge angle={localAngle()} size={96} />
      </div>

      {/* Slider */}
      <input
        type="range"
        min="0"
        max="180"
        value={localAngle()}
        onInput={onSlider}
        class="w-full accent-[var(--accent)] cursor-pointer"
        aria-label={`Ángulo del servo ${i + 1}`}
      />

      {/* Input numérico sincronizado */}
      <div class="flex items-center gap-2">
        <label class="text-xs text-text-secondary shrink-0">Ángulo</label>
        <input
          type="number"
          min="0"
          max="180"
          value={localAngle()}
          onInput={onNumInput}
          class="w-16 rounded border border-border bg-bg-surface px-2 py-1 text-xs text-text-primary
                 focus:outline-none focus:border-accent font-mono tabular"
          aria-label={`Valor numérico del servo ${i + 1}`}
        />
        <span class="text-xs text-text-secondary">deg</span>
      </div>

      {/* Presets */}
      <div class="grid grid-cols-2 gap-1.5">
        <button
          type="button"
          class={BTN + ' text-xs px-2 py-1'}
          aria-label={`Guardar preset 1 del servo ${i + 1}`}
          onClick={() => publish(cmdSavePreset(i, 0, localAngle()))}
        >
          Guardar 1 ({preset0()})
        </button>
        <button
          type="button"
          class={BTN + ' text-xs px-2 py-1'}
          aria-label={`Cargar preset 1 del servo ${i + 1}`}
          onClick={() => publish(cmdLoadPreset(i, 0))}
        >
          Ir a 1
        </button>
        <button
          type="button"
          class={BTN + ' text-xs px-2 py-1'}
          aria-label={`Guardar preset 2 del servo ${i + 1}`}
          onClick={() => publish(cmdSavePreset(i, 1, localAngle()))}
        >
          Guardar 2 ({preset1()})
        </button>
        <button
          type="button"
          class={BTN + ' text-xs px-2 py-1'}
          aria-label={`Cargar preset 2 del servo ${i + 1}`}
          onClick={() => publish(cmdLoadPreset(i, 1))}
        >
          Ir a 2
        </button>
      </div>
    </div>
  );
}

// ─── Componente principal ─────────────────────────────────────────────────────

export default function Control() {
  // Señal de DEVICE_ID editable (local al componente; no reactiva al store).
  const [deviceIdInput, setDeviceIdInput] = createSignal(getDeviceId());

  // Servo que tiene un movimiento pendiente (brownout protection entre tarjetas).
  const [pendingServo, setPendingServo] = createSignal<number | null>(null);

  // Flag de operación masiva en curso (deshabilita acciones globales durante la serialización).
  const [massRunning, setMassRunning] = createSignal(false);

  onCleanup(() => {
    // El cliente MQTT persiste entre rutas (instancia a nivel de modulo).
    // No hacemos destroy() aquí para no interrumpir la conexión.
  });

  // ─── Acciones masivas (serializadas, brownout protection) ──────────────────

  async function centerAll() {
    if (massRunning()) return;
    setMassRunning(true);
    const n = numServos();
    for (let i = 0; i < n; i++) {
      await delay(80 * i);
      publish(cmdMove(i, 90));
    }
    setMassRunning(false);
  }

  async function loadPresetAll(slot: number) {
    if (massRunning()) return;
    setMassRunning(true);
    const n = numServos();
    for (let i = 0; i < n; i++) {
      await delay(80 * i);
      publish(cmdLoadPreset(i, slot));
    }
    setMassRunning(false);
  }

  // ─── Guardar DEVICE_ID editado ─────────────────────────────────────────────

  function saveDeviceId() {
    const id = deviceIdInput().trim();
    if (!id) return;
    setDeviceId(id);
    // Re-suscribirse a los nuevos topics si el broker está activo.
    if (mqttStatus() === 'active') {
      mqttClient.subscribe(topicState());
      mqttClient.subscribe(topicOnline());
    }
  }

  // ─── Etiqueta del estado del broker ───────────────────────────────────────

  const brokerLabel = () => {
    const s = mqttStatus();
    if (s === 'active') return 'Broker';
    if (s === 'connecting' || s === 'reconnecting') return 'Conectando';
    return 'Desconectado';
  };

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div class="min-h-screen bg-bg-app flex flex-col">
      {/* Header */}
      <header class="flex items-center justify-between border-b border-border px-6 h-14 shrink-0">
        <div class="flex items-center gap-3">
          <A href="/" aria-label="Volver al inicio"
            class="flex items-center text-text-secondary hover:text-text-primary transition-colors">
            <IconBack />
          </A>
          <span class="font-semibold text-text-primary">Control de servos</span>
        </div>

        {/* Indicadores de conexión + ThemeToggle */}
        <div class="flex items-center gap-4">
          {/* Broker MQTT */}
          <div class="flex items-center gap-1.5">
            <StatusDot status={mqttStatus()} />
            <span class="text-xs text-text-secondary">{brokerLabel()}</span>
          </div>

          {/* ESP32 */}
          <div class="flex items-center gap-1.5">
            <Show
              when={esp32Online() === true}
              fallback={
                <div
                  class="w-2 h-2 rounded-full flex-shrink-0"
                  style={{ 'background-color': '#5a6169' }}
                  role="status"
                  aria-label="ESP32 desconectado"
                />
              }
            >
              <div
                class="w-2 h-2 rounded-full flex-shrink-0"
                style={{ 'background-color': '#10b981' }}
                role="status"
                aria-label="ESP32 en línea"
              />
            </Show>
            <span class="text-xs text-text-secondary">
              {esp32Online() === true ? 'ESP32' : 'Sin ESP32'}
            </span>
          </div>

          <ThemeToggle />
        </div>
      </header>

      {/* Cuerpo principal */}
      <main class="flex-1 flex flex-col">

        {/* Estado intermedio: conectando al broker (esp32Online === null y aún en timeout) */}
        <Show when={esp32Online() === null}>
          <div class="flex-1 flex flex-col items-center justify-center gap-3">
            <div
              class="w-5 h-5 rounded-full border-2 border-border border-t-accent"
              style={{ animation: 'spin 0.8s linear infinite' }}
            />
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
            <p class="text-sm text-text-secondary">Conectando al broker…</p>
          </div>
        </Show>

        {/* Panel de control (ESP32 online) */}
        <Show when={esp32Online() === true}>
          <div class="flex-1 flex flex-col gap-6 p-6 max-w-6xl mx-auto w-full">

            {/* Grid de tarjetas de servo */}
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <For each={Array.from({ length: numServos() })}>
                {(_, i) => (
                  <ServoCard
                    index={i()}
                    pendingServo={pendingServo}
                    setPendingServo={setPendingServo}
                  />
                )}
              </For>
            </div>

            {/* Barra de acciones globales */}
            <div class="border border-border rounded-md p-4 bg-bg-panel flex flex-col gap-4">
              <span class="text-xs font-semibold text-text-secondary uppercase tracking-wider">
                Acciones globales
              </span>

              {/* Botones de acción */}
              <div class="flex flex-wrap gap-2">
                <button
                  type="button"
                  class={BTN}
                  disabled={massRunning()}
                  aria-label="Centrar todos los servos a 90 grados"
                  onClick={() => void centerAll()}
                >
                  Centrar todos
                </button>
                <button
                  type="button"
                  class={BTN}
                  disabled={massRunning()}
                  aria-label="Cargar preset 1 en todos los servos"
                  onClick={() => void loadPresetAll(0)}
                >
                  Preset 1 en todos
                </button>
                <button
                  type="button"
                  class={BTN}
                  disabled={massRunning()}
                  aria-label="Cargar preset 2 en todos los servos"
                  onClick={() => void loadPresetAll(1)}
                >
                  Preset 2 en todos
                </button>
                <button
                  type="button"
                  class={BTN}
                  aria-label="Solicitar estado del ESP32"
                  onClick={() => requestState()}
                >
                  Solicitar estado
                </button>
              </div>

              {/* DEVICE_ID editable */}
              <div class="flex flex-col gap-1.5">
                <label class="text-xs text-text-secondary">ID del dispositivo</label>
                <div class="flex items-center gap-2">
                  <input
                    type="text"
                    value={deviceIdInput()}
                    onInput={(e) => setDeviceIdInput((e.currentTarget as HTMLInputElement).value)}
                    class="rounded border border-border bg-bg-surface px-2 py-1 text-xs text-text-primary
                           font-mono focus:outline-none focus:border-accent w-auto"
                    aria-label="DEVICE_ID del firmware"
                  />
                  <button
                    type="button"
                    class={BTN_PRIMARY + ' text-xs px-2 py-1'}
                    aria-label="Guardar DEVICE_ID"
                    onClick={saveDeviceId}
                  >
                    Guardar
                  </button>
                </div>
                <span class="font-mono text-xs text-text-secondary">
                  {topicState()}
                </span>
              </div>

              {/* Telemetría RSSI + uptime */}
              <Show when={rssi() !== null}>
                <p class="text-xs text-text-secondary tabular">
                  RSSI {rssi()} dBm · uptime {uptimeSec()} s
                </p>
              </Show>
            </div>
          </div>
        </Show>

        {/* Instructivo de flasheo (ESP32 offline o sin LWT tras timeout) */}
        {/* Se muestra cuando esp32Online() es false (no null, que muestra el spinner) */}
        <Show when={esp32Online() === false}>
          <div class="flex-1 flex flex-col items-center justify-center px-6 py-12">
            <div class="w-full max-w-xl flex flex-col gap-6">

              {/* Título */}
              <div>
                <h1 class="text-xl font-semibold text-text-primary">
                  ESP32 no encontrado
                </h1>
                <p class="mt-1 text-sm text-text-secondary">
                  No se detectó el firmware. Sigue estos pasos para flashearlo:
                </p>
              </div>

              {/* Pasos del instructivo */}
              <ol class="flex flex-col gap-4 list-none">

                {/* Paso 1: Arduino IDE y librerías */}
                <li class="flex flex-col gap-2">
                  <span class="text-sm font-medium text-text-primary">
                    1. Arduino IDE y librerías
                  </span>
                  <p class="text-xs text-text-secondary">
                    Abre el Library Manager e instala las siguientes librerías. Luego
                    agrega el soporte de placas ESP32 en el Boards Manager.
                  </p>
                  <pre class="border border-border bg-bg-surface rounded-md px-3 py-2 font-mono text-xs text-text-secondary overflow-x-auto">
                    <code>{`Adafruit PWM Servo Driver Library
PubSubClient
ArduinoJson
esp32 by Espressif (Boards Manager)`}</code>
                  </pre>
                </li>

                {/* Paso 2: secrets.h y config.h con DEVICE_ID actual */}
                <li class="flex flex-col gap-2">
                  <span class="text-sm font-medium text-text-primary">
                    2. Credenciales y DEVICE_ID
                  </span>
                  <p class="text-xs text-text-secondary">
                    Copia <code class="font-mono bg-bg-surface px-1 rounded">secrets.h.example</code> a{' '}
                    <code class="font-mono bg-bg-surface px-1 rounded">secrets.h</code> y
                    completa tu SSID y contraseña. En{' '}
                    <code class="font-mono bg-bg-surface px-1 rounded">config.h</code>{' '}
                    ajusta el <code class="font-mono bg-bg-surface px-1 rounded">DEVICE_ID</code> al
                    valor que la UI espera:
                  </p>
                  <pre class="border border-border bg-bg-surface rounded-md px-3 py-2 font-mono text-xs text-text-secondary overflow-x-auto">
                    <code>{`// firmware/esp32_servo_controller/config.h
#define DEVICE_ID  "${getDeviceId()}"
// (debe coincidir con el ID configurado en esta página)`}</code>
                  </pre>
                </li>

                {/* Paso 3: subir y verificar */}
                <li class="flex flex-col gap-2">
                  <span class="text-sm font-medium text-text-primary">
                    3. Compilar, subir y verificar
                  </span>
                  <p class="text-xs text-text-secondary">
                    Selecciona la placa <strong class="text-text-primary">ESP32 Dev Module</strong>{' '}
                    y el puerto COM correcto, luego sube el sketch. Abre el monitor serie
                    a 115200 baud: deberás ver el DEVICE_ID, la conexión Wi-Fi y el estado
                    MQTT <code class="font-mono bg-bg-surface px-1 rounded">OK</code>.
                  </p>
                  <pre class="border border-border bg-bg-surface rounded-md px-3 py-2 font-mono text-xs text-text-secondary overflow-x-auto">
                    <code>{`=== ESP32 Servo MQTT Controller ===
DEVICE_ID = ${getDeviceId()}
[WiFi] OK, IP=192.168.x.x
[MQTT] Conectando a broker.emqx.io:1883 ... OK`}</code>
                  </pre>
                </li>
              </ol>

              {/* DEVICE_ID editable en el instructivo */}
              <div class="flex flex-col gap-1.5">
                <label class="text-xs text-text-secondary">ID del dispositivo esperado</label>
                <div class="flex items-center gap-2">
                  <input
                    type="text"
                    value={deviceIdInput()}
                    onInput={(e) => setDeviceIdInput((e.currentTarget as HTMLInputElement).value)}
                    class="rounded border border-border bg-bg-surface px-2 py-1 text-xs text-text-primary
                           font-mono focus:outline-none focus:border-accent"
                    aria-label="DEVICE_ID del firmware"
                  />
                  <button
                    type="button"
                    class={BTN_PRIMARY + ' text-xs px-2 py-1'}
                    aria-label="Guardar DEVICE_ID"
                    onClick={saveDeviceId}
                  >
                    Guardar
                  </button>
                </div>
                <span class="font-mono text-xs text-text-secondary">
                  Suscrito a: {topicOnline()}
                </span>
              </div>

              {/* Botón de reintento */}
              <div class="flex items-center gap-3">
                <button
                  type="button"
                  class={BTN + ' flex items-center gap-2'}
                  aria-label="Reintentar conexión con el ESP32"
                  onClick={() => requestState()}
                >
                  <IconRefresh />
                  Reintentar conexión
                </button>

                {/* Nota de seguridad */}
                <InfoTip
                  label="Nota de seguridad del broker"
                  placement="top"
                  width="w-80"
                >
                  El broker EMQX es público. Usa un DEVICE_ID propio para no colisionar
                  con otros usuarios. Para producción, monta tu propio Mosquitto con TLS
                  y autenticación.
                </InfoTip>
              </div>
            </div>
          </div>
        </Show>

      </main>
    </div>
  );
}
