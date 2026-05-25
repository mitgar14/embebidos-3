// web/src/stores/servoStore.ts
// Estado reactivo de los servos del ESP32.
// Instancia a nivel de módulo (igual que wsStore.ts): una sola conexión MQTT
// que sobrevive re-renders y navegación entre rutas.

import { createSignal } from 'solid-js';
import { MqttClient, type MqttStatus } from '../lib/mqtt';
import {
  topicState,
  topicOnline,
  topicCmd,
  cmdRequestState,
} from '../lib/servoProtocol';

// ─── Signals de estado ────────────────────────────────────────────────────────

/** Estado de la conexión al broker MQTT. */
export const [mqttStatus, setMqttStatus] = createSignal<MqttStatus>('connecting');

/**
 * Estado del ESP32 según el Last Will retenido del topic online:
 *   null  = sin información aún (timeout de 5 s no cumplido)
 *   true  = online (firmware activo y conectado al broker)
 *   false = offline (firmware desconectado o timeout alcanzado)
 */
export const [esp32Online, setEsp32Online] = createSignal<boolean | null>(null);

/** Número de servos activos según la telemetría. */
export const [numServos, setNumServos] = createSignal<number>(0);

/** Canales PCA9685 activos (ej. [0, 7, 8, 15]). */
export const [channels, setChannels] = createSignal<number[]>([]);

/** Ángulos actuales de cada servo (grados, 0..180). */
export const [angles, setAngles] = createSignal<number[]>([]);

/** Presets guardados en la NVS del ESP32: presets()[servo][slot] = ángulo. */
export const [presets, setPresets] = createSignal<number[][]>([]);

/** Potencia de la señal Wi-Fi del ESP32 en dBm, o null si aún no llegó. */
export const [rssi, setRssi] = createSignal<number | null>(null);

/** Tiempo activo del ESP32 en segundos, o null si aún no llegó. */
export const [uptimeSec, setUptimeSec] = createSignal<number | null>(null);

// ─── Cliente MQTT (instancia única a nivel de módulo) ─────────────────────────

export const mqttClient = new MqttClient(
  'wss://broker.emqx.io:8084/mqtt',
  (s) => setMqttStatus(s),
);

// ─── Timeout de presencia (5 s) ───────────────────────────────────────────────
// Si tras 5 s desde la primera conexión no llegó ningún mensaje en el topic
// online, consideramos que el ESP32 no está conectado.

let onlineResolved = false;

const presenceTimer = setTimeout(() => {
  if (!onlineResolved) {
    setEsp32Online(false);
  }
}, 5000);

// ─── Suscripciones y handlers ─────────────────────────────────────────────────

// Suscribir DENTRO de 'connect' garantiza la re-suscripción en cada reconexión
// automática (clean: true borra las suscripciones al desconectarse).
mqttClient.on('message', (topic: string, payload: Buffer) => {
  if (topic === topicOnline()) {
    const val = payload.toString();
    onlineResolved = true;
    clearTimeout(presenceTimer);
    setEsp32Online(val === 'online');
    return;
  }

  if (topic === topicState()) {
    try {
      const data = JSON.parse(payload.toString()) as {
        num_servos?: number;
        channels?: number[];
        angles?: number[];
        presets?: number[][];
        rssi?: number;
        uptime_s?: number;
      };
      if (typeof data.num_servos === 'number') setNumServos(data.num_servos);
      if (Array.isArray(data.channels))        setChannels(data.channels);
      if (Array.isArray(data.angles))          setAngles(data.angles);
      if (Array.isArray(data.presets))         setPresets(data.presets);
      if (typeof data.rssi === 'number')       setRssi(data.rssi);
      if (typeof data.uptime_s === 'number')   setUptimeSec(data.uptime_s);
    } catch {
      // Payload malformado: ignorar silenciosamente.
    }
  }
});

// Suscribirse a los topics en cada (re)conexión al broker.
// El tipo de 'connect' no es parte de la interfaz simplificada MqttClient,
// pero el cliente subyacente mqtt.js lo emite; accedemos via la instancia.
(mqttClient as unknown as { _client: { on: (e: string, cb: () => void) => void } })
  ._client.on('connect', () => {
    mqttClient.subscribe(topicState());
    mqttClient.subscribe(topicOnline());
  });

// ─── Función de publicación de comandos ───────────────────────────────────────

/**
 * Publica un comando JSON en el topic cmd del ESP32.
 * La página de control importa esta función; no expone topicCmd directamente.
 */
export function publish(payload: string): void {
  mqttClient.publish(topicCmd(), payload);
}

/**
 * Fuerza una nueva solicitud de estado al ESP32.
 * Útil para el botón "Reintentar conexión" cuando el broker ya está activo.
 */
export function requestState(): void {
  if (mqttStatus() === 'active') {
    publish(cmdRequestState());
  }
}
