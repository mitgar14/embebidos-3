// web/src/lib/servoProtocol.ts
// Protocolo MQTT para los servos del ESP32.
// Gestiona el DEVICE_ID persistido en localStorage y los topics derivados.
// Exporta las funciones de comando JSON que se publican en topicCmd().

// ─── DEVICE_ID ────────────────────────────────────────────────────────────────

const DEVICE_ID_KEY = 'servo_device_id';

/** ID largo y propio del proyecto para no colisionar en el broker público EMQX.
 *  El firmware tiene 'ece88b61'; aquí usamos un ID más descriptivo y privado. */
const DEFAULT_DEVICE_ID = 'tiny-trash-esp32-ctrl';

export function getDeviceId(): string {
  return localStorage.getItem(DEVICE_ID_KEY) ?? DEFAULT_DEVICE_ID;
}

export function setDeviceId(id: string): void {
  localStorage.setItem(DEVICE_ID_KEY, id);
}

// ─── Topics ───────────────────────────────────────────────────────────────────

/** Topic de comandos (publicar desde el navegador). */
export function topicCmd(): string {
  return `servos/${getDeviceId()}/cmd`;
}

/** Topic de telemetría del ESP32 (suscribir, retained). */
export function topicState(): string {
  return `servos/${getDeviceId()}/state`;
}

/** Topic de presencia del ESP32 (suscribir, Last Will retenido: "online"/"offline"). */
export function topicOnline(): string {
  return `servos/${getDeviceId()}/online`;
}

// ─── Comandos JSON ────────────────────────────────────────────────────────────

/** Mover un servo a un ángulo (0..180). */
export function cmdMove(servo: number, angle: number): string {
  return JSON.stringify({ action: 'move', servo, angle });
}

/** Mover todos los servos a la vez (usar solo serializado para evitar brownout). */
export function cmdMoveAll(angles: number[]): string {
  return JSON.stringify({ action: 'move_all', angles });
}

/** Guardar la posición actual como preset. */
export function cmdSavePreset(servo: number, slot: number, angle: number): string {
  return JSON.stringify({ action: 'save_preset', servo, slot, angle });
}

/** Cargar un preset en un servo. */
export function cmdLoadPreset(servo: number, slot: number): string {
  return JSON.stringify({ action: 'load_preset', servo, slot });
}

/** Cargar un preset en todos los servos (serializar llamadas para evitar brownout). */
export function cmdLoadAllPreset(slot: number): string {
  return JSON.stringify({ action: 'load_all_preset', slot });
}

/** Pedir al ESP32 que publique un snapshot completo de su estado. */
export function cmdRequestState(): string {
  return JSON.stringify({ action: 'request_state' });
}
