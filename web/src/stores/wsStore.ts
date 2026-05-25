// web/src/stores/wsStore.ts
// Instancia global del cliente WebSocket reconectante.
// Creada a nivel de módulo (fuera del árbol reactivo de SolidJS) para que
// sobreviva re-renders y no se destruya ni recree al navegar entre rutas.

import { createSignal } from 'solid-js';
import { ReconnectingWebSocket, getWsUrl, type WSStatus } from '../lib/ws';

/** Signal reactivo que refleja el estado actual de la conexión WS. */
export const [wsStatus, setWsStatus] = createSignal<WSStatus>('connecting');

/** Instancia global del cliente WS — única en toda la aplicación. */
export const ws = new ReconnectingWebSocket(getWsUrl(), {
  onStatusChange: setWsStatus,
});
