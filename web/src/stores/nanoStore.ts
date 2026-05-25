// web/src/stores/nanoStore.ts
// Estado del Nano para mostrar inline en el hub (HUB-02).
// La conexión la provee wsStore (wsStatus); acá agregamos:
//  - lastMessageAt: timestamp del último mensaje no-pong del WS (frame/detección
//    del Nano), o sea la última inferencia recibida.
//  - nanoHealth: sondeo de /health, solo cuando el WS NO está activo (cuando lo
//    está, la propia conexión ya prueba que el Nano vive; así evitamos ruido de
//    CORS en el caso normal de la demo).

import { createSignal } from 'solid-js';
import { ws, wsStatus } from './wsStore';
import { getWsUrl } from '../lib/ws';

export type NanoHealth = 'ok' | 'down' | 'unknown';

export const [nanoHealth, setNanoHealth]       = createSignal<NanoHealth>('unknown');
export const [lastMessageAt, setLastMessageAt] = createSignal<number | null>(null);

// Cada mensaje no-pong del WS es un frame/detección: marca la última inferencia.
ws.addEventListener('message', () => setLastMessageAt(Date.now()));

/** Deriva la URL HTTP de /health desde la URL WS configurada (ws->http, wss->https). */
function healthUrl(): string {
  try {
    const u = new URL(getWsUrl());
    return `${u.protocol === 'wss:' ? 'https:' : 'http:'}//${u.host}/health`;
  } catch {
    return 'http://100.64.0.2:8000/health';
  }
}

async function probeHealth(): Promise<void> {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 3000);
  try {
    const r = await fetch(healthUrl(), { signal: ctrl.signal });
    setNanoHealth(r.ok ? 'ok' : 'down');
  } catch {
    setNanoHealth('down');
  } finally {
    clearTimeout(timer);
  }
}

// Solo sondeamos cuando el WS no está activo (necesitamos el dato extra solo si
// la conexión no nos lo da). Cuando el WS está activo, lo dejamos quieto.
setInterval(() => { if (wsStatus() !== 'active') probeHealth(); }, 8000);
probeHealth();
