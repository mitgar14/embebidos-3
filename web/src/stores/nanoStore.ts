// web/src/stores/nanoStore.ts
// Estado del Nano compartido entre el hub (HUB-02) y el Dashboard (DASH-02).
//  - lastMessageAt: timestamp del último mensaje no-pong del WS (frame/detección
//    del Nano), o sea la última inferencia recibida.
//  - nanoHealth / gpuTempC / ramMb: sondeo de /health solo con el WS activo. El
//    Nano expone CORS (el dashboard previo ya hacía fetch desde el browser);
//    sirve la temperatura de GPU y la RAM libre al Dashboard. No sondeamos con
//    el WS caído: ya sabemos que está inalcanzable y evitamos ruido en consola.

import { createSignal } from 'solid-js';
import { ws, wsStatus } from './wsStore';
import { getWsUrl } from '../lib/ws';

export type NanoHealth = 'ok' | 'down' | 'unknown';

export const [nanoHealth, setNanoHealth]       = createSignal<NanoHealth>('unknown');
export const [gpuTempC, setGpuTempC]           = createSignal<number | null>(null);
export const [ramMb, setRamMb]                 = createSignal<number | null>(null);
export const [lastMessageAt, setLastMessageAt] = createSignal<number | null>(null);

// Cada mensaje no-pong del WS es un frame/detección: marca la última inferencia.
ws.addEventListener('message', () => setLastMessageAt(Date.now()));

/** URL base HTTP del Nano derivada de la URL WS configurada (ws->http, wss->https, sin /ws). */
export function nanoHttpBase(): string {
  try {
    const u = new URL(getWsUrl());
    return `${u.protocol === 'wss:' ? 'https:' : 'http:'}//${u.host}`;
  } catch {
    return 'http://100.64.0.2:8000';
  }
}

/** Deriva la URL HTTP de /health desde la URL WS configurada. */
function healthUrl(): string {
  return `${nanoHttpBase()}/health`;
}

async function probeHealth(): Promise<void> {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 3000);
  try {
    const r = await fetch(healthUrl(), { signal: ctrl.signal });
    if (!r.ok) { setNanoHealth('down'); return; }
    setNanoHealth('ok');
    const j = await r.json().catch(() => null);
    if (j) {
      if (typeof j.gpu_temp_c === 'number')       setGpuTempC(j.gpu_temp_c);
      if (typeof j.ram_available_mb === 'number') setRamMb(j.ram_available_mb);
    }
  } catch {
    setNanoHealth('down');
  } finally {
    clearTimeout(timer);
  }
}

// Sondeamos /health SOLO con el WS activo: si el WS conecta, el Nano es
// alcanzable y /health responde sin error; si está caído, ya lo sabemos por
// wsStatus y golpear /health solo generaría errores de red en consola
// (ERR_CONNECTION_TIMED_OUT no es capturable por JS). Cadencia 3s como el
// dashboard previo; además sondeamos al instante en cada (re)conexión.
ws.addEventListener('open', () => { void probeHealth(); });
setInterval(() => { if (wsStatus() === 'active') void probeHealth(); }, 3000);
