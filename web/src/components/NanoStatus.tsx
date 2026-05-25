// web/src/components/NanoStatus.tsx
// HUB-02: estado del Nano inline (conexión, salud y última inferencia), sin
// que el usuario tenga que hacer nada. Cluster compacto en mono, dots/texto;
// sin cards ni border-left. La conexión la pinta StatusDot; salud e inferencia
// se derivan de wsStatus + lastMessageAt (el sondeo /health solo refina cuando
// el WS no está activo, ver nanoStore).

import { createSignal, onCleanup } from 'solid-js';
import { StatusDot } from './StatusDot';
import { wsStatus } from '../stores/wsStore';
import { nanoHealth, lastMessageAt } from '../stores/nanoStore';

const CONN_LABELS: Record<string, string> = {
  connecting:   'conectando',
  active:       'conectado',
  reconnecting: 'reconectando',
  closed:       'sin conexión',
};

/** Tiempo transcurrido en formato corto: "hace 3s" / "hace 2m" / "sin datos". */
function desdeInferencia(ts: number | null, now: number): string {
  if (ts == null) return 'sin datos';
  const s = Math.max(0, Math.round((now - ts) / 1000));
  if (s < 1)  return 'ahora';
  if (s < 60) return `hace ${s}s`;
  return `hace ${Math.floor(s / 60)}m`;
}

export function NanoStatus() {
  // Tick de 1s para refrescar el "hace Xs" sin depender de nuevos mensajes.
  const [now, setNow] = createSignal(Date.now());
  const id = setInterval(() => setNow(Date.now()), 1000);
  onCleanup(() => clearInterval(id));

  // Salud: el WS activo ya prueba que el Nano vive; nunca mostramos "caída" si
  // el WS está conectado (evita falsos negativos por CORS en el sondeo /health).
  const salud = () => {
    if (wsStatus() === 'active') return 'operativo';
    if (nanoHealth() === 'ok')   return 'operativo';
    if (nanoHealth() === 'down') return 'caída';
    return '—';
  };

  return (
    <div class="flex items-center gap-2.5 font-mono text-xs text-text-secondary">
      <span class="flex items-center gap-1.5">
        <StatusDot status={wsStatus()} />
        <span>{CONN_LABELS[wsStatus()] ?? wsStatus()}</span>
      </span>
      <span class="text-border" aria-hidden="true">·</span>
      <span>salud {salud()}</span>
      <span class="text-border" aria-hidden="true">·</span>
      <span>inferencia {desdeInferencia(lastMessageAt(), now())}</span>
    </div>
  );
}
