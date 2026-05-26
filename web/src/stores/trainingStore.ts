// web/src/stores/trainingStore.ts
// Estado del entrenamiento en la nube (Vast.ai vía SkyPilot) leído desde el proxy
// serverless /api/training-status, que a su vez lee los artefactos que el
// notebook publica en HF Hub vía CommitScheduler:
//   - runs/heartbeat.jsonl          progreso en vivo (epoch, loss, eta, ...)
//   - manifests/manifest.json       existe => el run terminó
//   - manifests/eval_summary.json   métricas finales sobre el split de validación
// El HF_TOKEN vive solo en el servidor (Vercel); el cliente nunca lo ve.
//
// Polling bajo demanda: solo sondeamos mientras la ruta de monitoreo está montada
// (startTrainingPolling/stopTrainingPolling con refcount). El entrenamiento es
// esporádico; no tiene sentido golpear la función serverless cada 30 s cuando
// nadie está mirando. Mismo criterio "sondea solo si hace falta" que nanoStore.

import { createSignal } from 'solid-js';

// Una línea de heartbeat.jsonl (el notebook escribe una cada 30 s).
export interface Heartbeat {
  ts?: string;            // ISO UTC del tick
  epoch?: number;
  total_epochs?: number;
  loss?: number;
  grad_norm?: number | null;
  lr?: number;
  gpu_mem_mb?: number;
  elapsed_s?: number;     // segundos desde que arrancó el entrenamiento
  eta_s?: number;         // estimación de segundos restantes
}

export interface EvalPerClass {
  precision?: number;
  recall?: number;
  AP50?: number;
}

// manifests/eval_summary.json — métricas finales (split de validación, holdout 15%).
export interface EvalSummary {
  mAP50?: number;
  mAP50_95?: number;
  precision_mean?: number;
  recall_mean?: number;
  per_class?: Record<string, EvalPerClass>;
}

// Respuesta del proxy /api/training-status.
export interface TrainingStatus {
  ok: boolean;
  running?: boolean;          // hay heartbeat y aún no terminó
  done?: boolean;             // manifest.json presente => run completado
  latest?: Heartbeat | null;  // última línea válida del heartbeat
  manifest?: unknown;         // manifest.json crudo (solo se usa su presencia)
  eval?: EvalSummary | null;  // métricas finales (cuando ya evaluó)
  ts?: number;                // timestamp del proxy
  error?: string;
}

// Estado de la conexión con el proxy (distinto del estado del entrenamiento).
export type TrainingConn = 'idle' | 'loading' | 'ok' | 'error';

export const [trainingStatus, setTrainingStatus] = createSignal<TrainingStatus | null>(null);
export const [trainingConn, setTrainingConn]     = createSignal<TrainingConn>('idle');
export const [lastPollAt, setLastPollAt]         = createSignal<number | null>(null);

const ENDPOINT = '/api/training-status';

/** Un sondeo al proxy. Timeout de 8 s para no acumular requests colgados. */
export async function pollTrainingStatus(): Promise<void> {
  const ctrl  = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 8000);
  if (trainingConn() === 'idle') setTrainingConn('loading');
  try {
    const r = await fetch(ENDPOINT, { signal: ctrl.signal, cache: 'no-store' });
    if (!r.ok) { setTrainingConn('error'); return; }
    const data = (await r.json()) as TrainingStatus;
    setTrainingStatus(data);
    setTrainingConn(data.ok ? 'ok' : 'error');
    setLastPollAt(Date.now());
  } catch {
    // Red caída, o entorno sin la función serverless (dev local sin Vercel).
    setTrainingConn('error');
  } finally {
    clearTimeout(timer);
  }
}

// Polling compartido por refcount: varias vistas pueden suscribirse y el timer
// se apaga cuando la última se desmonta.
let timer: ReturnType<typeof setInterval> | null = null;
let watchers = 0;

/** Arranca (o se suscribe a) el polling. Sondea de inmediato al montar. */
export function startTrainingPolling(intervalMs = 30000): void {
  watchers++;
  void pollTrainingStatus();
  if (!timer) timer = setInterval(() => void pollTrainingStatus(), intervalMs);
}

/** Cancela una suscripción; detiene el timer cuando ya nadie observa. */
export function stopTrainingPolling(): void {
  watchers = Math.max(0, watchers - 1);
  if (watchers === 0 && timer) { clearInterval(timer); timer = null; }
}
