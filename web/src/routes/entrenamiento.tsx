// web/src/routes/entrenamiento.tsx
// Monitor del entrenamiento en la nube (Fase 6 / track de automatización). Lee el
// proxy serverless /api/training-status (Vercel Edge), que reenvía los artefactos
// que el notebook publica en HF Hub mientras entrena en Vast.ai (orquestado por
// SkyPilot). NO toca el Nano ni el WS de inferencia: es una vista de solo lectura.
//
// El notebook publica vía CommitScheduler cada ~10 min, así que el primer reporte
// puede tardar. El polling es bajo demanda (trainingStore con refcount): solo
// sondea mientras esta ruta está montada.

import { onMount, onCleanup, For, Show, type JSX } from 'solid-js';
import { A } from '@solidjs/router';
import { ThemeToggle } from '../components/ThemeToggle';
import { StatusDot, type DotStatus } from '../components/StatusDot';
import {
  trainingStatus, trainingConn, lastPollAt,
  startTrainingPolling, stopTrainingPolling,
  type EvalPerClass,
} from '../stores/trainingStore';

// Las 4 clases del modelo v1d (el cartón se agregó en v1d; el resto del frontend
// aún asume 3, así que este monitor mantiene su propio mapa de etiquetas/colores).
const CLASS_LABELS: Record<string, string> = {
  glass: 'Vidrio', paper: 'Papel', plastic: 'Plástico', cardboard: 'Cartón',
};
const CLASS_COLORS: Record<string, string> = {
  glass: '#56b4e9', paper: '#f0e442', plastic: '#009e73', cardboard: '#e69f00',
};

// ─── Formato (es-CO: coma decimal) ───────────────────────────────────────────
const NF = (d: number) => new Intl.NumberFormat('es-CO', { minimumFractionDigits: d, maximumFractionDigits: d });
function fmtNum(v: number | null | undefined, d = 3): string {
  return v == null || isNaN(v) ? '—' : NF(d).format(v);
}
function fmtPct(v: number | null | undefined, d = 1): string {
  return v == null || isNaN(v) ? '—' : `${NF(d).format(v * 100)} %`;
}
function fmtLr(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return '—';
  return v >= 0.001 ? NF(4).format(v) : v.toExponential(2).replace('.', ',');
}
function fmtMem(mb: number | null | undefined): string {
  return mb == null || isNaN(mb) ? '—' : `${NF(1).format(mb / 1024)} GB`;
}
function fmtDuration(s?: number | null): string {
  if (s == null || isNaN(s)) return '—';
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s} s`;
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  if (h) return `${h} h ${m} min`;
  return r ? `${m} min ${r} s` : `${m} min`;
}
function fmtRelative(ms?: number | null): string {
  if (ms == null) return '';
  const s = Math.floor((Date.now() - ms) / 1000);
  if (s < 5) return 'recién';
  if (s < 60) return `hace ${s} s`;
  if (s < 3600) return `hace ${Math.floor(s / 60)} min`;
  return `hace ${Math.floor(s / 3600)} h`;
}
function fmtIsoRelative(iso?: string): string {
  if (!iso) return '';
  const t = Date.parse(iso);
  return isNaN(t) ? '' : fmtRelative(t);
}

// ─── Iconos (Lucide, trazo currentColor) ─────────────────────────────────────
function Icon(props: { children: JSX.Element; size?: number }) {
  return (
    <svg width={props.size ?? 16} height={props.size ?? 16} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      {props.children}
    </svg>
  );
}
const IconBack     = () => <Icon><path d="m15 18-6-6 6-6" /></Icon>;
const IconCloudCpu = () => <Icon><path d="M17.5 19a4.5 4.5 0 1 0-1.5-8.74A6 6 0 1 0 6 17" /><rect width="6" height="6" x="9" y="12" rx="1" /></Icon>;
const IconActivity = () => <Icon><path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2" /></Icon>;
const IconTarget   = () => <Icon><circle cx="12" cy="12" r="10" /><circle cx="12" cy="12" r="6" /><circle cx="12" cy="12" r="2" /></Icon>;
const IconInfo     = () => <Icon><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></Icon>;

// Estado del entrenamiento (distinto de la conexión con el proxy).
type TrainState = 'running' | 'done' | 'waiting' | 'offline';

const STATE_META: Record<TrainState, { label: string; color: string }> = {
  running: { label: 'entrenando',                color: 'var(--accent)' },
  done:    { label: 'completado',                color: '#16a34a' },
  waiting: { label: 'esperando primer reporte',  color: '#e69f00' },
  offline: { label: 'sin conexión',              color: '#6b7280' },
};

export default function Entrenamiento() {
  onMount(() => startTrainingPolling(30000));
  onCleanup(() => stopTrainingPolling());

  const st       = () => trainingStatus();
  const latest   = () => st()?.latest ?? null;
  const evalS    = () => st()?.eval ?? null;
  const conn     = () => trainingConn();

  // Máquina de estados: la conexión manda solo si aún no tenemos datos.
  const state = (): TrainState => {
    const s = st();
    if (s?.done) return 'done';
    if (s?.running && latest()) return 'running';
    if (conn() === 'error' && !s) return 'offline';
    return 'waiting';
  };

  const connDot = (): DotStatus =>
    conn() === 'ok' ? 'active' : conn() === 'loading' ? 'connecting' : conn() === 'error' ? 'closed' : 'reconnecting';

  const epoch  = () => latest()?.epoch ?? null;
  const total  = () => latest()?.total_epochs ?? null;
  const pct     = () => {
    const e = epoch(), t = total();
    return e != null && t ? Math.min(100, Math.round((e / t) * 100)) : 0;
  };

  // per_class -> filas ordenadas por el orden canónico de clases.
  const perClassRows = (): Array<{ name: string } & EvalPerClass> => {
    const pc = evalS()?.per_class;
    if (!pc) return [];
    const order = ['glass', 'paper', 'plastic', 'cardboard'];
    const keys = Object.keys(pc).sort((a, b) => order.indexOf(a) - order.indexOf(b));
    return keys.map((k) => ({ name: k, ...pc[k] }));
  };

  return (
    <div class="h-screen overflow-hidden bg-bg-app flex flex-col">
      <header class="flex items-center justify-between border-b border-border px-6 h-14 gap-4 shrink-0">
        <div class="flex items-center gap-3 min-w-0">
          <A href="/" aria-label="Volver al inicio"
            class="flex items-center text-text-secondary hover:text-text-primary transition-colors">
            <IconBack />
          </A>
          <span class="font-semibold text-text-primary">Entrenamiento en la nube</span>
        </div>
        <div class="flex items-center gap-4">
          <span class="flex items-center gap-2 font-mono text-xs text-text-secondary">
            <StatusDot status={connDot()} />
            <span>{lastPollAt() ? `actualizado ${fmtRelative(lastPollAt())}` : 'conectando'}</span>
          </span>
          <ThemeToggle />
        </div>
      </header>

      <div class="flex-1 flex min-h-0">
        {/* Principal: estado + progreso + métricas */}
        <section class="flex-1 min-w-0 overflow-y-auto">
          <div class="px-6 py-5 border-b border-border">
            <h2 class="flex items-center gap-2 mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary">
              <IconCloudCpu /> Estado del entrenamiento
            </h2>

            <div class="flex items-center gap-3 mb-4">
              <span class="inline-flex items-center gap-2 rounded-md px-2.5 py-1 text-sm"
                style={{ 'background-color': `color-mix(in srgb, ${STATE_META[state()].color} 13%, transparent)`,
                         color: STATE_META[state()].color }}>
                <span class="w-2 h-2 rounded-full" classList={{ 'animate-pulse': state() === 'running' }}
                  style={{ 'background-color': STATE_META[state()].color }} />
                {STATE_META[state()].label}
              </span>
              <Show when={latest()?.ts}>
                <span class="font-mono text-xs text-text-secondary">último reporte {fmtIsoRelative(latest()!.ts)}</span>
              </Show>
            </div>

            {/* Barra de progreso por épocas (solo con datos de heartbeat). */}
            <Show when={latest() && total()}>
              <div class="mb-4 max-w-2xl">
                <div class="flex items-center justify-between mb-1.5 text-xs">
                  <span class="text-text-secondary">
                    época <span class="font-mono tabular text-text-primary">{epoch()}</span> de{' '}
                    <span class="font-mono tabular text-text-primary">{total()}</span>
                  </span>
                  <span class="font-mono tabular text-text-primary">{pct()}%</span>
                </div>
                <div class="h-1.5 w-full rounded-full bg-bg-surface overflow-hidden">
                  <div class="h-full transition-[width] duration-500"
                    style={{ width: `${state() === 'done' ? 100 : pct()}%`,
                             'background-color': STATE_META[state()].color }} />
                </div>
              </div>
            </Show>

            {/* Mensaje guía cuando aún no hay datos. */}
            <Show when={state() === 'waiting'}>
              <p class="max-w-2xl text-sm text-text-secondary leading-relaxed">
                El notebook publica su progreso en HF Hub cada ~10 minutos, así que el primer
                reporte puede tardar tras lanzar el entrenamiento. Esta vista se actualiza sola.
              </p>
            </Show>
            <Show when={state() === 'offline'}>
              <p class="max-w-2xl text-sm text-text-secondary leading-relaxed">
                No se pudo contactar el proxy de estado. Verificá que <code class="font-mono">/api/training-status</code>
                {' '}esté desplegado y que el servidor tenga configurada la variable <code class="font-mono">HF_TOKEN</code>.
              </p>
            </Show>
          </div>

          {/* Métricas en vivo (heartbeat). */}
          <Show when={latest()}>
            <div class="px-6 py-5 border-b border-border">
              <h2 class="flex items-center gap-2 mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary">
                <IconActivity /> Métricas en vivo
              </h2>
              <dl class="grid grid-cols-2 sm:grid-cols-3 gap-x-8 gap-y-4 max-w-2xl">
                <Metric label="Pérdida"        value={fmtNum(latest()!.loss, 4)} />
                <Metric label="ETA"            value={fmtDuration(latest()!.eta_s)} />
                <Metric label="Transcurrido"   value={fmtDuration(latest()!.elapsed_s)} />
                <Metric label="Learning rate"  value={fmtLr(latest()!.lr)} />
                <Metric label="Grad norm"      value={fmtNum(latest()!.grad_norm ?? null, 3)} />
                <Metric label="Memoria GPU"    value={fmtMem(latest()!.gpu_mem_mb)} />
              </dl>
            </div>
          </Show>

          {/* Métricas finales (eval_summary, al terminar). */}
          <Show when={evalS()}>
            <div class="px-6 py-5">
              <h2 class="flex items-center gap-2 mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary">
                <IconTarget /> Evaluación (split de validación)
              </h2>
              <dl class="grid grid-cols-2 sm:grid-cols-4 gap-x-8 gap-y-4 max-w-2xl mb-5">
                <Metric label="mAP@50"     value={fmtPct(evalS()!.mAP50)} accent />
                <Metric label="mAP@50-95"  value={fmtPct(evalS()!.mAP50_95)} accent />
                <Metric label="Precisión"  value={fmtPct(evalS()!.precision_mean)} />
                <Metric label="Recall"     value={fmtPct(evalS()!.recall_mean)} />
              </dl>

              <Show when={perClassRows().length}>
                <table class="w-full max-w-2xl text-sm">
                  <thead>
                    <tr class="text-left text-[11px] uppercase tracking-wider text-text-secondary">
                      <th class="font-medium pb-2">Clase</th>
                      <th class="font-medium pb-2 text-right tabular">AP@50</th>
                      <th class="font-medium pb-2 text-right tabular">Precisión</th>
                      <th class="font-medium pb-2 text-right tabular">Recall</th>
                    </tr>
                  </thead>
                  <tbody>
                    <For each={perClassRows()}>{(row) => (
                      <tr class="border-t border-border">
                        <td class="py-2">
                          <span class="flex items-center gap-2 text-text-primary">
                            <span class="w-2 h-2 rounded-full shrink-0"
                              style={{ 'background-color': CLASS_COLORS[row.name] ?? '#6b7280' }} />
                            {CLASS_LABELS[row.name] ?? row.name}
                          </span>
                        </td>
                        <td class="py-2 text-right font-mono tabular text-text-primary">{fmtPct(row.AP50)}</td>
                        <td class="py-2 text-right font-mono tabular text-text-secondary">{fmtPct(row.precision)}</td>
                        <td class="py-2 text-right font-mono tabular text-text-secondary">{fmtPct(row.recall)}</td>
                      </tr>
                    )}</For>
                  </tbody>
                </table>
              </Show>
            </div>
          </Show>
        </section>

        {/* Panel lateral: contexto del run */}
        <aside class="w-80 shrink-0 border-l border-border bg-bg-panel overflow-y-auto">
          <Section title="Detalles del run" icon={IconInfo}>
            <dl class="space-y-3">
              <Field label="Modelo" value="YOLOv8n · 4 clases (v1d)" />
              <Field label="Repo HF" value="mitgar14/embebidos-3-models-v1d" mono />
              <Field label="Dataset" value="embebidos3-dataset-v1d · 163 imágenes" />
              <Field label="Infra" value="Vast.ai · RTX 4090 (SkyPilot)" />
            </dl>
          </Section>
          <Section title="Clases" icon={IconTarget}>
            <div class="flex flex-wrap gap-x-4 gap-y-2">
              <For each={Object.keys(CLASS_LABELS)}>{(c) => (
                <span class="flex items-center gap-1.5 text-xs text-text-secondary">
                  <span class="w-2 h-2 rounded-full" style={{ 'background-color': CLASS_COLORS[c] }} />
                  {CLASS_LABELS[c]}
                </span>
              )}</For>
            </div>
          </Section>
          <Section title="Persistencia" icon={IconCloudCpu}>
            <p class="text-xs text-text-secondary leading-relaxed">
              El notebook empuja <code class="font-mono">runs/</code>, <code class="font-mono">manifests/</code> y
              {' '}<code class="font-mono">exports/</code> a HF Hub cada ~10 min. Al terminar, SkyPilot destruye la
              instancia (autodown) y el <code class="font-mono">best.onnx</code> queda listo para el Engine del Nano.
            </p>
          </Section>
        </aside>
      </div>
    </div>
  );
}

// ─── Subcomponentes ──────────────────────────────────────────────────────────
function Section(props: { title: string; icon: () => JSX.Element; children: JSX.Element }) {
  return (
    <section class="border-t border-border first:border-t-0 p-4">
      <h2 class="flex items-center gap-2 mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary">
        <props.icon /> {props.title}
      </h2>
      {props.children}
    </section>
  );
}

function Metric(props: { label: string; value: string; accent?: boolean }) {
  return (
    <div>
      <dt class="text-xs text-text-secondary">{props.label}</dt>
      <dd class="mt-0.5">
        <b class="font-mono tabular text-lg font-semibold"
          classList={{ 'text-text-primary': !props.accent, 'text-accent': props.accent }}>
          {props.value}
        </b>
      </dd>
    </div>
  );
}

function Field(props: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt class="text-xs text-text-secondary">{props.label}</dt>
      <dd class={`mt-0.5 text-sm text-text-primary ${props.mono ? 'font-mono break-all' : ''}`}>{props.value}</dd>
    </div>
  );
}
