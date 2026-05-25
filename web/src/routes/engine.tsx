// web/src/routes/engine.tsx
// Centro del Engine del Modelo (Fase 4, ENG-01..03). Consume el backend FastAPI
// del Nano por HTTP + SSE (no toca el WS de inferencia):
//   GET    /model/state           estado del modelo (poll cada 3 s)
//   POST   /model/build {force}   descarga ONNX (HF) + compila engine TensorRT
//   GET    /jobs/{id}/logs (SSE)  logs del build en vivo (eventos 'log' / 'done')
//   DELETE /jobs/{id}             cancela el build en curso
//   POST   /model/check-updates   compara ONNX local vs HF Hub
//   GET    /jobs?limit=N          historial de engines
//   POST   /model/rollback        revierte al engine previo
//
// Logs: buffer NO reactivo + flush por rAF a un <pre> imperativo (perf, igual que
// el frame del Dashboard), con auto-scroll que se pausa si el usuario sube y
// coloreo heurístico de líneas de error/warning. La conexión la refleja wsStatus
// (mismo Nano/puerto); si /model/state no responde, mostramos un aviso.

import { createSignal, onMount, onCleanup, For, Show, type JSX } from 'solid-js';
import { A } from '@solidjs/router';
import { ThemeToggle } from '../components/ThemeToggle';
import { StatusDot } from '../components/StatusDot';
import { ws, wsStatus } from '../stores/wsStore';
import { nanoHttpBase } from '../stores/nanoStore';
import { WONG, CLASS_LABEL_ES, type ClsName } from '../lib/detection';

const CONN_LABELS: Record<string, string> = {
  connecting: 'conectando', active: 'conectado', reconnecting: 'reconectando', closed: 'sin conexión',
};

const BTN = 'rounded-md border border-border px-3 py-1.5 text-sm text-text-primary ' +
  'hover:border-accent hover:bg-bg-surface transition-colors ' +
  'disabled:opacity-40 disabled:hover:border-border disabled:hover:bg-transparent';
const BTN_GHOST = 'rounded-md px-3 py-1.5 text-sm text-text-secondary ' +
  'hover:text-text-primary transition-colors disabled:opacity-40';
const BTN_PRIMARY = 'rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white ' +
  'hover:opacity-90 transition-opacity disabled:opacity-40 disabled:hover:opacity-40';

type EngineMeta = {
  engine_sha256?: string; onnx_sha256?: string; hf_revision?: string;
  hf_commit_date?: string; build_completed_at?: string; build_duration_s?: number;
  trtexec_args?: string[]; adopted?: boolean; from_fallback?: boolean;
};
type ActiveJob = { job_id?: string; phase?: string; progress_pct?: number; started_at?: string };
type ModelStateT = {
  state: 'no_model' | 'ready' | 'building' | 'degraded' | 'update_available';
  active_engine?: EngineMeta | null; previous_engine?: EngineMeta | null;
  active_job?: ActiveJob | null; engine_binary_present?: boolean; hf?: { repo?: string };
};
type EngineRow = {
  engine_sha256_short?: string; hf_revision_short?: string; build_duration_s?: number;
  status?: 'active' | 'previous' | 'archived'; build_completed_at?: string;
};

const STATE_META: Record<string, { label: string; color: string }> = {
  no_model:         { label: 'sin modelo',                color: '#6b7280' },
  ready:            { label: 'listo',                     color: '#16a34a' },
  building:         { label: 'compilando',                color: '#1192e8' },
  degraded:         { label: 'degradado',                 color: '#e69f00' },
  update_available: { label: 'actualización disponible',  color: '#e69f00' },
};

const PHASE_LABELS: Record<string, string> = {
  acquired_lock: 'preparando', downloaded_manifest: 'descargando manifest',
  downloaded_onnx: 'descargando ONNX', verified_sha: 'verificando SHA',
  stopped_server: 'liberando GPU', prep_nano: 'preparando Nano',
  trtexec_built: 'compilando TensorRT', validated: 'validando',
  backed_up_previous: 'respaldando', atomic_swap: 'intercambio atómico',
  restoring_nano: 'restaurando', starting_server: 'reiniciando server', done: 'completado',
};

const ROW_STATUS: Record<string, string> = { active: 'activo', previous: 'previo', archived: 'archivado' };
const CLASSES: ClsName[] = ['glass', 'paper', 'plastic'];

// ─── Formato ──────────────────────────────────────────────────────────────────
function fmtLocal(iso?: string): string {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const p = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'America/Bogota', year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(d);
  const g = (t: string) => (p.find((x) => x.type === t) || { value: '' }).value;
  return `${g('year')}-${g('month')}-${g('day')} ${g('hour')}:${g('minute')}`;
}
function fmtRelative(iso?: string): string {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return '';
  const s = Math.floor((Date.now() - d.getTime()) / 1000);
  if (s < 60) return 'hace unos segundos';
  if (s < 3600) return `hace ${Math.floor(s / 60)} min`;
  if (s < 86400) return `hace ${Math.floor(s / 3600)} h`;
  return `hace ${Math.floor(s / 86400)} d`;
}
function fmtDuration(s?: number): string {
  if (s == null) return '—';
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return r ? `${m} min ${r} s` : `${m} min`;
}
function precisionOf(args?: string[]): string {
  if (!args || !args.length) return '—';
  const ws = args.find((a) => a.startsWith('--workspace='));
  const fp16 = args.includes('--fp16');
  const parts: string[] = [fp16 ? 'FP16' : 'FP32'];
  if (ws) parts.push(`WS ${ws.split('=')[1]} MiB`);
  return parts.join(' · ');
}
const shortHash = (h?: string, n = 8) => (h ? h.slice(0, n) : '—');

// Mensaje de error en español a partir de un fallo de fetch (sin filtrar el
// "Failed to fetch" crudo del navegador).
function errMsg(e: unknown): string {
  return (e as { name?: string })?.name === 'AbortError'
    ? 'El Nano no respondió a tiempo.'
    : 'No se pudo conectar con el Nano.';
}

// ─── Iconos ────────────────────────────────────────────────────────────────────
function Icon(props: { children: JSX.Element; size?: number }) {
  return (
    <svg width={props.size ?? 16} height={props.size ?? 16} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      {props.children}
    </svg>
  );
}
const IconBack    = () => <Icon><path d="m15 18-6-6 6-6" /></Icon>;
const IconChip    = () => <Icon><rect width="16" height="16" x="4" y="4" rx="2" /><rect width="6" height="6" x="9" y="9" rx="1" /><path d="M15 2v2M9 2v2M15 20v2M9 20v2M2 15h2M2 9h2M20 15h2M20 9h2" /></Icon>;
const IconBolt    = () => <Icon><path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z" /></Icon>;
const IconTerminal= () => <Icon><path d="m4 17 6-6-6-6" /><path d="M12 19h8" /></Icon>;
const IconHistory = () => <Icon><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l4 2" /></Icon>;

export default function Engine() {
  const [model, setModel]         = createSignal<ModelStateT | null>(null);
  const [unreachable, setUnreach] = createSignal(false);
  const [engines, setEngines]     = createSignal<EngineRow[]>([]);
  const [notice, setNotice]       = createSignal<{ text: string; kind: 'info' | 'warn' | 'error' } | null>(null);
  const [launching, setLaunching] = createSignal(false);
  const [logCount, setLogCount]   = createSignal(0);

  const st       = () => model()?.state ?? 'no_model';
  const building = () => st() === 'building';
  const job      = () => model()?.active_job ?? null;

  // ── Logs (buffer no reactivo + flush por rAF) ────────────────────────────────
  let logsPane!: HTMLPreElement;
  const LOG_CAP = 2000;
  let pending: string[] = [];
  let logRaf: number | null = null;
  let follow = true;
  let sse: EventSource | null = null;
  let sseJobId: string | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;
  let disposed = false;

  function lineColor(line: string): string | null {
    if (/(error|fail|failed|traceback|exception|fatal|abort)/i.test(line)) return '#f87171';
    if (/(warn|warning)/i.test(line)) return '#fbbf24';
    return null;
  }
  function flushLogs() {
    logRaf = null;
    if (!pending.length || !logsPane) return;
    const atBottom = (logsPane.scrollHeight - logsPane.scrollTop - logsPane.clientHeight) < 28;
    const frag = document.createDocumentFragment();
    for (const line of pending) {
      const div = document.createElement('div');
      div.textContent = line || ' ';
      const c = lineColor(line);
      if (c) div.style.color = c;
      frag.appendChild(div);
    }
    pending = [];
    logsPane.appendChild(frag);
    while (logsPane.childElementCount > LOG_CAP) logsPane.removeChild(logsPane.firstChild!);
    setLogCount(logsPane.childElementCount);
    if (atBottom && follow) logsPane.scrollTop = logsPane.scrollHeight;
  }
  function enqueueLog(line: string) {
    pending.push(line);
    if (logRaf === null) logRaf = requestAnimationFrame(flushLogs);
  }
  function resetLogs() {
    pending = [];
    if (logRaf !== null) { cancelAnimationFrame(logRaf); logRaf = null; }
    if (logsPane) logsPane.replaceChildren();
    setLogCount(0);
    follow = true;
  }
  function stopLogStream() {
    if (sse) { sse.close(); sse = null; }
    sseJobId = null;
    if (logRaf !== null) { cancelAnimationFrame(logRaf); logRaf = null; }
  }
  function ensureLogStream(jobId: string) {
    if (sseJobId === jobId && sse) return;       // ya conectados a este job
    stopLogStream();
    resetLogs();
    sse = new EventSource(`${nanoHttpBase()}/jobs/${jobId}/logs`);
    sseJobId = jobId;
    sse.addEventListener('log', (ev) => {
      try { const d = JSON.parse((ev as MessageEvent).data); if (typeof d.line === 'string') enqueueLog(d.line); }
      catch { /* línea malformada */ }
    });
    sse.addEventListener('done', () => {
      flushLogs();
      stopLogStream();
      void fetchState();
      void fetchEngines();
    });
    sse.onerror = () => { stopLogStream(); };
  }

  // ── Fetchers ─────────────────────────────────────────────────────────────────
  // fetch con timeout: el poll cada 3 s no debe acumular requests colgados si el
  // Nano está lento o caído (Headscale puede tener latencia).
  async function fetchT(url: string, opts: RequestInit = {}, ms = 4000): Promise<Response> {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try { return await fetch(url, { ...opts, signal: ctrl.signal }); }
    finally { clearTimeout(t); }
  }

  async function fetchState() {
    try {
      const r = await fetchT(`${nanoHttpBase()}/model/state`);
      if (!r.ok) { setUnreach(true); return; }
      const data = (await r.json()) as ModelStateT;
      setUnreach(false);
      setModel(data);
      if (data.state === 'building' && data.active_job?.job_id) ensureLogStream(data.active_job.job_id);
    } catch {
      setUnreach(true);
    }
  }
  async function fetchEngines() {
    try {
      const r = await fetchT(`${nanoHttpBase()}/jobs?limit=20`);
      if (!r.ok) return;
      const data = await r.json();
      if (Array.isArray(data?.engines)) setEngines(data.engines as EngineRow[]);
    } catch { /* sin historial */ }
  }

  function flash(text: string, kind: 'info' | 'warn' | 'error' = 'info') {
    setNotice({ text, kind });
    setTimeout(() => { if (!disposed) setNotice(null); }, 7000);
  }

  // ── Acciones ─────────────────────────────────────────────────────────────────
  async function triggerBuild(force = false) {
    if (launching() || building()) return;
    setLaunching(true);
    try {
      const r = await fetchT(`${nanoHttpBase()}/model/build`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ force }),
      }, 10000);
      let data: any = {};
      try { data = await r.json(); } catch { /* sin body */ }
      if (r.status === 409) {
        flash(`Ya hay un build en curso (${data?.detail?.active_job_id ?? data?.active_job_id ?? '—'}).`, 'warn');
      } else if (!r.ok) {
        flash(data?.detail?.error || data?.error || `No se pudo lanzar el build (HTTP ${r.status}).`, 'error');
      } else {
        flash(`Build lanzado: ${data?.job_id ?? ''}. Los logs salen en vivo abajo.`, 'info');
      }
    } catch (e) {
      flash(errMsg(e), 'error');
    } finally {
      setLaunching(false);
      void fetchState();
    }
  }
  async function cancelBuild() {
    const id = job()?.job_id;
    if (!id) return;
    try {
      await fetchT(`${nanoHttpBase()}/jobs/${id}`, { method: 'DELETE' }, 8000);
      flash('Cancelación enviada al builder.', 'warn');
    } catch (e) {
      flash(errMsg(e), 'error');
    } finally {
      void fetchState();
    }
  }
  async function checkUpdates() {
    try {
      const r = await fetchT(`${nanoHttpBase()}/model/check-updates`, { method: 'POST' }, 15000);
      const d = await r.json();
      if (!d?.has_engine) flash(`Sin engine local. HF tiene ${shortHash(d?.latest_revision, 7)}.`, 'warn');
      else if (d?.up_to_date) flash(`Modelo al día (commit ${shortHash(d?.current_revision, 7)}).`, 'info');
      else flash(`Hay actualización: ${shortHash(d?.current_revision, 7)} → ${shortHash(d?.latest_revision, 7)}.`, 'warn');
    } catch (e) {
      flash(errMsg(e), 'error');
    }
  }
  async function rollback() {
    try {
      const r = await fetchT(`${nanoHttpBase()}/model/rollback`, { method: 'POST' }, 15000);
      const d = await r.json().catch(() => ({}));
      if (r.ok && d?.ok !== false) flash('Revertido al engine anterior.', 'info');
      else flash(d?.error || 'No se pudo revertir.', 'error');
    } catch (e) {
      flash(errMsg(e), 'error');
    } finally {
      void fetchState();
      void fetchEngines();
    }
  }

  // ── Ciclo de vida ─────────────────────────────────────────────────────────────
  // Solo consultamos /model/state con el WS activo (mismo criterio que /health):
  // con el Nano caído, wsStatus ya lo refleja y evitamos errores de red
  // incapturables que ensuciarían la consola cada 3 s.
  function refresh() {
    if (wsStatus() === 'active') { void fetchState(); void fetchEngines(); }
    else setUnreach(true);
  }

  onMount(() => {
    refresh();
    ws.addEventListener('open', refresh);            // refresco inmediato al (re)conectar
    pollTimer = setInterval(refresh, 3000);
    // El listener de scroll define si seguimos el final (pausa al subir).
    if (logsPane) {
      logsPane.addEventListener('scroll', () => {
        follow = (logsPane.scrollHeight - logsPane.scrollTop - logsPane.clientHeight) < 28;
      }, { passive: true });
    }
  });
  onCleanup(() => {
    disposed = true;
    if (pollTimer) clearInterval(pollTimer);
    ws.removeEventListener('open', refresh);
    stopLogStream();
  });

  const noticeColor = (k: string) => (k === 'error' ? '#f87171' : k === 'warn' ? '#e69f00' : 'var(--text-secondary)');

  return (
    <div class="h-screen overflow-hidden bg-bg-app flex flex-col">
      <header class="flex items-center justify-between border-b border-border px-6 h-14 gap-4 shrink-0">
        <div class="flex items-center gap-3 min-w-0">
          <A href="/" aria-label="Volver al inicio"
            class="flex items-center text-text-secondary hover:text-text-primary transition-colors">
            <IconBack />
          </A>
          <span class="font-semibold text-text-primary">Engine del Modelo</span>
        </div>
        <div class="flex items-center gap-4">
          <span class="flex items-center gap-2 font-mono text-xs text-text-secondary">
            <StatusDot status={wsStatus()} />
            <span>{CONN_LABELS[wsStatus()] ?? wsStatus()}</span>
          </span>
          <ThemeToggle />
        </div>
      </header>

      <div class="flex-1 flex min-h-0">
        {/* Principal: estado + logs */}
        <section class="flex-1 min-w-0 flex flex-col">
          {/* Estado del modelo */}
          <div class="border-b border-border px-6 py-4 shrink-0">
            <h2 class="flex items-center gap-2 mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary">
              <IconChip /> Estado del modelo
            </h2>

            <Show when={!unreachable()} fallback={
              <p class="text-sm text-text-secondary">
                No se pudo contactar al Nano. Verificá la conexión (Headscale) y que el servidor esté arriba.
              </p>
            }>
              <div class="flex items-center gap-3 mb-4">
                <span class="inline-flex items-center gap-2 rounded-md px-2.5 py-1 text-sm"
                  style={{ 'background-color': (STATE_META[st()]?.color ?? '#6b7280') + '22',
                           'color': STATE_META[st()]?.color ?? '#6b7280' }}>
                  <span class="w-2 h-2 rounded-full" classList={{ 'animate-pulse': building() }}
                    style={{ 'background-color': STATE_META[st()]?.color ?? '#6b7280' }} />
                  {STATE_META[st()]?.label ?? st()}
                </span>
                <Show when={model()?.hf?.repo}>
                  <span class="font-mono text-xs text-text-secondary truncate">{model()!.hf!.repo}</span>
                </Show>
              </div>

              {/* Progreso del build en curso */}
              <Show when={building()}>
                <div class="mb-4">
                  <div class="flex items-center justify-between mb-1.5 text-xs">
                    <span class="text-text-secondary">{PHASE_LABELS[job()?.phase ?? ''] ?? job()?.phase ?? '—'}</span>
                    <span class="font-mono tabular text-text-primary">{job()?.progress_pct ?? 0}%</span>
                  </div>
                  <div class="h-1.5 w-full rounded-full bg-bg-surface overflow-hidden">
                    <div class="h-full bg-accent transition-[width] duration-500"
                      style={{ width: `${job()?.progress_pct ?? 0}%` }} />
                  </div>
                </div>
              </Show>

              {/* Metadatos del engine activo */}
              <dl class="grid grid-cols-2 gap-x-8 gap-y-2.5 max-w-2xl">
                <Field label="Revisión HF"
                  value={shortHash(model()?.active_engine?.hf_revision, 7)}
                  sub={fmtRelative(model()?.active_engine?.hf_commit_date)} />
                <Field label="Compilado"
                  value={fmtLocal(model()?.active_engine?.build_completed_at)}
                  sub={fmtRelative(model()?.active_engine?.build_completed_at)} />
                <Field label="Duración del build" value={fmtDuration(model()?.active_engine?.build_duration_s)} />
                <Field label="Precisión" value={precisionOf(model()?.active_engine?.trtexec_args)} />
                <Field label="Engine SHA" value={shortHash(model()?.active_engine?.engine_sha256, 12)} mono />
                <Field label="ONNX SHA" value={shortHash(model()?.active_engine?.onnx_sha256, 12)} mono />
              </dl>

              <div class="mt-4 flex items-center gap-4">
                <span class="text-xs text-text-secondary">Clases</span>
                <div class="flex items-center gap-3">
                  <For each={CLASSES}>{(c) => (
                    <span class="flex items-center gap-1.5 text-xs text-text-secondary">
                      <span class="w-2 h-2 rounded-full" style={{ 'background-color': WONG[c] }} />
                      {CLASS_LABEL_ES[c]}
                    </span>
                  )}</For>
                </div>
                <Show when={model()?.active_engine?.from_fallback}>
                  <span class="rounded px-1.5 py-0.5 text-[11px]" style={{ 'background-color': '#e69f0022', color: '#e69f00' }}>fallback</span>
                </Show>
                <Show when={model()?.active_engine?.adopted}>
                  <span class="rounded px-1.5 py-0.5 text-[11px] text-text-secondary border border-border">adoptado</span>
                </Show>
              </div>
            </Show>
          </div>

          {/* Logs del build */}
          <div class="flex-1 min-h-0 flex flex-col px-6 py-4">
            <h2 class="flex items-center gap-2 mb-3 text-[11px] font-medium uppercase tracking-wider text-text-secondary shrink-0">
              <IconTerminal /> Logs del build
            </h2>
            <div class="relative flex-1 min-h-0">
              <pre ref={logsPane}
                class="absolute inset-0 overflow-auto rounded-md bg-bg-surface border border-border p-3
                       font-mono text-xs leading-relaxed text-text-secondary whitespace-pre-wrap break-words" />
              <Show when={logCount() === 0}>
                <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <span class="text-sm text-text-secondary">
                    {building() ? 'Esperando logs del build…' : 'Sin build en curso. Lanzá una compilación para ver los logs en vivo.'}
                  </span>
                </div>
              </Show>
            </div>
          </div>
        </section>

        {/* Panel lateral: acciones + historial */}
        <aside class="w-80 shrink-0 border-l border-border bg-bg-panel overflow-y-auto">
          <Section title="Acciones" icon={IconBolt}>
            <div class="space-y-2">
              <button class={`${BTN_PRIMARY} w-full`} disabled={building() || launching() || unreachable()}
                onClick={() => triggerBuild(false)}>
                {launching() ? 'lanzando…' : 'Compilar engine'}
              </button>
              <button class={`${BTN} w-full`} disabled={building() || launching() || unreachable()}
                onClick={() => triggerBuild(true)}>
                Forzar recompilación
              </button>
              <button class={`${BTN_GHOST} w-full`} disabled={unreachable()} onClick={checkUpdates}>
                Verificar actualizaciones
              </button>
              <Show when={building()}>
                <button class={`${BTN} w-full`} onClick={cancelBuild}>Cancelar build</button>
              </Show>
              <Show when={model()?.previous_engine && !building()}>
                <button class={`${BTN_GHOST} w-full`} onClick={rollback}>Revertir al engine anterior</button>
              </Show>
            </div>
            <Show when={notice()}>
              <p class="mt-3 text-xs leading-snug" style={{ color: noticeColor(notice()!.kind) }}>{notice()!.text}</p>
            </Show>
          </Section>

          <Section title="Historial de engines" icon={IconHistory}>
            <Show when={engines().length} fallback={<p class="text-xs text-text-secondary">Sin historial disponible.</p>}>
              <ul class="space-y-2.5">
                <For each={engines()}>{(e) => (
                  <li class="flex items-center gap-2 text-xs">
                    <span class="w-1.5 h-1.5 rounded-full shrink-0"
                      style={{ 'background-color': e.status === 'active' ? '#16a34a' : e.status === 'previous' ? '#e69f00' : '#6b7280' }} />
                    <span class="font-mono text-text-primary">{e.hf_revision_short || shortHash(e.engine_sha256_short, 7)}</span>
                    <span class="text-text-secondary">{ROW_STATUS[e.status ?? ''] ?? ''}</span>
                    <span class="ml-auto text-text-secondary tabular">{fmtRelative(e.build_completed_at) || fmtDuration(e.build_duration_s)}</span>
                  </li>
                )}</For>
              </ul>
            </Show>
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

function Field(props: { label: string; value: string; sub?: string; mono?: boolean }) {
  return (
    <div>
      <dt class="text-xs text-text-secondary">{props.label}</dt>
      <dd class="mt-0.5">
        <span class={`text-sm text-text-primary ${props.mono ? 'font-mono' : ''}`}>{props.value}</span>
        <Show when={props.sub}>
          <span class="ml-1.5 text-xs text-text-secondary">{props.sub}</span>
        </Show>
      </dd>
    </div>
  );
}
