// modelo.js — gestión de la pestaña modelo
(() => {
  "use strict";
  const state = {
    pollTimer: null,
    sse: null,
    lastState: null,
  };

  function apiBase() {
    const wsInput = document.getElementById('ws-url');
    const wsUrl = wsInput ? wsInput.value : '';
    if (wsUrl) {
      return wsUrl.replace(/^ws/, 'http').replace(/\/ws$/, '');
    }
    return window.location.origin;
  }

  function api(path) {
    return apiBase() + path;
  }

  async function fetchState() {
    try {
      const r = await fetch(api('/model/state'));
      const data = await r.json();
      state.lastState = data;
      render(data);
    } catch (e) {
      renderError(e);
    }
  }

  function render(s) {
    const main = document.getElementById('modelo-content');
    if (!main) return;
    const tpl = TEMPLATES[s.state] || TEMPLATES.no_model;
    main.innerHTML = tpl(s);
    renderSidebar(s);
    wireActions(s);
  }

  function renderError(e) {
    const main = document.getElementById('modelo-content');
    if (main) {
      main.innerHTML = `<div class="card-error">No se pudo cargar el estado: ${e.message}</div>`;
    }
  }

  const TEMPLATES = {
    no_model: (s) => `
      <div class="hero-card">
        <h2>Sin modelo cargado</h2>
        <p>Descargá el último ONNX desde HF Hub y compilá el engine TRT optimizado para este Jetson Nano.</p>
        <button id="btn-build" class="primary">descargar y compilar engine</button>
        <p class="hint">Tarda entre 15 y 40 minutos. Podés cerrar la pestaña; el proceso sigue en el servidor.</p>
      </div>`,
    ready: (s) => readyTemplate(s),
    update_available: (s) => readyTemplate(s, { banner: true }),
    degraded: (s) => degradedTemplate(s),
    building: (s) => buildingTemplate(s),
  };

  function readyTemplate(s, opts = {}) {
    const m = s.active_engine || {};
    const banner = opts.banner ? `
      <div class="banner-update">
        <strong>Nuevo entrenamiento disponible en HF Hub.</strong>
        <button id="btn-build" class="primary">actualizar engine</button>
      </div>` : '';
    return `
      ${banner}
      <div class="modelo-card">
        <h2>Modelo activo</h2>
        <dl class="modelo-info">
          <dt>origen</dt><dd>commit <span class="mono">${(m.hf_revision || '').slice(0,7)}</span> · ${(m.hf_commit_date || '—')}</dd>
          <dt>onnx</dt><dd>sha <span class="mono">${(m.onnx_sha256 || '').slice(0,8)}</span></dd>
          <dt>engine</dt><dd>sha <span class="mono">${(m.engine_sha256 || '').slice(0,8)}</span> · FP16</dd>
          <dt>compilado</dt><dd>${(m.build_completed_at || '—')} · ${m.build_duration_s || '—'} s</dd>
          <dt>workspace</dt><dd>${(m.trtexec_args || []).find(a => a.startsWith('--workspace=')) || '—'}</dd>
        </dl>
        <div class="modelo-actions">
          <button id="btn-check-updates">verificar actualizaciones</button>
          <button id="btn-force-rebuild">forzar recompilación</button>
        </div>
      </div>`;
  }

  function degradedTemplate(s) {
    return `
      <div class="banner-warn">
        <strong>Usando engine anterior.</strong>
        El último intento de actualización falló. Engine en uso: commit
        <span class="mono">${(s.active_engine?.hf_revision || '').slice(0,7)}</span>.
      </div>
      ${readyTemplate(s)}`;
  }

  function buildingTemplate(s) {
    const j = s.active_job || {};
    const pct = j.progress_pct || 0;
    return `
      <div class="modelo-card">
        <h2>Compilando engine — <span class="mono">${j.job_id || ''}</span></h2>
        <dl class="modelo-info">
          <dt>fase actual</dt><dd>${j.phase || '—'}</dd>
          <dt>progreso</dt><dd>
            <div class="progress"><div class="progress-bar" style="width:${pct}%"></div></div>
            <span class="mono">${pct}%</span>
          </dd>
          <dt>origen</dt><dd>commit <span class="mono">${(j.onnx_source?.hf_revision || '').slice(0,7)}</span></dd>
        </dl>
        <div class="modelo-actions">
          <button id="btn-cancel" class="danger">cancelar build</button>
        </div>
        <div class="logs-pane">
          <h3>logs en vivo</h3>
          <pre id="logs-stream" class="logs-stream"></pre>
        </div>
      </div>`;
  }

  function renderSidebar(s) {
    const side = document.querySelector('.modelo-side');
    if (!side) return;
    const m = s.active_engine || {};
    const prev = s.previous_engine;
    side.innerHTML = `
      <section class="side-card">
        <h3>servidor</h3>
        <dl class="kv">
          <dt>estado</dt><dd>activo</dd>
          <dt>endpoint</dt><dd class="mono">${apiBase()}</dd>
        </dl>
      </section>

      <section class="side-card">
        <h3>HF Hub</h3>
        <dl class="kv">
          <dt>repo</dt><dd class="mono small">mitgar14/embebidos-3-models</dd>
          <dt>revision activa</dt><dd class="mono">${(m.hf_revision || '—').slice(0,7)}</dd>
        </dl>
        <button id="btn-side-check" class="ghost">verificar ahora</button>
      </section>

      <section class="side-card">
        <h3>acciones</h3>
        <button id="btn-side-rebuild" class="ghost">forzar recompilación</button>
        <button id="btn-side-rollback" class="ghost" ${prev ? '' : 'disabled'}>revertir a engine anterior</button>
      </section>

      <section class="side-card">
        <h3>histórico</h3>
        <p class="hint">últimos jobs (próximamente)</p>
      </section>
    `;
    const sb = document.getElementById('btn-side-check');
    if (sb) sb.onclick = () => checkUpdates();
    const fb = document.getElementById('btn-side-rebuild');
    if (fb) fb.onclick = () => triggerBuild(true);
    const rb = document.getElementById('btn-side-rollback');
    if (rb && !rb.disabled) rb.onclick = () => rollback();
  }

  async function rollback() {
    if (!confirm('¿Revertir al engine anterior?')) return;
    try {
      const r = await fetch(api('/model/rollback'), { method: 'POST' });
      const data = await r.json();
      if (!data.ok) {
        alert('No se pudo: ' + (data.detail?.error || 'unknown'));
      }
      fetchState();
    } catch (e) { alert(e.message); }
  }

  function wireActions(s) {
    const btnBuild = document.getElementById('btn-build');
    if (btnBuild) btnBuild.onclick = () => triggerBuild();

    const btnCheck = document.getElementById('btn-check-updates');
    if (btnCheck) btnCheck.onclick = () => checkUpdates();

    const btnForce = document.getElementById('btn-force-rebuild');
    if (btnForce) btnForce.onclick = () => triggerBuild(true);

    const btnCancel = document.getElementById('btn-cancel');
    if (btnCancel) btnCancel.onclick = () => cancelBuild(s.active_job?.job_id);

    if (s.state === 'building' && s.active_job?.job_id) {
      startLogsStream(s.active_job.job_id);
    } else {
      stopLogsStream();
    }
  }

  async function triggerBuild(force = false) {
    try {
      const r = await fetch(api('/model/build'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force }),
      });
      const data = await r.json();
      if (r.status === 409) {
        alert(`Ya hay un build activo: ${data.detail?.active_job_id || ''}`);
      } else if (!r.ok) {
        alert(`No se pudo lanzar build: ${data.detail?.error || r.status}`);
      }
      fetchState();
    } catch (e) { alert(e.message); }
  }

  async function cancelBuild(jobId) {
    if (!jobId || !confirm('¿Cancelar el build en curso?')) return;
    try {
      await fetch(api('/jobs/' + jobId), { method: 'DELETE' });
      fetchState();
    } catch (e) { alert(e.message); }
  }

  async function checkUpdates() {
    try {
      const r = await fetch(api('/model/check-updates'), { method: 'POST' });
      const data = await r.json();
      if (data.up_to_date) {
        alert('Modelo al día.');
      } else {
        const latest = (data.latest_revision || '').slice(0, 7);
        const current = (data.current_revision || '—').slice(0, 7);
        alert(`Hay novedad: ${latest} (actual ${current})`);
      }
      fetchState();
    } catch (e) { alert(e.message); }
  }

  function startLogsStream(jobId) {
    stopLogsStream();
    const url = api(`/jobs/${jobId}/logs`);
    state.sse = new EventSource(url);
    state.sse.addEventListener('log', (ev) => {
      const data = JSON.parse(ev.data);
      const pane = document.getElementById('logs-stream');
      if (pane) {
        pane.textContent += data.line + '\n';
        pane.scrollTop = pane.scrollHeight;
      }
    });
    state.sse.addEventListener('done', () => {
      stopLogsStream();
      fetchState();
    });
    state.sse.onerror = () => {
      stopLogsStream();
    };
  }

  function stopLogsStream() {
    if (state.sse) { state.sse.close(); state.sse = null; }
  }

  window.initModelTab = function () {
    fetchState();
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(fetchState, 3000);
  };

  // expose for sidebar (F3)
  window._modelo = { state, api, fetchState, triggerBuild, cancelBuild, checkUpdates, rollback, startLogsStream, stopLogsStream };
})();
