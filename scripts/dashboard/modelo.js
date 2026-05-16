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
    no_model: (s) => noModelTemplate(s),
    ready: (s) => readyTemplate(s),
    update_available: (s) => readyTemplate(s, { banner: true }),
    degraded: (s) => degradedTemplate(s),
    building: (s) => buildingTemplate(s),
  };

  function noModelTemplate(s) {
    const orphan = s.engine_binary_present === true;
    const orphanBlock = orphan ? `
      <div class="banner-warn" style="margin-top:16px">
        <strong>Hay un engine TRT en el Nano sin tracking.</strong>
        <p class="hint" style="margin:6px 0">
          Encontramos <span class="mono">best_fp16.engine</span> en disco pero sin metadatos
          (probablemente compilado antes del sistema de tracking). Podés <em>adoptarlo</em>
          si confiás en que corresponde a la versión actual de HF Hub — el sistema le creará
          un meta retroactivo apuntando a la HEAD actual.
        </p>
        <button id="btn-adopt" class="primary" title="Hashea el .engine local y lo asocia con la revision/ONNX SHA256 actual de HF Hub. NO recompila.">adoptar engine existente</button>
      </div>` : '';
    return `
      <div class="hero-card">
        <h2>Sin modelo cargado</h2>
        <p>Descargá el último ONNX desde HF Hub y compilá el engine TRT optimizado para este Jetson Nano.</p>
        <button id="btn-build" class="primary">descargar y compilar engine</button>
        <p class="hint">Tarda entre 15 y 40 minutos. Podés cerrar la pestaña; el proceso sigue en el servidor.</p>
        ${orphanBlock}
      </div>`;
  }

  function readyTemplate(s, opts = {}) {
    const m = s.active_engine || {};
    const banner = opts.banner ? `
      <div class="banner-update">
        <strong>Nuevo entrenamiento disponible en HF Hub.</strong>
        <button id="btn-build" class="primary">actualizar engine</button>
      </div>` : '';
    const adoptedBadge = m.adopted ? ` <span class="badge-adopted" title="Engine pre-existente cuyo origen fue inferido (no compilado por este sistema). Compilar de nuevo con 'recompilar engine' para tener tracking completo de duración y parámetros.">adoptado</span>` : '';
    return `
      ${banner}
      <div class="modelo-card">
        <h2>Modelo activo${adoptedBadge}</h2>
        <dl class="modelo-info">
          <dt>origen</dt><dd>commit <span class="mono">${(m.hf_revision || '').slice(0,7)}</span> · ${(m.hf_commit_date || '—')}</dd>
          <dt>onnx</dt><dd>sha <span class="mono">${(m.onnx_sha256 || '').slice(0,8)}</span></dd>
          <dt>engine</dt><dd>sha <span class="mono">${(m.engine_sha256 || '').slice(0,8)}</span> · FP16</dd>
          <dt>compilado</dt><dd>${(m.build_completed_at || '—')} · ${m.build_duration_s || '—'} s</dd>
          <dt>workspace</dt><dd>${(m.trtexec_args || []).find(a => a.startsWith('--workspace=')) || '—'}</dd>
        </dl>
        <div class="modelo-actions">
          <button id="btn-check-updates" title="Compara la revisión y el SHA256 del ONNX local contra HF Hub.">verificar actualizaciones</button>
          <button id="btn-force-rebuild" title="Vuelve a descargar el ONNX actual de HF y compila el engine desde cero. Útil si el engine local está corrupto o cambiaste parámetros de build.">recompilar engine</button>
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
        <button id="btn-side-check" class="ghost" title="Consulta HF Hub: compara el SHA256 del ONNX local contra el último publicado.">verificar ahora</button>
      </section>

      <section class="side-card">
        <h3>acciones</h3>
        <button id="btn-side-rebuild" class="ghost" title="Vuelve a descargar el ONNX actual de HF y compila el engine desde cero. Útil si el engine local está corrupto o cambiaste parámetros de build.">recompilar engine</button>
        <button id="btn-side-rollback" class="ghost" title="Restaura el engine anterior (swap inverso). Solo disponible si hay un backup." ${prev ? '' : 'disabled'}>revertir a engine anterior</button>
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

  async function adoptEngine() {
    if (!confirm(
      'Adoptar el engine existente como si correspondiera a la versión actual de HF Hub.\n\n' +
      'Esto hashea el binario local y le asocia los metadatos de la HEAD de HF. NO recompila.\n\n' +
      '¿Continuar?'
    )) return;
    try {
      const r = await fetch(api('/model/adopt'), { method: 'POST' });
      const data = await r.json();
      if (r.status === 404) {
        alert('No hay engine binario en el Nano. Usá "descargar y compilar engine".');
      } else if (r.status === 409) {
        alert('El engine ya tiene meta asociado. No es necesario adoptar.');
      } else if (!r.ok) {
        alert('No se pudo adoptar: ' + (data.detail?.error || r.status));
      } else {
        alert(`Engine adoptado.\n\nhf_revision: ${(data.meta.hf_revision || '').slice(0, 7)}\nonnx_sha256: ${(data.meta.onnx_sha256 || '').slice(0, 8)}`);
      }
      fetchState();
    } catch (e) { alert(e.message); }
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

    const btnAdopt = document.getElementById('btn-adopt');
    if (btnAdopt) btnAdopt.onclick = () => adoptEngine();

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
      alert(formatCheckUpdatesMessage(data));
      fetchState();
    } catch (e) { alert(e.message); }
  }

  function formatCheckUpdatesMessage(data) {
    const latestRev = (data.latest_revision || '').slice(0, 7) || '—';
    const currentRev = (data.current_revision || '').slice(0, 7) || '—';
    const latestSha = (data.latest_onnx_sha256 || '').slice(0, 8) || '—';
    const currentSha = (data.current_onnx_sha256 || '').slice(0, 8) || '—';

    if (!data.has_engine) {
      return `Sin engine compilado en el Nano.\n` +
             `HF Hub tiene la versión ${latestRev} (ONNX ${latestSha}).\n\n` +
             `Usá "descargar y compilar engine" para traerla.`;
    }
    if (data.up_to_date && data.same_revision) {
      return `Modelo al día.\n` +
             `Commit ${currentRev} · ONNX ${currentSha}.`;
    }
    if (data.up_to_date && !data.same_revision) {
      return `Modelo al día (mismo ONNX ${currentSha}).\n\n` +
             `HF tiene commits nuevos sin tocar el modelo: ${currentRev} → ${latestRev}.\n` +
             `No es necesario recompilar.`;
    }
    if (!data.same_onnx && !data.same_revision) {
      return `Nueva iteración del modelo disponible.\n\n` +
             `ONNX: ${currentSha} → ${latestSha}\n` +
             `Commit: ${currentRev} → ${latestRev}\n\n` +
             `Usá "recompilar engine" para actualizar.`;
    }
    return `Inconsistencia detectada:\n` +
           `mismo commit (${currentRev}) pero ONNX distinto ` +
           `(${currentSha} ≠ ${latestSha}).\n\n` +
           `Revisá HF Hub manualmente.`;
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
