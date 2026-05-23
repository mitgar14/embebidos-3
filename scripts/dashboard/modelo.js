// modelo.js — gestión de la pestaña modelo
(() => {
  "use strict";
  const state = {
    pollTimer: null,
    sse: null,
    sseJobId: null,   // job_id del SSE activo, para evitar re-abrir en cada poll
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

  // Formato datetime para metadata tecnica:
  //   - Convierte ISO UTC a hora local del usuario (en este proyecto, Colombia UTC-5)
  //   - Devuelve "YYYY-MM-DD HH:MM:SS" + el ISO original separado para tooltip
  //   - Patron Cloudscape/Oxide: relativo o local visible, ISO absoluto en title
  function _fmtLocalDateTime(iso) {
    if (!iso) return { display: '—', iso: '' };
    const d = new Date(iso);
    if (isNaN(d.getTime())) return { display: iso, iso: iso };
    // Forzamos timezone America/Bogota (UTC-5, sin DST). Sin dependencias externas.
    const opts = {
      timeZone: 'America/Bogota',
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false,
    };
    const parts = new Intl.DateTimeFormat('en-CA', opts).formatToParts(d);
    const get = (t) => (parts.find(p => p.type === t) || {}).value || '';
    const display = `${get('year')}-${get('month')}-${get('day')} ${get('hour')}:${get('minute')}:${get('second')}`;
    return { display, iso };
  }

  // Tiempo relativo corto: "hace 12 min", "hace 3 h", "hace 2 d"
  function _fmtRelative(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    const sec = Math.floor((Date.now() - d.getTime()) / 1000);
    if (sec < 60) return 'hace unos segundos';
    if (sec < 3600) return `hace ${Math.floor(sec / 60)} min`;
    if (sec < 86400) return `hace ${Math.floor(sec / 3600)} h`;
    if (sec < 604800) return `hace ${Math.floor(sec / 86400)} d`;
    return _fmtLocalDateTime(iso).display.slice(0, 10);
  }

  // Helper: ISO timestamp inline con time element para tooltip nativo del browser
  function _timeEl(iso) {
    if (!iso) return '<span class="meta-empty">—</span>';
    const f = _fmtLocalDateTime(iso);
    const rel = _fmtRelative(iso);
    const relSuffix = rel ? ` <span class="meta-rel">· ${rel}</span>` : '';
    return `<time datetime="${iso}" title="${iso} (UTC)">${f.display}</time>${relSuffix}`;
  }

  function _trtexecArgsSummary(args) {
    if (!args || !args.length) return '';
    const ws = args.find(a => a.startsWith('--workspace='));
    const fp16 = args.includes('--fp16');
    const wsVal = ws ? ws.split('=')[1] : null;
    const parts = [];
    if (wsVal) parts.push(`WS ${wsVal} MiB`);
    parts.push(fp16 ? 'FP16' : 'FP32');
    return parts.join(' · ');
  }

  async function fetchState() {
    try {
      const r = await fetch(api('/model/state'));
      const data = await r.json();
      state.lastState = data;
      state.lastFetchOk = Date.now();
      render(data);
    } catch (e) {
      renderUnreachable(e);
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

  // Cuando el server no responde (típicamente: build en curso que detuvo el server).
  // Mantenemos el último estado conocido visible con un overlay friendly + retry.
  function renderUnreachable(err) {
    const main = document.getElementById('modelo-content');
    if (!main) return;
    const last = state.lastState;
    const lastWasBuilding = last && last.state === 'building';
    const hint = lastWasBuilding
      ? 'El servidor se reinicia automáticamente cuando termina la compilación. Reintento en curso…'
      : 'El servidor está temporalmente fuera de alcance. Reintento en curso…';
    main.innerHTML = `
      <div class="card-warn">
        <h2>Servidor no responde</h2>
        <p>${hint}</p>
        <p class="hint mono">${err && err.message ? err.message : 'network error'}</p>
        <button id="btn-retry-state" aria-describedby="retry-status">reintentar ahora</button>
        <p id="retry-status" role="status" aria-live="polite" class="sr-only"></p>
      </div>`;
    const btn = document.getElementById('btn-retry-state');
    const statusEl = document.getElementById('retry-status');
    if (btn) {
      btn.onclick = () => window.withButtonLoading(btn, 'reintentando…', async () => {
        if (statusEl) statusEl.textContent = 'Reintentando conexión al servidor…';
        try {
          await fetchState();
          if (statusEl) statusEl.textContent = 'Conexión restablecida.';
        } catch (_) {
          if (statusEl) statusEl.textContent = 'Aún no responde. Intentá de nuevo.';
        }
      });
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
        <button id="btn-adopt" class="primary" title="Hashea el .engine local y lo asocia con la revision/ONNX SHA256 actual de HF Hub. NO recompila.">Adoptar engine existente</button>
      </div>` : '';
    return `
      <section class="modelo-section">
        <header class="modelo-section-head">
          <h2>Sin modelo cargado</h2>
        </header>
        <p>Descargá el último ONNX desde HF Hub y compilá el engine TRT optimizado para este Jetson Nano.</p>
        <button id="btn-build" class="primary">Descargar y compilar engine</button>
        <p class="hint">Tarda entre 15 y 40 minutos. Podés cerrar la pestaña; el proceso sigue en el servidor.</p>
        ${orphanBlock}
      </section>`;
  }

  function readyTemplate(s, opts = {}) {
    const m = s.active_engine || {};
    const banner = opts.banner ? `
      <div class="banner-update">
        <strong>Nuevo entrenamiento disponible en HF Hub.</strong>
        <button id="btn-build" class="primary">Actualizar engine</button>
      </div>` : '';
    const adoptedBadge = m.adopted ? ` <span class="badge-adopted" title="Engine pre-existente cuyo origen fue inferido (no compilado por este sistema). Compilá de nuevo con «Recompilar engine» para tener tracking completo de duración y parámetros.">adoptado</span>` : '';

    const commitShort = (m.hf_revision || '').slice(0, 7) || '—';
    const commitFull = m.hf_revision || '';
    const onnxShort = (m.onnx_sha256 || '').slice(0, 12);
    const engineShort = (m.engine_sha256 || '').slice(0, 12);
    const trainedHtml = _timeEl(m.hf_commit_date);
    const compiledHtml = _timeEl(m.build_completed_at);
    const durationLabel = m.build_duration_s
      ? `${Math.round(m.build_duration_s)} s`
      : '<span class="meta-empty">sin tracking</span>';
    const buildSummary = _trtexecArgsSummary(m.trtexec_args) || '<span class="meta-empty">sin tracking</span>';

    return `
      ${banner}
      <section class="modelo-section" aria-labelledby="modelo-h">
        <header class="modelo-section-head">
          <h2 id="modelo-h">Modelo activo${adoptedBadge}</h2>
        </header>

        <div class="meta-grid" role="group" aria-label="Identidad del artefacto">
          <div class="meta-row">
            <span class="meta-label">
              <svg class="meta-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3"/><line x1="3" y1="12" x2="9" y2="12"/><line x1="15" y1="12" x2="21" y2="12"/></svg>
              Commit
            </span>
            <span class="meta-value mono" title="${commitFull}">${commitShort}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">ONNX</span>
            <span class="meta-value mono" title="${m.onnx_sha256 || ''}">${onnxShort || '<span class="meta-empty">—</span>'}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Engine</span>
            <span class="meta-value mono" title="${m.engine_sha256 || ''}">${engineShort || '<span class="meta-empty">—</span>'}</span>
          </div>
        </div>

        <div class="meta-grid meta-grid--bordered" role="group" aria-label="Información del build">
          <div class="meta-row">
            <span class="meta-label">
              <svg class="meta-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
              Entrenado
            </span>
            <span class="meta-value">${trainedHtml}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">
              <svg class="meta-ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94z"/></svg>
              Compilado
            </span>
            <span class="meta-value">${compiledHtml}</span>
          </div>
          <div class="meta-row meta-row--inline">
            <span class="meta-label">Build</span>
            <span class="meta-value">${durationLabel} <span class="meta-sep">·</span> ${buildSummary}</span>
          </div>
        </div>

        <div class="modelo-actions">
          <button id="btn-check-updates" title="Compara la revisión y el SHA256 del ONNX local contra HF Hub.">Verificar actualizaciones</button>
          <button id="btn-force-rebuild" title="Vuelve a descargar el ONNX actual de HF y compila el engine desde cero. Útil si el engine local está corrupto o cambiaste parámetros de build.">Recompilar engine</button>
        </div>
      </section>`;
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
    const startedHtml = _timeEl(j.started_at);
    const commitShort = (j.onnx_source?.hf_revision || '').slice(0, 7) || '—';
    return `
      <section class="modelo-section">
        <header class="modelo-section-head">
          <h2>Compilando engine <span class="job-id mono">${j.job_id || ''}</span></h2>
        </header>

        <div class="build-progress">
          <div class="build-progress-bar"><div class="build-progress-fill" style="width:${pct}%"></div></div>
          <div class="build-progress-meta">
            <span class="build-phase">${j.phase || '—'}</span>
            <span class="build-pct mono">${pct}%</span>
          </div>
        </div>

        <div class="meta-grid">
          <div class="meta-row">
            <span class="meta-label">Commit</span>
            <span class="meta-value mono">${commitShort}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">Iniciado</span>
            <span class="meta-value">${startedHtml}</span>
          </div>
        </div>

        <div class="modelo-actions">
          <button id="btn-cancel" class="danger">Cancelar build</button>
        </div>

        <div class="logs-pane">
          <h3>Logs en vivo</h3>
          <pre id="logs-stream" class="logs-stream"></pre>
        </div>
      </section>`;
  }

  function renderSidebar(s) {
    const side = document.querySelector('.modelo-side');
    if (!side) return;
    const m = s.active_engine || {};
    const prev = s.previous_engine;
    side.innerHTML = `
      <section class="side-card">
        <h3>Servidor</h3>
        <dl class="kv">
          <dt>Estado</dt><dd>activo</dd>
          <dt>Endpoint</dt><dd class="mono">${apiBase()}</dd>
        </dl>
      </section>

      <section class="side-card">
        <h3>HF Hub</h3>
        <dl class="kv">
          <dt>Repo</dt><dd class="mono small">${(s.hf && s.hf.repo) || 'mitgar14/embebidos-3-models'}</dd>
          <dt>Revisión activa</dt><dd class="mono">${(m.hf_revision || '—').slice(0,7)}</dd>
        </dl>
        <button id="btn-side-check" class="ghost" title="Consulta HF Hub: compara el SHA256 del ONNX local contra el último publicado.">Verificar ahora</button>
      </section>

      <section class="side-card">
        <h3>Acciones</h3>
        <button id="btn-side-rebuild" class="ghost" title="Vuelve a descargar el ONNX actual de HF y compila el engine desde cero. Útil si el engine local está corrupto o cambiaste parámetros de build.">Recompilar engine</button>
        <button id="btn-side-rollback" class="ghost" title="Restaura el engine anterior (swap inverso). Solo disponible si hay un backup." ${prev ? '' : 'disabled'}>Revertir a engine anterior</button>
      </section>

      <section class="side-card">
        <h3>Histórico</h3>
        <p class="hint">Últimos jobs (próximamente)</p>
      </section>
    `;
    const sb = document.getElementById('btn-side-check');
    if (sb) sb.onclick = () => checkUpdates(sb);
    const fb = document.getElementById('btn-side-rebuild');
    if (fb) fb.onclick = () => triggerBuild(true, fb);
    const rb = document.getElementById('btn-side-rollback');
    if (rb && !rb.disabled) rb.onclick = () => rollback(rb);
  }

  async function adoptEngine(btn) {
    const ok = await window.openModal({
      title: 'Adoptar engine existente',
      body:
        'Vamos a hashear el binario .engine que ya está en el Nano y asociarle ' +
        'los metadatos (hf_revision + onnx_sha256) de la HEAD actual de HF Hub.\n\n' +
        'NO se recompila el engine — solo se registra retroactivamente. Útil si el ' +
        'binario fue compilado antes del sistema de tracking.',
      confirmText: 'adoptar',
      cancelText: 'cancelar',
    });
    if (!ok) return;
    await window.withButtonLoading(btn, 'adoptando…', async () => {
      try {
        const r = await fetch(api('/model/adopt'), { method: 'POST' });
        const data = await r.json();
        if (r.status === 404) {
          window.showToast('No hay engine binario en el Nano. Usá "descargar y compilar engine".',
                           'error', { title: 'No se puede adoptar' });
        } else if (r.status === 409) {
          window.showToast('El engine ya tiene meta asociado. No es necesario adoptar.',
                           'warn', { title: 'Adopción innecesaria' });
        } else if (!r.ok) {
          window.showToast(data.detail && data.detail.error ? data.detail.error : ('HTTP ' + r.status),
                           'error', { title: 'No se pudo adoptar' });
        } else {
          const rev = (data.meta.hf_revision || '').slice(0, 7);
          const sha = (data.meta.onnx_sha256 || '').slice(0, 8);
          window.showToast(`Commit ${rev} · ONNX ${sha}`,
                           'success', { title: 'Engine adoptado' });
        }
        fetchState();
      } catch (e) {
        window.showToast(e.message, 'error', { title: 'Error de red' });
      }
    });
  }

  async function rollback(btn) {
    const ok = await window.openModal({
      title: 'Revertir al engine anterior',
      body: 'Esto restaura el .engine y .meta.json guardados como .previous. Solo si hay backup.',
      confirmText: 'revertir',
      cancelText: 'cancelar',
      danger: true,
    });
    if (!ok) return;
    await window.withButtonLoading(btn, 'revirtiendo…', async () => {
      try {
        const r = await fetch(api('/model/rollback'), { method: 'POST' });
        const data = await r.json();
        if (!data.ok) {
          window.showToast(data.detail && data.detail.error ? data.detail.error : 'unknown',
                           'error', { title: 'No se pudo revertir' });
        } else {
          window.showToast('El engine anterior está activo de nuevo.', 'success',
                           { title: 'Rollback completo' });
        }
        fetchState();
      } catch (e) {
        window.showToast(e.message, 'error', { title: 'Error de red' });
      }
    });
  }

  function wireActions(s) {
    const btnBuild = document.getElementById('btn-build');
    if (btnBuild) btnBuild.onclick = () => triggerBuild(false, btnBuild);

    const btnAdopt = document.getElementById('btn-adopt');
    if (btnAdopt) btnAdopt.onclick = () => adoptEngine(btnAdopt);

    const btnCheck = document.getElementById('btn-check-updates');
    if (btnCheck) btnCheck.onclick = () => checkUpdates(btnCheck);

    const btnForce = document.getElementById('btn-force-rebuild');
    if (btnForce) btnForce.onclick = () => triggerBuild(true, btnForce);

    const btnCancel = document.getElementById('btn-cancel');
    if (btnCancel) btnCancel.onclick = () => cancelBuild(s.active_job?.job_id, btnCancel);

    if (s.state === 'building' && s.active_job?.job_id) {
      // FIX 2026-05-16: solo abrir SSE si no hay uno activo para el mismo job.
      // Antes: cada poll (3 s) cerraba y reabría → el server re-leía el .log
      // desde el principio → logs en bucle visible para el usuario.
      if (state.sse && state.sseJobId === s.active_job.job_id) {
        // ya conectado, no tocar
      } else {
        startLogsStream(s.active_job.job_id);
      }
    } else {
      stopLogsStream();
    }
  }

  // Verificación post-error: el patrón "naive rollback creates a false negative" — si el POST
  // falló pero el server ya creó el job (timeout intermedio, server respondiendo lento), no
  // mostramos error sin antes consultar GET /model/state. Si hay job activo nuevo, fue éxito.
  async function _confirmBuildLaunchedDespiteError(originalErr) {
    try {
      const r = await fetch(api('/model/state'));
      if (!r.ok) return null;
      const data = await r.json();
      if (data.state === 'building' && data.active_job && data.active_job.job_id) {
        return data.active_job.job_id;
      }
      return null;
    } catch (_) {
      return null;
    }
  }

  async function triggerBuild(force = false, btn = null) {
    await window.withButtonLoading(btn, 'lanzando build…', async () => {
      let lastErr = null;
      try {
        const r = await fetch(api('/model/build'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ force }),
        });
        let data = {};
        try { data = await r.json(); } catch (_) { /* sin body */ }
        if (r.status === 409) {
          window.showToast(`Job activo: ${data.detail && data.detail.active_job_id || ''}`,
                           'warn', { title: 'Ya hay un build en curso' });
        } else if (!r.ok) {
          // antes de declarar fallo: verificar si el job sí nació (race condition real)
          const jobId = await _confirmBuildLaunchedDespiteError();
          if (jobId) {
            window.showToast(`Job ${jobId} — el server respondió con error pero el build sí inició. El dashboard sigue el progreso.`,
                             'warn', { title: 'Build lanzado (con aviso del server)', durationMs: 8000 });
          } else {
            const errMsg = (data.detail && data.detail.error) || ('HTTP ' + r.status);
            window.showToast(errMsg, 'error', { title: 'No se pudo lanzar build' });
          }
        } else {
          window.showToast(`Job ${data.job_id} — el dashboard sigue el progreso en vivo.`,
                           'info', { title: 'Build lanzado' });
        }
      } catch (e) {
        lastErr = e;
        // Network error: pudo haberse procesado del lado del server. Verificamos.
        const jobId = await _confirmBuildLaunchedDespiteError();
        if (jobId) {
          window.showToast(`Job ${jobId} — la respuesta no llegó pero el build sí inició.`,
                           'warn', { title: 'Build lanzado (red intermitente)', durationMs: 8000 });
        } else {
          window.showToast(e.message, 'error', { title: 'Error de red' });
        }
      } finally {
        fetchState();
      }
    });
  }

  async function cancelBuild(jobId, btn = null) {
    if (!jobId) return;
    const ok = await window.openModal({
      title: 'Cancelar build',
      body: `Se envía SIGTERM al builder ${jobId} y se restaura el engine anterior. Continuar?`,
      confirmText: 'cancelar build',
      cancelText: 'mantener',
      danger: true,
    });
    if (!ok) return;
    await window.withButtonLoading(btn, 'cancelando…', async () => {
      try {
        await fetch(api('/jobs/' + jobId), { method: 'DELETE' });
        window.showToast('Solicitud de cancelación enviada.', 'info', { title: 'Cancelando' });
        fetchState();
      } catch (e) {
        window.showToast(e.message, 'error', { title: 'Error de red' });
      }
    });
  }

  async function checkUpdates(btn = null) {
    await window.withButtonLoading(btn, 'verificando…', async () => {
      try {
        const r = await fetch(api('/model/check-updates'), { method: 'POST' });
        const data = await r.json();
        const ui = formatCheckUpdates(data);
        window.showToast(ui.body, ui.type, { title: ui.title, durationMs: ui.durationMs });
        fetchState();
      } catch (e) {
        window.showToast(e.message, 'error', { title: 'No se pudo consultar HF' });
      }
    });
  }

  // Devuelve { title, body, type, durationMs } para el toast de check-updates.
  function formatCheckUpdates(data) {
    const latestRev = (data.latest_revision || '').slice(0, 7) || '—';
    const currentRev = (data.current_revision || '').slice(0, 7) || '—';
    const latestSha = (data.latest_onnx_sha256 || '').slice(0, 8) || '—';
    const currentSha = (data.current_onnx_sha256 || '').slice(0, 8) || '—';

    if (!data.has_engine) {
      return {
        title: 'Sin engine local',
        body: `HF Hub tiene la versión ${latestRev} (ONNX ${latestSha}).\nUsá "descargar y compilar engine".`,
        type: 'warn',
        durationMs: 9000,
      };
    }
    if (data.up_to_date && data.same_revision) {
      return {
        title: 'Modelo al día',
        body: `Commit ${currentRev} · ONNX ${currentSha}`,
        type: 'success',
        durationMs: 5000,
      };
    }
    if (data.up_to_date && !data.same_revision) {
      return {
        title: 'Modelo al día (commit cosmético)',
        body: `Mismo ONNX ${currentSha}.\nHF tiene commits nuevos sin tocar el modelo:\n${currentRev} → ${latestRev}`,
        type: 'success',
        durationMs: 8000,
      };
    }
    if (!data.same_onnx && !data.same_revision) {
      return {
        title: 'Nueva iteración disponible',
        body:
          `ONNX: ${currentSha} → ${latestSha}\n` +
          `Commit: ${currentRev} → ${latestRev}\n` +
          `Usá "recompilar engine" para actualizar.`,
        type: 'warn',
        durationMs: 10000,
      };
    }
    return {
      title: 'Inconsistencia detectada',
      body: `Mismo commit (${currentRev}) pero ONNX distinto: ${currentSha} ≠ ${latestSha}.\nRevisá HF Hub manualmente.`,
      type: 'error',
      durationMs: 10000,
    };
  }

  // Buffer + flush por RAF para no congelar el browser con bursts de TRT.
  // Cap circular: máx LOG_CAP líneas en pantalla (FIFO drop al inicio).
  // Auto-follow scroll solo si el usuario está cerca del bottom.
  // 25 líneas = experiencia "en vivo" estricta — siempre las últimas TRT spam lines
  // visibles, sin scroll histórico. El server además seek(EOF) en cada conexión SSE.
  const LOG_CAP = 25;
  const _logs = {
    pending: [],     // líneas que aún no se commitearon al DOM
    rendered: [],    // líneas actualmente en el DOM (para cap circular)
    rafId: null,
    follow: true,
  };

  function _enqueueLog(line) {
    _logs.pending.push(line);
    if (_logs.rafId !== null) return;
    _logs.rafId = requestAnimationFrame(_flushLogs);
  }

  function _flushLogs() {
    _logs.rafId = null;
    const pane = document.getElementById('logs-stream');
    if (!pane) { _logs.pending.length = 0; return; }
    if (_logs.pending.length === 0) return;

    // ¿el usuario está siguiendo el final? (margen 24px) — capturar antes de mutar
    const wasAtBottom = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;

    const batchLines = _logs.pending;
    _logs.pending = [];
    const chunk = batchLines.join('\n') + '\n';

    for (let i = 0; i < batchLines.length; i++) _logs.rendered.push(batchLines[i]);

    if (_logs.rendered.length > LOG_CAP) {
      // overflow: cap circular. Una sola O(n) cuando excede.
      _logs.rendered.splice(0, _logs.rendered.length - LOG_CAP);
      pane.textContent = _logs.rendered.join('\n') + '\n';
    } else {
      // path caliente: append textNode (O(1) por inserción)
      pane.appendChild(document.createTextNode(chunk));
    }

    if (wasAtBottom && _logs.follow) {
      pane.scrollTop = pane.scrollHeight;
    }
  }

  function startLogsStream(jobId) {
    stopLogsStream();
    _logs.pending.length = 0;
    _logs.rendered.length = 0;
    _logs.follow = true;
    const url = api(`/jobs/${jobId}/logs`);
    state.sse = new EventSource(url);
    state.sseJobId = jobId;
    state.sse.addEventListener('log', (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (typeof data.line === 'string') _enqueueLog(data.line);
      } catch (_) { /* línea malformada — descartar */ }
    });
    state.sse.addEventListener('done', () => {
      // flush pendientes antes de cerrar
      if (_logs.rafId !== null) { cancelAnimationFrame(_logs.rafId); _logs.rafId = null; }
      _flushLogs();
      stopLogsStream();
      fetchState();
    });
    state.sse.onerror = () => {
      stopLogsStream();
    };

    // listener para detectar scroll manual del usuario (pausa auto-follow)
    setTimeout(() => {
      const pane = document.getElementById('logs-stream');
      if (!pane) return;
      pane.addEventListener('scroll', () => {
        const atBottom = (pane.scrollHeight - pane.scrollTop - pane.clientHeight) < 24;
        _logs.follow = atBottom;
      }, { passive: true });
    }, 0);
  }

  function stopLogsStream() {
    if (state.sse) { state.sse.close(); state.sse = null; }
    state.sseJobId = null;
    if (_logs.rafId !== null) { cancelAnimationFrame(_logs.rafId); _logs.rafId = null; }
  }

  window.initModelTab = function () {
    fetchState();
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(fetchState, 3000);
  };

  // expose for sidebar (F3) + dev/smoke testing
  window._modelo = { state, api, fetchState, render, triggerBuild, cancelBuild, checkUpdates, rollback, startLogsStream, stopLogsStream };
})();
