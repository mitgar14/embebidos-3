// embebidos-3 dashboard — vanilla JS, ES2020+

(() => {
  "use strict";

  const COLORS = {
    glass:   "oklch(76% 0.16 150)",
    paper:   "oklch(70% 0.15 245)",
    plastic: "oklch(72% 0.17 55)",
  };
  const CLASS_LABEL_ES = { glass: "vidrio", paper: "papel", plastic: "plástico" };

  const $ = (id) => document.getElementById(id);
  const els = {
    video: $("video"),
    overlay: $("overlay"),
    canvasWrap: $("canvas-wrap"),
    emptyHint: $("empty-hint"),

    // header chips
    chipStatus: $("chip-status"),
    statusLabel: $("status-label"),
    chipTemp: $("chip-temp"),
    hFps: $("h-fps"),
    hLat: $("h-lat"),
    hGpu: $("h-gpu"),
    hRam: $("h-ram"),

    // controls
    wsUrl: $("ws-url"),
    btnConnect: $("btn-connect"),
    btnDisconnect: $("btn-disconnect"),
    camSelect: $("cam-select"),
    btnCamStart: $("btn-cam-start"),
    btnCamStop: $("btn-cam-stop"),
    camSize: $("cam-size"),

    confSlider: $("conf-slider"),
    confValue: $("conf-value"),
    fpsSlider: $("fps-slider"),
    fpsValue: $("fps-value"),
    showVideo: $("show-video"),

    // metrics
    mInfer: $("m-infer"),
    mNet: $("m-net"),
    mDets: $("m-dets"),
    mTotalFrames: $("m-total-frames"),
    cGlass: $("c-glass"),
    cPaper: $("c-paper"),
    cPlastic: $("c-plastic"),

    btnSnapshot: $("btn-snapshot"),
  };

  const state = {
    ws: null,
    stream: null,
    capturing: false,
    targetFps: 14,
    confTh: 0.5,
    seqOut: 0,
    seqIn: 0,
    pendingFrames: new Map(),
    classCounts: { glass: 0, paper: 0, plastic: 0 },
    fpsWindow: [],
    captureCanvas: document.createElement("canvas"),
    captureCtx: null,
    overlayCtx: null,
    rafId: null,
    inFlight: 0,
    maxInFlight: 2,
    healthFailCount: 0,
    healthTimer: null,
  };
  state.captureCtx = state.captureCanvas.getContext("2d");
  state.overlayCtx = els.overlay.getContext("2d");

  // ---------- WebSocket ------------------------------------------------------
  function connect() {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) return;
    const url = els.wsUrl.value.trim();
    state.ws = new WebSocket(url);
    state.ws.binaryType = "arraybuffer";

    state.ws.onopen = () => {
      setStatus(true);
      sendControl({ type: "conf", value: state.confTh });
    };
    state.ws.onclose = () => setStatus(false);
    state.ws.onerror = () => setStatus(false);
    state.ws.onmessage = (ev) => {
      try { handleServerMessage(JSON.parse(ev.data)); }
      catch (e) { console.error("parse error", e); }
    };
  }

  function disconnect() {
    if (state.ws) { state.ws.close(); state.ws = null; }
    setStatus(false);
  }

  function setStatus(on) {
    els.chipStatus.dataset.state = on ? "active" : "inactive";
    els.statusLabel.textContent = on ? "activa" : "inactiva";
  }

  function sendControl(obj) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(obj));
    }
  }

  function handleServerMessage(msg) {
    if (msg.type === "conf_ack" || msg.type === "pong") return;
    if (!msg.ok) { console.warn("server err", msg); return; }
    const pending = state.pendingFrames.get(msg.seq);
    if (pending) state.pendingFrames.delete(msg.seq);
    state.inFlight = Math.max(0, state.inFlight - 1);

    const now = performance.now();
    state.fpsWindow.push(now);
    state.fpsWindow = state.fpsWindow.filter((t) => now - t < 1000);
    state.seqIn = msg.seq;

    const dets = msg.bboxes || [];
    drawDetections(dets);

    const total = pending ? Math.round(now - pending.sendTs) : null;
    const infer = msg.t_infer_ms || 0;
    const net = total !== null ? Math.max(0, total - infer) : null;

    els.hFps.textContent = state.fpsWindow.length.toFixed(1);
    els.hLat.textContent = total !== null ? `${total}` : "—";
    els.mInfer.textContent = infer.toFixed(1);
    els.mNet.textContent = net !== null ? `${net.toFixed(0)}` : "—";
    els.mDets.textContent = dets.length;
    els.mTotalFrames.textContent = state.seqIn;

    // empty hint visibility
    els.emptyHint.dataset.hidden = dets.length > 0 ? "true" : "false";

    for (const d of dets) {
      if (state.classCounts[d.cls_name] !== undefined) {
        state.classCounts[d.cls_name]++;
      }
    }
    els.cGlass.textContent = state.classCounts.glass;
    els.cPaper.textContent = state.classCounts.paper;
    els.cPlastic.textContent = state.classCounts.plastic;
  }

  // ---------- Health polling -------------------------------------------------
  async function pollHealth() {
    const wsUrl = els.wsUrl.value.trim();
    const httpUrl = wsUrl.replace(/^ws/, "http").replace(/\/ws$/, "/health");
    try {
      const r = await fetch(httpUrl);
      const j = await r.json();
      state.healthFailCount = 0;

      // unstale
      els.chipTemp.dataset.stale = "false";

      if (typeof j.gpu_temp_c === "number") {
        els.hGpu.textContent = j.gpu_temp_c.toFixed(0);
        if (j.gpu_temp_c > 70)      els.chipTemp.dataset.state = "err";
        else if (j.gpu_temp_c > 60) els.chipTemp.dataset.state = "warn";
        else                         els.chipTemp.dataset.state = "ok";
      }
      if (typeof j.ram_available_mb === "number") {
        els.hRam.textContent = j.ram_available_mb;
      }
    } catch (e) {
      state.healthFailCount++;
      if (state.healthFailCount >= 3) {
        els.chipTemp.dataset.stale = "true";
        els.hGpu.textContent = "—";
        els.hRam.textContent = "—";
      }
    }
  }

  function startHealthPoll() {
    if (state.healthTimer) return;
    pollHealth();
    state.healthTimer = setInterval(pollHealth, 3000);
  }

  // ---------- Cámara ---------------------------------------------------------
  async function enumerateCams() {
    try {
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true });
      tmp.getTracks().forEach((t) => t.stop());
    } catch (e) {
      console.warn("permiso cámara denegado:", e);
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    els.camSelect.innerHTML = "";
    cams.forEach((c, i) => {
      const opt = document.createElement("option");
      opt.value = c.deviceId;
      opt.textContent = c.label || `cámara ${i + 1}`;
      els.camSelect.appendChild(opt);
    });
  }

  async function startCam() {
    if (state.capturing) return;
    const deviceId = els.camSelect.value;
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
      audio: false,
    });
    els.video.srcObject = state.stream;
    await els.video.play();
    const w = els.video.videoWidth, h = els.video.videoHeight;
    els.camSize.textContent = `${w} × ${h}`;
    state.captureCanvas.width = w;
    state.captureCanvas.height = h;
    els.overlay.width = w;
    els.overlay.height = h;
    // aspect-ratio del wrap = aspect real del video (sin barras)
    els.canvasWrap.style.aspectRatio = `${w} / ${h}`;
    state.capturing = true;
    captureLoop();
  }

  function stopCam() {
    state.capturing = false;
    if (state.rafId) { cancelAnimationFrame(state.rafId); state.rafId = null; }
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    els.video.srcObject = null;
    state.overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  }

  let lastCaptureTs = 0;

  function captureLoop() {
    if (!state.capturing) return;
    const now = performance.now();
    const minInterval = 1000 / state.targetFps;
    const elapsed = now - lastCaptureTs;
    if (elapsed >= minInterval && state.inFlight < state.maxInFlight) {
      lastCaptureTs = now;
      sendFrame();
    }
    state.rafId = requestAnimationFrame(captureLoop);
  }

  function sendFrame() {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    if (els.video.readyState < 2) return;
    const w = state.captureCanvas.width, h = state.captureCanvas.height;
    state.captureCtx.drawImage(els.video, 0, 0, w, h);
    state.captureCanvas.toBlob(
      (blob) => {
        if (!blob || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
        const seq = ++state.seqOut;
        state.pendingFrames.set(seq, { sendTs: performance.now() });
        state.inFlight++;
        blob.arrayBuffer().then((buf) => state.ws.send(buf));
      },
      "image/jpeg",
      0.7,
    );
  }

  // ---------- Render bboxes --------------------------------------------------
  function drawDetections(dets) {
    const ctx = state.overlayCtx;
    const W = els.overlay.width, H = els.overlay.height;
    ctx.clearRect(0, 0, W, H);

    if (!els.showVideo.checked) {
      ctx.fillStyle = "oklch(10% 0.005 80)";
      ctx.fillRect(0, 0, W, H);
    }
    els.video.style.opacity = els.showVideo.checked ? "1" : "0";

    ctx.lineWidth = 2;
    ctx.font = "700 13px 'Source Sans 3', system-ui, sans-serif";

    for (const d of dets) {
      const color = COLORS[d.cls_name] || "#fff";
      const x = d.x1, y = d.y1, w = d.x2 - d.x1, h = d.y2 - d.y1;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      const labelEs = CLASS_LABEL_ES[d.cls_name] || d.cls_name;
      const label = `${labelEs} ${(d.conf * 100).toFixed(0)}%`;
      const tw = ctx.measureText(label).width + 10;
      const th = 18;
      const ty = Math.max(th, y);
      // pill background con esquinas suaves
      ctx.fillStyle = color;
      roundRect(ctx, x, ty - th, tw, th, 3);
      ctx.fill();
      // texto
      ctx.fillStyle = "oklch(20% 0.020 80)";
      ctx.fillText(label, x + 5, ty - 5);
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ---------- Snapshot -------------------------------------------------------
  function snapshot() {
    const w = state.captureCanvas.width, h = state.captureCanvas.height;
    if (!w || !h) return;
    const out = document.createElement("canvas");
    out.width = w; out.height = h;
    const octx = out.getContext("2d");
    if (els.showVideo.checked) {
      octx.drawImage(els.video, 0, 0, w, h);
    } else {
      octx.fillStyle = "oklch(10% 0.005 80)";
      octx.fillRect(0, 0, w, h);
    }
    octx.drawImage(els.overlay, 0, 0, w, h);
    out.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `snapshot-${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // ---------- UI wiring ------------------------------------------------------
  els.btnConnect.onclick = connect;
  els.btnDisconnect.onclick = disconnect;
  els.btnCamStart.onclick = startCam;
  els.btnCamStop.onclick = stopCam;
  els.btnSnapshot.onclick = snapshot;

  els.confSlider.oninput = (e) => {
    const pct = parseInt(e.target.value, 10);
    state.confTh = pct / 100;
    els.confValue.innerHTML = `${pct}<span class="u">%</span>`;
    sendControl({ type: "conf", value: state.confTh });
  };
  els.fpsSlider.oninput = (e) => {
    state.targetFps = parseInt(e.target.value, 10);
    els.fpsValue.innerHTML = `${state.targetFps}<span class="u">/s</span>`;
  };

  // ---------- Routing simple hash-based -----------------------------------
  function setTab(name) {
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(t => t.setAttribute('aria-selected', t.dataset.tab === name ? 'true' : 'false'));
    document.querySelectorAll('[data-pane]').forEach(p => {
      p.hidden = p.dataset.pane !== name;
    });
    if (name === 'modelo' && typeof window.initModelTab === 'function') {
      window.initModelTab();
    }
    if (window.location.hash !== '#' + name) {
      history.replaceState(null, '', '#' + name);
    }
  }

  function currentTab() {
    return (window.location.hash.replace('#', '') || 'live');
  }

  document.querySelectorAll('.tab').forEach(t => {
    t.onclick = () => setTab(t.dataset.tab);
  });
  window.addEventListener('hashchange', () => setTab(currentTab()));

  // ---------- Init -----------------------------------------------------------
  (async () => {
    setStatus(false);
    await enumerateCams().catch(() => {});
    startHealthPoll();
    connect();
    try { await startCam(); } catch (e) { console.warn("auto-start cam falló:", e); }
    setTab(currentTab());
  })();
})();
