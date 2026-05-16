// ui.js — toast + modal helpers reutilizables para reemplazar alert/confirm.
// Expone: window.showToast(message, type, opts), window.openModal({...}) → Promise<bool>
(() => {
  "use strict";

  function ensureToastHost() {
    let host = document.getElementById("toast-host");
    if (!host) {
      host = document.createElement("div");
      host.id = "toast-host";
      host.className = "toast-host";
      document.body.appendChild(host);
    }
    return host;
  }

  // showToast(message, type='info', { title?, durationMs? })
  // type ∈ {info, success, warn, error}
  function showToast(message, type, opts) {
    type = type || "info";
    opts = opts || {};
    const durationMs = opts.durationMs != null ? opts.durationMs : 6000;
    const host = ensureToastHost();
    const toast = document.createElement("div");
    toast.className = "toast toast-" + type;
    toast.setAttribute("role", "status");
    toast.setAttribute("aria-live", "polite");

    const closeBtn = document.createElement("button");
    closeBtn.className = "toast-close";
    closeBtn.setAttribute("aria-label", "cerrar");
    closeBtn.textContent = "×";

    const body = document.createElement("div");
    body.className = "toast-body";

    if (opts.title) {
      const t = document.createElement("strong");
      t.className = "toast-title";
      t.textContent = opts.title;
      body.appendChild(t);
    }
    // mensaje multilínea: cada \n → <br>
    const msg = document.createElement("div");
    msg.className = "toast-msg";
    String(message).split("\n").forEach((line, i) => {
      if (i > 0) msg.appendChild(document.createElement("br"));
      msg.appendChild(document.createTextNode(line));
    });
    body.appendChild(msg);

    toast.appendChild(body);
    toast.appendChild(closeBtn);
    host.appendChild(toast);

    let dismissed = false;
    const dismiss = () => {
      if (dismissed) return;
      dismissed = true;
      toast.classList.add("toast-leaving");
      setTimeout(() => toast.remove(), 200);
    };
    closeBtn.addEventListener("click", dismiss);
    if (durationMs > 0) setTimeout(dismiss, durationMs);
    // animate-in
    requestAnimationFrame(() => toast.classList.add("toast-in"));
    return { dismiss };
  }

  // openModal({ title, body, confirmText='Confirmar', cancelText='Cancelar', danger=false })
  // body puede ser string (texto plano) o HTMLElement.
  // Retorna Promise<boolean>.
  function openModal(opts) {
    opts = opts || {};
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "modal-overlay";
      overlay.setAttribute("role", "dialog");
      overlay.setAttribute("aria-modal", "true");

      const dialog = document.createElement("div");
      dialog.className = "modal-dialog";

      if (opts.title) {
        const h = document.createElement("h3");
        h.className = "modal-title";
        h.textContent = opts.title;
        dialog.appendChild(h);
      }

      const bodyEl = document.createElement("div");
      bodyEl.className = "modal-body";
      if (typeof opts.body === "string") {
        opts.body.split("\n").forEach((line, i) => {
          if (i > 0) bodyEl.appendChild(document.createElement("br"));
          bodyEl.appendChild(document.createTextNode(line));
        });
      } else if (opts.body instanceof HTMLElement) {
        bodyEl.appendChild(opts.body);
      }
      dialog.appendChild(bodyEl);

      const actions = document.createElement("div");
      actions.className = "modal-actions";
      const cancelBtn = document.createElement("button");
      cancelBtn.className = "ghost";
      cancelBtn.textContent = opts.cancelText || "Cancelar";
      const confirmBtn = document.createElement("button");
      confirmBtn.className = opts.danger ? "danger" : "primary";
      confirmBtn.textContent = opts.confirmText || "Confirmar";
      actions.appendChild(cancelBtn);
      actions.appendChild(confirmBtn);
      dialog.appendChild(actions);

      overlay.appendChild(dialog);
      document.body.appendChild(overlay);
      requestAnimationFrame(() => overlay.classList.add("modal-in"));

      const close = (result) => {
        overlay.classList.add("modal-leaving");
        setTimeout(() => overlay.remove(), 180);
        document.removeEventListener("keydown", onKey);
        resolve(result);
      };
      const onKey = (e) => {
        if (e.key === "Escape") close(false);
        if (e.key === "Enter") close(true);
      };
      cancelBtn.addEventListener("click", () => close(false));
      confirmBtn.addEventListener("click", () => close(true));
      overlay.addEventListener("click", (e) => { if (e.target === overlay) close(false); });
      document.addEventListener("keydown", onKey);
      setTimeout(() => confirmBtn.focus(), 50);
    });
  }

  // withButtonLoading(btn, loadingLabel, asyncFn)
  // Patrón sustentado en Nielsen (response times) + gomakethings (aria-busy) + thelinuxcode (vanilla JS):
  //   1. flag inFlight (data-busy) previene doble-submit antes de que disabled aplique
  //   2. disabled + aria-busy="true" anuncian a screen readers
  //   3. label change descriptivo (no solo spinner) explica QUÉ está pasando
  //   4. finally restaura estado SIEMPRE — si falla, el usuario puede reintentar sin recargar
  //   5. min-width preservado por CSS para evitar layout shift
  // Si loadingLabel es null/undefined, se preserva el texto original (solo spinner + disabled).
  async function withButtonLoading(btn, loadingLabel, fn) {
    if (!btn) return await fn();
    if (btn.dataset.busy === '1') return; // idempotencia: segundo click es no-op
    btn.dataset.busy = '1';
    const originalText = btn.textContent;
    const originalAriaBusy = btn.getAttribute('aria-busy');
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
    btn.classList.add('is-loading');
    if (loadingLabel) btn.textContent = loadingLabel;
    try {
      return await fn();
    } finally {
      btn.disabled = false;
      if (originalAriaBusy === null) btn.removeAttribute('aria-busy');
      else btn.setAttribute('aria-busy', originalAriaBusy);
      btn.classList.remove('is-loading');
      btn.textContent = originalText;
      delete btn.dataset.busy;
    }
  }

  window.showToast = showToast;
  window.openModal = openModal;
  window.withButtonLoading = withButtonLoading;
})();
