/**
 * Modal de bienvenida (3 slides) con persistencia en localStorage.
 * Aparece la primera vez. Reabribile via botón "?" en la barra flotante.
 */
const STORAGE_KEY = 'pca9685-guide.welcome.dismissed.v1';

const SLIDES = [
  {
    title: '¿Qué es esto?',
    body:
      'Guía 3D interactiva para conectar un <strong>ESP32-WROOM-32</strong> (en su placa expansora) ' +
      'a un <strong>PCA9685</strong> y mover hasta 4 servos SG90. La hicimos porque las guías ' +
      'planas (Fritzing, fotos) confunden con tantos pines.',
    bullets: [
      'Sigue 6 pasos guiados (4 obligatorios + 2 opcionales).',
      'Cada paso enfoca su cable, lo anima y explica el porqué.',
      'Trae los sketches Arduino listos para flashear.'
    ]
  },
  {
    title: 'Cómo navegar',
    body:
      'Tres formas de mirar la escena. Empieza con <strong>órbita</strong> (default) y prueba ' +
      '<strong>vuelo libre</strong> si quieres acercarte a un pin específico.',
    table: [
      ['Órbita',            'Arrastra el ratón sobre la escena · rueda zoom'],
      ['Vuelo libre (F)',   'WASD mover · arrastra para mirar · rueda velocidad'],
      ['Pasos',             '← → para avanzar · A para vista general'],
      ['Etiquetas (L)',     'Mostrar solo pines usados o todos'],
      ['Hover sobre pin',   'Tooltip con nombre y tipo'],
      ['Click sobre pin',   'Ficha técnica completa en el panel']
    ]
  },
  {
    title: 'Listo para empezar',
    body:
      'Empieza por el paso <strong>1 · Tierra común (GND)</strong>. El recorrido se completa ' +
      'en 2 minutos. Al final tendrás el bus I²C activo y los 4 servos respondiendo.',
    bullets: [
      'Pulsa <kbd>?</kbd> en cualquier momento para volver a abrir esta guía.',
      'El botón <strong>"Probar servos"</strong> simula el sweep CH0→CH3.',
      'La carpeta <code>sketch_*</code> tiene los .ino reales.'
    ],
    cta: 'Empezar la guía'
  }
];

export class WelcomeModal {
  constructor() {
    this.index = 0;
    this._build();
  }

  _build() {
    this.overlay = document.createElement('div');
    this.overlay.className = 'welcome-overlay';
    this.overlay.innerHTML = `
      <div class="welcome-card" role="dialog" aria-modal="true" aria-labelledby="welcome-title">
        <button class="welcome-close" aria-label="Cerrar">×</button>
        <h2 class="welcome-title" id="welcome-title"></h2>
        <div class="welcome-body" id="welcome-body"></div>
        <div class="welcome-nav">
          <div class="welcome-dots" id="welcome-dots"></div>
          <div class="welcome-actions">
            <button class="welcome-btn ghost" id="welcome-prev">Atrás</button>
            <button class="welcome-btn primary" id="welcome-next">Siguiente</button>
          </div>
        </div>
        <label class="welcome-skip">
          <input type="checkbox" id="welcome-no-show"> No mostrar al iniciar
        </label>
      </div>
    `;
    document.body.appendChild(this.overlay);

    this.titleEl  = this.overlay.querySelector('#welcome-title');
    this.bodyEl   = this.overlay.querySelector('#welcome-body');
    this.dotsEl   = this.overlay.querySelector('#welcome-dots');
    this.prevBtn  = this.overlay.querySelector('#welcome-prev');
    this.nextBtn  = this.overlay.querySelector('#welcome-next');
    this.closeBtn = this.overlay.querySelector('.welcome-close');
    this.noShow   = this.overlay.querySelector('#welcome-no-show');

    this.prevBtn.onclick = () => this.go(this.index - 1);
    this.nextBtn.onclick = () => {
      if (this.index === SLIDES.length - 1) this.close();
      else this.go(this.index + 1);
    };
    this.closeBtn.onclick = () => this.close();
    this.overlay.addEventListener('click', e => {
      if (e.target === this.overlay) this.close();
    });
    this._onKey = e => { if (e.key === 'Escape') this.close(); };
  }

  go(i) {
    this.index = Math.max(0, Math.min(SLIDES.length - 1, i));
    const s = SLIDES[this.index];
    this.titleEl.textContent = s.title;
    let html = `<p>${s.body}</p>`;
    if (s.bullets) {
      html += '<ul>' + s.bullets.map(b => `<li>${b}</li>`).join('') + '</ul>';
    }
    if (s.table) {
      html += '<table class="welcome-table">' +
        s.table.map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`).join('') +
        '</table>';
    }
    this.bodyEl.innerHTML = html;

    this.dotsEl.innerHTML = SLIDES.map((_, idx) =>
      `<span class="dot ${idx === this.index ? 'active' : ''}"></span>`
    ).join('');

    this.prevBtn.disabled = this.index === 0;
    this.nextBtn.textContent = this.index === SLIDES.length - 1
      ? (s.cta || 'Empezar')
      : 'Siguiente';
    this.nextBtn.classList.toggle('cta', this.index === SLIDES.length - 1);
  }

  open(force = false) {
    if (!force && localStorage.getItem(STORAGE_KEY) === '1') return false;
    this.go(0);
    this.overlay.classList.add('visible');
    document.addEventListener('keydown', this._onKey);
    return true;
  }

  close() {
    this.overlay.classList.remove('visible');
    document.removeEventListener('keydown', this._onKey);
    if (this.noShow.checked) localStorage.setItem(STORAGE_KEY, '1');
  }
}
