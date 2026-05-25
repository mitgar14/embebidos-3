import { WIRE_COLORS } from '../../connections.js';

/** Panel lateral con título, descripción, snippet de código y badges. */
export class InfoPanel {
  constructor(root) {
    this.root = root;
    this.title = root.querySelector('#step-title');
    this.description = root.querySelector('#step-description');
    this.code = root.querySelector('#step-code');
    this.badge = root.querySelector('#step-badge');
    this.colorChip = root.querySelector('#step-color');
    this.counter = root.querySelector('#step-counter');
  }

  renderStep(step, index, total) {
    this.title.textContent = step.title;
    this.description.textContent = step.description;
    this.code.textContent = step.code;
    this.counter.textContent = `${index + 1} / ${total}`;

    const hex = WIRE_COLORS[step.color];
    this.colorChip.style.background = `#${hex.toString(16).padStart(6, '0')}`;

    this.badge.textContent = step.required ? 'Obligatorio' : 'Opcional';
    this.badge.className = 'badge ' + (step.required ? 'badge-required' : 'badge-optional');
  }

  showOverview(total) {
    this.title.textContent = 'Vista general · todas las conexiones';
    this.description.textContent =
      'Todas las conexiones visibles a la vez. Usa el ratón para orbitar y haz clic en un pin ' +
      'para inspeccionarlo. Usa "Paso a paso" para revisarlas una por una.';
    this.code.textContent = '// 4 obligatorias + 2 opcionales';
    this.counter.textContent = `${total} conexiones`;
    this.badge.textContent = 'Resumen';
    this.badge.className = 'badge badge-overview';
    this.colorChip.style.background = 'linear-gradient(90deg,#d62828,#f1c40f,#0077ff,#111)';
  }
}
