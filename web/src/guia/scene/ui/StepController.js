/**
 * Estado del recorrido paso-a-paso vs. vista general.
 * No toca DOM ni Three.js; emite eventos onChange.
 */
export class StepController {
  constructor(steps) {
    this.steps = steps;
    this.index = 0;
    this.mode = 'step'; // 'step' | 'overview'
    this._listeners = [];
    this.visited = new Set();   // ids de pasos ya vistos (checklist)
  }

  onChange(fn) { this._listeners.push(fn); }
  _emit() { this._listeners.forEach(fn => fn(this.snapshot())); }

  snapshot() {
    return {
      mode: this.mode,
      index: this.index,
      step: this.steps[this.index],
      total: this.steps.length,
      visited: this.visited
    };
  }

  markVisited() {
    if (this.mode === 'step') this.visited.add(this.steps[this.index].id);
  }

  next() {
    if (this.mode !== 'step') this.mode = 'step';
    this.index = (this.index + 1) % this.steps.length;
    this._emit();
  }

  prev() {
    if (this.mode !== 'step') this.mode = 'step';
    this.index = (this.index - 1 + this.steps.length) % this.steps.length;
    this._emit();
  }

  goTo(i) {
    this.mode = 'step';
    this.index = Math.max(0, Math.min(this.steps.length - 1, i));
    this._emit();
  }

  showOverview() {
    this.mode = 'overview';
    this._emit();
  }

  start() { this._emit(); }
}
