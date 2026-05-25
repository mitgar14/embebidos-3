/** Tooltip flotante que sigue al cursor al hacer hover sobre un pin. */
export class Tooltip {
  constructor(root) {
    this.root = root;
    this.title = root.querySelector('.tt-title');
    this.type = root.querySelector('.tt-type');
    this._visible = false;
    document.addEventListener('pointermove', e => {
      if (!this._visible) return;
      const pad = 14;
      this.root.style.left = (e.clientX + pad) + 'px';
      this.root.style.top = (e.clientY + pad) + 'px';
    });
  }

  show(pin) {
    this.title.textContent = pin.label;
    this.type.textContent = pin.type;
    this.root.classList.add('visible');
    this._visible = true;
  }

  hide() {
    this.root.classList.remove('visible');
    this._visible = false;
  }
}
