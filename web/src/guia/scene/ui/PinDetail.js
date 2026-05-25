/**
 * Tarjeta flotante con ficha técnica del pin clicado.
 */
const TYPE_INFO = {
  GND:   { name:'Tierra (GND)', volts:'0 V', desc:'Referencia del circuito. Todos los GND deben estar interconectados.' },
  VCC:   { name:'Voltaje lógico (VCC)', volts:'3.3 V o 5 V', desc:'Alimentación de la lógica del integrado.' },
  POWER: { name:'Riel de potencia', volts:'3.3 V / 5 V / VIN', desc:'Línea de alimentación del expansor.' },
  VPLUS: { name:'V+ — Alimentación de servos', volts:'5 – 6 V', desc:'NO alimenta el chip; solo las salidas de servo.' },
  SDA:   { name:'I²C — Datos (SDA)', volts:'0–3.3 V (open-drain)', desc:'Línea bidireccional de datos. Pull-up requerido (PCA9685 lo trae).' },
  SCL:   { name:'I²C — Reloj (SCL)', volts:'0–3.3 V (open-drain)', desc:'Reloj generado por el maestro.' },
  IO:    { name:'GPIO digital', volts:'0–3.3 V', desc:'Pin de propósito general. Puede ser entrada o salida.' },
  IN:    { name:'GPIO solo entrada', volts:'0–3.3 V', desc:'Pin sin función de salida (GPIO34/35/36/39).' },
  ADC:   { name:'Entrada analógica (ADC)', volts:'0–3.3 V', desc:'Conversión analógico–digital de 12 bits.' },
  UART:  { name:'UART', volts:'0–3.3 V', desc:'Línea serie. TX0/RX0 las usa el USB de programación.' },
  EN:    { name:'Chip Enable', volts:'3.3 V (alto activo)', desc:'Resetea el ESP32 si va a LOW.' },
  OE:    { name:'Output Enable (PCA9685)', volts:'0–3.3 V (bajo activo)', desc:'En LOW habilita las 16 salidas PWM. Útil para parada de emergencia.' },
  CH:    { name:'Canal de servo (CH0–CH15)', volts:'pulso 50 Hz', desc:'Cada canal expone S (señal) / V+ (potencia) / GND.' }
};

export class PinDetail {
  constructor() {
    this.root = document.createElement('div');
    this.root.className = 'pin-detail';
    this.root.innerHTML = `
      <button class="pd-close" aria-label="Cerrar">x</button>
      <div class="pd-header">
        <span class="pd-type" id="pd-type"></span>
        <h3 class="pd-name" id="pd-name"></h3>
        <div class="pd-sub" id="pd-sub"></div>
      </div>
      <div class="pd-row"><span>Voltaje</span><strong id="pd-volts"></strong></div>
      <div class="pd-row"><span>Función</span><strong id="pd-fn"></strong></div>
      <p class="pd-desc" id="pd-desc"></p>
    `;
    document.body.appendChild(this.root);
    this.root.querySelector('.pd-close').onclick = () => this.hide();
  }

  show(pin) {
    const info = TYPE_INFO[pin.type] || TYPE_INFO.IO;
    this.root.querySelector('#pd-type').textContent = pin.type;
    this.root.querySelector('#pd-name').textContent = pin.label;
    this.root.querySelector('#pd-sub').textContent = pin.sub || pin.id;
    this.root.querySelector('#pd-volts').textContent = info.volts;
    this.root.querySelector('#pd-fn').textContent = info.name;
    this.root.querySelector('#pd-desc').textContent = info.desc;
    this.root.classList.add('visible');
  }

  hide() { this.root.classList.remove('visible'); }
}
