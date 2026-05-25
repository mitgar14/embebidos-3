/**
 * Fuente de verdad: pines, conexiones y pasos.
 *
 * Topología:
 *   PSU (externo) —
 *                  v
 *   Expansion Board <— ESP32 (embebido, secundario)
 *        |
 *        v
 *      PCA9685
 *
 * Las conexiones primarias salen de la EXPANSION BOARD (bloques I2C y de potencia).
 * Los pines del ESP32 quedan como secundarios — visibles solo con el toggle "todos".
 */

export const WIRE_COLORS = {
  GND:    0x111111,
  VCC:    0xd62828,
  SDA:    0x0077ff,
  SCL:    0xf1c40f,
  OE:     0xffffff,
  VPLUS:  0xb71c1c,
  SIGNAL: 0x8e44ad
};

/* ════════════════════════════════════════════════════════════
   1. ESP32-WROOM-32 DOIT DevKit V1 — pinout completo (referencia)
   ════════════════════════════════════════════════════════════ */
export const ESP32_PINS = {
  'EN':    { side: 'left', row: 0,  label: 'EN',   type: 'EN',    sub: 'Reset' },
  'VP':    { side: 'left', row: 1,  label: 'VP',   type: 'ADC',   sub: 'GPIO36' },
  'VN':    { side: 'left', row: 2,  label: 'VN',   type: 'ADC',   sub: 'GPIO39' },
  'D34':   { side: 'left', row: 3,  label: 'D34',  type: 'IN',    sub: 'GPIO34' },
  'D35':   { side: 'left', row: 4,  label: 'D35',  type: 'IN',    sub: 'GPIO35' },
  'D32':   { side: 'left', row: 5,  label: 'D32',  type: 'IO',    sub: 'GPIO32' },
  'D33':   { side: 'left', row: 6,  label: 'D33',  type: 'IO',    sub: 'GPIO33' },
  'D25':   { side: 'left', row: 7,  label: 'D25',  type: 'IO',    sub: 'GPIO25' },
  'D26':   { side: 'left', row: 8,  label: 'D26',  type: 'IO',    sub: 'GPIO26' },
  'D27':   { side: 'left', row: 9,  label: 'D27',  type: 'IO',    sub: 'GPIO27' },
  'D14':   { side: 'left', row: 10, label: 'D14',  type: 'IO',    sub: 'GPIO14' },
  'D12':   { side: 'left', row: 11, label: 'D12',  type: 'IO',    sub: 'GPIO12' },
  'D13':   { side: 'left', row: 12, label: 'D13',  type: 'IO',    sub: 'GPIO13' },
  'GND_L': { side: 'left', row: 13, label: 'GND',  type: 'GND' },
  'VIN':   { side: 'left', row: 14, label: 'VIN',  type: 'POWER', sub: '5 V' },

  'D23':   { side: 'right', row: 0,  label: 'D23', type: 'IO',   sub: 'GPIO23' },
  'D22':   { side: 'right', row: 1,  label: 'D22', type: 'SCL',  sub: 'GPIO22' },
  'TX0':   { side: 'right', row: 2,  label: 'TX0', type: 'UART', sub: 'GPIO1' },
  'RX0':   { side: 'right', row: 3,  label: 'RX0', type: 'UART', sub: 'GPIO3' },
  'D21':   { side: 'right', row: 4,  label: 'D21', type: 'SDA',  sub: 'GPIO21' },
  'D19':   { side: 'right', row: 5,  label: 'D19', type: 'IO',   sub: 'GPIO19' },
  'D18':   { side: 'right', row: 6,  label: 'D18', type: 'IO',   sub: 'GPIO18' },
  'D5':    { side: 'right', row: 7,  label: 'D5',  type: 'IO',   sub: 'GPIO5' },
  'D17':   { side: 'right', row: 8,  label: 'D17', type: 'UART', sub: 'GPIO17 · TX2' },
  'D16':   { side: 'right', row: 9,  label: 'D16', type: 'UART', sub: 'GPIO16 · RX2' },
  'D4':    { side: 'right', row: 10, label: 'D4',  type: 'IO',   sub: 'GPIO4' },
  'D2':    { side: 'right', row: 11, label: 'D2',  type: 'IO',   sub: 'GPIO2' },
  'D15':   { side: 'right', row: 12, label: 'D15', type: 'IO',   sub: 'GPIO15' },
  'GND_R': { side: 'right', row: 13, label: 'GND', type: 'GND' },
  '3V3':   { side: 'right', row: 14, label: '3V3', type: 'POWER', sub: '3.3 V' }
};

/* ════════════════════════════════════════════════════════════
   2. Expansion Board — bloques principales
   ════════════════════════════════════════════════════════════ */

// 13 GPIOs por lado se exponen como tripletes S(signal)/V(voltaje)/G(GND)
export const EXP_TRIPLET_GPIOS = {
  left:  ['EN','VP','VN','D34','D35','D32','D33','D25','D26','D27','D14','D12','D13'],
  right: ['D23','D22','TX0','RX0','D21','D19','D18','D5','D17','D16','D4','D2','D15']
};

// Pines "macro" de la expansion board: los conectables/etiquetables principales
export const EXP_PINS = {
  // Bloque I2C dedicado (esquina inferior-derecha)
  'I2C_D22':  { region: 'i2c', row: 0, label: 'D22', type: 'SCL', sub: 'I²C SCL' },
  'I2C_D21':  { region: 'i2c', row: 1, label: 'D21', type: 'SDA', sub: 'I²C SDA' },
  'I2C_VCC':  { region: 'i2c', row: 2, label: 'VCC', type: 'VCC', sub: 'jumper 3V3/5V' },
  'I2C_GND':  { region: 'i2c', row: 3, label: 'GND', type: 'GND' },

  // Bloque de alimentación (esquina inferior-izquierda)
  'PWR_GND':  { region: 'pwr', col: 0, label: 'GND', type: 'GND' },
  'PWR_5V':   { region: 'pwr', col: 1, label: '5V',  type: 'POWER', sub: '5 V salida' },
  'PWR_3V3':  { region: 'pwr', col: 2, label: '3V3', type: 'POWER', sub: '3.3 V salida' },

  // Pin S del triplete D5 (usado para OE en paso 6)
  'D5_S':     { region: 'triplet', side: 'right', gpio: 'D5',
                label: 'D5 · S', type: 'IO', sub: 'Señal triplete' }
};

/* ════════════════════════════════════════════════════════════
   3. PCA9685 — header lógico + V+ terminal
   ════════════════════════════════════════════════════════════ */
export const PCA_PINS = {
  'GND':   { row: 0, label: 'GND', type: 'GND' },
  'OE':    { row: 1, label: 'OE',  type: 'OE',    sub: 'Output Enable' },
  'SCL':   { row: 2, label: 'SCL', type: 'SCL',   sub: 'Reloj I²C' },
  'SDA':   { row: 3, label: 'SDA', type: 'SDA',   sub: 'Datos I²C' },
  'VCC':   { row: 4, label: 'VCC', type: 'VCC',   sub: '2.3–5.5 V' },
  'VPLUS': { row: 5, label: 'V+',  type: 'VPLUS', sub: 'Potencia servos' }
};

/* ════════════════════════════════════════════════════════════
   4. Pasos — todas las conexiones primarias salen de la expansion board
   ════════════════════════════════════════════════════════════ */
export const STEPS = [
  {
    id: 'gnd',
    title: '1 · Tierra común (GND)',
    from: { board: 'exp', pin: 'PWR_GND' },
    to:   { board: 'pca', pin: 'GND' },
    color: 'GND',
    required: true,
    description:
      'Conecta cualquier GND del bloque de alimentación de la expansion board al GND del header ' +
      'lógico del PCA9685. La expansion ya enruta el GND del ESP32 a un bloque accesible (3 pines ' +
      'duplicados). Sin tierra común el bus I²C falla aunque las señales lleguen.',
    code: '// Bloque PWR: 3x GND, 3x 5V, 3x 3V3 — todos compartidos.'
  },
  {
    id: 'vcc',
    title: '2 · Alimentación lógica (VCC)',
    from: { board: 'exp', pin: 'I2C_VCC' },
    to:   { board: 'pca', pin: 'VCC' },
    color: 'VCC',
    required: true,
    description:
      'El bloque I²C dedicado de la expansion board agrupa D22/D21/VCC/GND en una sola tira: ' +
      'ideal para cablear el PCA9685. La línea VCC depende del jumper 3V3/5V; déjalo en 3V3 ' +
      'para emparejar el nivel lógico del ESP32.',
    code: '// Jumper en 3V3 → VCC = 3.3 V (recomendado).'
  },
  {
    id: 'sda',
    title: '3 · I²C — SDA (datos)',
    from: { board: 'exp', pin: 'I2C_D21' },
    to:   { board: 'pca', pin: 'SDA' },
    color: 'SDA',
    required: true,
    description:
      'D21 (GPIO21) del ESP32 ya está enrutado por la expansion al pin SDA del bloque I²C. ' +
      'Conecta de ahí al SDA del PCA9685. Las pull-ups (10 kΩ) están a bordo del PCA9685.',
    code: 'Wire.begin(21, 22); // SDA=21, SCL=22'
  },
  {
    id: 'scl',
    title: '4 · I²C — SCL (reloj)',
    from: { board: 'exp', pin: 'I2C_D22' },
    to:   { board: 'pca', pin: 'SCL' },
    color: 'SCL',
    required: true,
    description:
      'D22 (GPIO22) → SCL. La expansion board te ahorra cables al tener D22 justo al lado de ' +
      'D21. Dirección por defecto del PCA9685 = 0x40.',
    code: 'pwm.begin(); pwm.setPWMFreq(50);'
  },
  {
    id: 'vplus',
    title: '5 · V+ — Alimentación de servos (5V del expansor)',
    from: { board: 'exp', pin: 'PWR_5V' },
    to:   { board: 'pca', pin: 'TERM_V+' },
    color: 'VPLUS',
    required: true,
    description:
      'V+ recibe los 5 V del bloque PWR del expansor (mismo riel que alimenta el ESP32). Esto ' +
      'permite mover servos pequeños (SG90) sin fuente externa, PERO compartes el regulador ' +
      'AMS1117 (~1 A): si dos servos arrancan a la vez bajo carga, el ESP32 hará brownout. ' +
      'Para uso serio o servos MG996R, usa una fuente externa y une GND común.',
    code: '// CH0..CH15 ahora reciben 5V en V+ del header — los servos se mueven.'
  },
  {
    id: 'oe',
    title: '6 · OE — Output Enable (opcional)',
    from: { board: 'exp', pin: 'D5_S' },
    to:   { board: 'pca', pin: 'OE' },
    color: 'OE',
    required: false,
    description:
      'OE es activo en BAJO. Lo conectamos al pin S (signal) del triplete D5 — es el pin de la ' +
      'columna amarilla. Permite cortar todas las salidas PWM en una sola instrucción.',
    code: 'pinMode(5, OUTPUT); digitalWrite(5, HIGH); // todas las salidas off'
  }
];

export const BOARD_INFO = {
  exp: {
    name: 'ESP32 Expansion Board (Electrodemy-style)',
    voltage: 'Selector 3V3/5V · Entrada DC 6.5–16 V o USB',
    notes: '26 tripletes S/V/G + bloque I²C dedicado + 3x GND / 3x 5V / 3x 3V3'
  },
  esp32: {
    name: 'ESP32-WROOM-32 (DevKit V1, 30 pines)',
    voltage: '3.3 V lógica · 5 V por USB/VIN',
    notes: 'Embebido sobre la expansion board. I²C: SDA=21, SCL=22'
  },
  pca: {
    name: 'PCA9685 (16 canales PWM, 12-bit)',
    voltage: 'VCC 2.3–5.5 V · V+ hasta 6 V',
    notes: 'I²C addr 0x40 · pull-ups 10 kΩ a bordo'
  }
};
