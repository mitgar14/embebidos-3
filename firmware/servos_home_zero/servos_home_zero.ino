// servos_home_zero — manda los 4 servos a 0° y los deja sostenidos.
// V+ desde USB 5V del ESP32: vamos uno por uno para evitar pico de corriente.
// Si algun servo zumba al llegar a 0° (golpea end-stop mecanico), subir HOME_DEG.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static const uint8_t  SDA_PIN      = 21;
static const uint8_t  SCL_PIN      = 22;
static const uint32_t I2C_FREQ_HZ  = 400000;
static const uint8_t  PCA_ADDR     = 0x40;
static const float    PWM_FREQ_HZ  = 50.0f;

// Calibracion SG90 (mismos valores que en servo_sweep.ino)
static const uint16_t SERVO_PULSE_MIN = 150;
static const uint16_t SERVO_PULSE_MAX = 600;

// Canales reales donde hay servos conectados en el PCA9685
static const uint8_t SERVO_CHANNELS[] = {0, 7, 8, 15};
static const uint8_t NUM_SERVOS = sizeof(SERVO_CHANNELS) / sizeof(SERVO_CHANNELS[0]);

// Posicion objetivo. 0 = extremo segun calibracion (~150 counts ~ 733 us).
// Si zumba: subir a 5 o 10.
static const uint8_t HOME_DEG = 0;

// Asumimos que el servo puede estar en cualquier posicion. Para no dar tirones,
// arrancamos comandando ~90° y descendemos gradualmente.
static const uint8_t START_DEG = 90;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA_ADDR);

uint16_t degToPulse(uint8_t deg) {
  return map(deg, 0, 180, SERVO_PULSE_MIN, SERVO_PULSE_MAX);
}

void setServoDeg(uint8_t ch, uint8_t deg) {
  pwm.setPWM(ch, 0, degToPulse(deg));
}

void releaseServo(uint8_t ch) {
  pwm.setPWM(ch, 0, 4096);
}

void slowRamp(uint8_t ch, int16_t fromDeg, int16_t toDeg, uint8_t stepDeg, uint16_t stepDelayMs) {
  int8_t dir = (toDeg >= fromDeg) ? +1 : -1;
  for (int16_t d = fromDeg; (dir > 0) ? (d <= toDeg) : (d >= toDeg); d += dir * stepDeg) {
    setServoDeg(ch, (uint8_t)d);
    delay(stepDelayMs);
  }
  setServoDeg(ch, (uint8_t)toDeg);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("\n=== servos_home_zero ==="));
  Serial.printf("Objetivo: los %u servos a %u° y mantenidos.\n", NUM_SERVOS, HOME_DEG);

  Wire.begin(SDA_PIN, SCL_PIN, I2C_FREQ_HZ);
  if (!pwm.begin()) {
    Serial.println(F("[ERROR] PCA9685 no responde. Revisar I2C."));
    while (true) { delay(1000); }
  }
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(PWM_FREQ_HZ);

  // Estado seguro al arranque: liberar TODOS los canales (0..15)
  for (uint8_t ch = 0; ch < 16; ch++) releaseServo(ch);
  delay(800);

  // Bajar cada servo a HOME_DEG, uno a uno
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    uint8_t ch = SERVO_CHANNELS[i];
    Serial.printf("  canal %u: %u° -> %u°\n", ch, START_DEG, HOME_DEG);
    setServoDeg(ch, START_DEG);
    delay(400);
    slowRamp(ch, START_DEG, HOME_DEG, 3, 25);
    // Lo dejamos sostenido en HOME_DEG (PWM activo)
    setServoDeg(ch, HOME_DEG);
    delay(400);
  }

  Serial.println(F("\nLos 4 servos quedaron en 0° sostenidos."));
  Serial.println(F("Si alguno zumba: subir HOME_DEG en el sketch (probar 5, 10)."));
}

void loop() {
  // Mantener los pulsos activos: la libreria ya conserva el ultimo setPWM en el chip,
  // pero re-afirmamos cada 5s por si hay glitch en el bus.
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    setServoDeg(SERVO_CHANNELS[i], HOME_DEG);
  }
  Serial.println(F("[hb] sosteniendo 0°"));
  delay(5000);
}
