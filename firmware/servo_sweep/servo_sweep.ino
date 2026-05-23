// Servo sweep ESP32 + PCA9685 — modo cauto (V+ desde 5V del ESP32 via USB)
// Estrategia:
//   - 1 servo activo a la vez
//   - Pasos chicos con delay (limita inrush)
//   - Liberar PWM entre pruebas (servo desenergizado, sin hold current)
//   - Fase 1: wiggle estrecho 80°-100° para detectar interferencia mecanica
//   - Fase 2: barrido 30°-150°
//   - Fase 3: centro 90° y liberar

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static const uint8_t  SDA_PIN      = 21;
static const uint8_t  SCL_PIN      = 22;
static const uint32_t I2C_FREQ_HZ  = 400000;   // 400 kHz, soportado por PCA9685
static const uint8_t  PCA_ADDR     = 0x40;
static const float    PWM_FREQ_HZ  = 50.0f;    // estandar servos hobby

// Calibracion pulso para SG90 (12-bit, 0..4095). 50 Hz -> 20 ms periodo.
// 150 = ~733 us  (0°)
// 600 = ~2930 us (180°)
static const uint16_t SERVO_PULSE_MIN = 150;
static const uint16_t SERVO_PULSE_MAX = 600;

static const uint8_t NUM_SERVOS = 4;
static const uint8_t CENTER_DEG = 90;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA_ADDR);

uint16_t degToPulse(uint8_t deg) {
  // map respetando rango calibrado de SG90
  return map(deg, 0, 180, SERVO_PULSE_MIN, SERVO_PULSE_MAX);
}

void setServoDeg(uint8_t ch, uint8_t deg) {
  pwm.setPWM(ch, 0, degToPulse(deg));
}

void releaseServo(uint8_t ch) {
  // bit 12 del registro OFF_H pone la salida en OFF total (sin PWM)
  pwm.setPWM(ch, 0, 4096);
}

void releaseAll() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) releaseServo(i);
}

void slowMove(uint8_t ch, int16_t fromDeg, int16_t toDeg, uint8_t stepDeg = 5, uint16_t stepDelayMs = 30) {
  int8_t dir = (toDeg >= fromDeg) ? +1 : -1;
  for (int16_t d = fromDeg; (dir > 0) ? (d <= toDeg) : (d >= toDeg); d += dir * stepDeg) {
    setServoDeg(ch, (uint8_t)d);
    delay(stepDelayMs);
  }
  setServoDeg(ch, (uint8_t)toDeg);  // garantiza llegar al destino
}

void testChannel(uint8_t ch) {
  Serial.printf("\n--- Servo en canal %u ---\n", ch);

  // Fase 1: wiggle pequeno 80-100
  Serial.println(F("  [1] wiggle 80-100 (busca chasquidos / interferencia)"));
  setServoDeg(ch, CENTER_DEG); delay(500);
  slowMove(ch, 90, 80, 2, 40);
  delay(300);
  slowMove(ch, 80, 100, 2, 40);
  delay(300);
  slowMove(ch, 100, 90, 2, 40);
  delay(500);

  // Fase 2: barrido medio 60-120
  Serial.println(F("  [2] barrido medio 60-120"));
  slowMove(ch, 90, 60, 5, 30);
  delay(300);
  slowMove(ch, 60, 120, 5, 30);
  delay(300);
  slowMove(ch, 120, 90, 5, 30);
  delay(500);

  // Fase 3: barrido amplio 30-150
  Serial.println(F("  [3] barrido amplio 30-150"));
  slowMove(ch, 90, 30, 5, 30);
  delay(300);
  slowMove(ch, 30, 150, 5, 30);
  delay(300);
  slowMove(ch, 150, 90, 5, 30);
  delay(500);

  // Liberar
  releaseServo(ch);
  Serial.printf("  [ok] canal %u liberado (sin PWM)\n", ch);
  delay(800);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("\n=== Servo sweep ESP32+PCA9685 (modo cauto USB) ==="));
  Serial.println(F("V+ del PCA9685 toma 5V del ESP32. Un servo a la vez."));
  Serial.printf("SDA=%u SCL=%u  PCA@0x%02X  PWM=%.0fHz  N=%u\n",
                SDA_PIN, SCL_PIN, PCA_ADDR, PWM_FREQ_HZ, NUM_SERVOS);

  Wire.begin(SDA_PIN, SCL_PIN, I2C_FREQ_HZ);
  if (!pwm.begin()) {
    Serial.println(F("[ERROR] PCA9685 no responde. Revisar I2C."));
    while (true) { delay(1000); }
  }
  pwm.setOscillatorFrequency(27000000);  // ajuste empirico Adafruit p/ RC interno
  pwm.setPWMFreq(PWM_FREQ_HZ);

  releaseAll();  // arranque seguro: sin pulso
  Serial.println(F("PCA9685 OK. Servos en estado libre."));
  delay(1500);  // settling time

  // Centrar uno por uno antes del test (asi no hay tirones desde posicion random)
  Serial.println(F("\nCentrando los 4 servos a 90° (uno por uno)..."));
  for (uint8_t ch = 0; ch < NUM_SERVOS; ch++) {
    setServoDeg(ch, CENTER_DEG);
    delay(600);
    releaseServo(ch);
    delay(200);
  }
  Serial.println(F("Listo. Empieza la batida de tests.\n"));
  delay(1000);
}

void loop() {
  for (uint8_t ch = 0; ch < NUM_SERVOS; ch++) {
    testChannel(ch);
  }
  Serial.println(F("\n=== Ciclo completo. Reposando 5s antes de repetir... ==="));
  delay(5000);
}
