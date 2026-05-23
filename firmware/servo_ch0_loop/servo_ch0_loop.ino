// servo_ch0_loop — el servo del canal 0 oscila 0° <-> 180° en loop.
// Canales 1-3 quedan sostenidos en 0° (siguen como en home-zero).
// V+ desde USB 5V del ESP32: solo 1 servo se mueve, los demas hold de baja corriente.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static const uint8_t  SDA_PIN      = 21;
static const uint8_t  SCL_PIN      = 22;
static const uint32_t I2C_FREQ_HZ  = 400000;
static const uint8_t  PCA_ADDR     = 0x40;
static const float    PWM_FREQ_HZ  = 50.0f;

static const uint16_t SERVO_PULSE_MIN = 150;
static const uint16_t SERVO_PULSE_MAX = 600;

static const uint8_t  TARGET_CH = 0;
// Los OTROS servos quedan sostenidos en 0°
static const uint8_t  HOLD_CHANNELS[] = {7, 8, 15};
static const uint8_t  NUM_HOLD = sizeof(HOLD_CHANNELS) / sizeof(HOLD_CHANNELS[0]);
static const uint8_t  MIN_DEG   = 0;
static const uint8_t  MAX_DEG   = 180;
static const uint8_t  STEP_DEG  = 3;     // pasos chicos = inrush controlado
static const uint16_t STEP_MS   = 20;    // velocidad de barrido

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(PCA_ADDR);

uint16_t degToPulse(uint8_t deg) {
  return map(deg, 0, 180, SERVO_PULSE_MIN, SERVO_PULSE_MAX);
}

void setServoDeg(uint8_t ch, uint8_t deg) {
  pwm.setPWM(ch, 0, degToPulse(deg));
}

void slowMove(uint8_t ch, int16_t fromDeg, int16_t toDeg) {
  int8_t dir = (toDeg >= fromDeg) ? +1 : -1;
  for (int16_t d = fromDeg; (dir > 0) ? (d <= toDeg) : (d >= toDeg); d += dir * STEP_DEG) {
    setServoDeg(ch, (uint8_t)d);
    delay(STEP_MS);
  }
  setServoDeg(ch, (uint8_t)toDeg);
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println(F("\n=== servo_ch0_loop ==="));
  Serial.printf("Canal %u oscilando %u° <-> %u° (paso %u°, %u ms)\n",
                TARGET_CH, MIN_DEG, MAX_DEG, STEP_DEG, STEP_MS);

  Wire.begin(SDA_PIN, SCL_PIN, I2C_FREQ_HZ);
  if (!pwm.begin()) {
    Serial.println(F("[ERROR] PCA9685 no responde"));
    while (true) { delay(1000); }
  }
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(PWM_FREQ_HZ);

  // Servos en 7, 8, 15 quedan sostenidos en 0°
  for (uint8_t i = 0; i < NUM_HOLD; i++) {
    setServoDeg(HOLD_CHANNELS[i], 0);
  }
  delay(500);

  // Llevar canal 0 a 0° antes de arrancar el loop
  setServoDeg(TARGET_CH, 0);
  delay(800);
  Serial.println(F("Empieza el loop."));
}

void loop() {
  Serial.println(F("[ch0] 0° -> 180°"));
  slowMove(TARGET_CH, MIN_DEG, MAX_DEG);
  delay(300);

  Serial.println(F("[ch0] 180° -> 0°"));
  slowMove(TARGET_CH, MAX_DEG, MIN_DEG);
  delay(300);
}
