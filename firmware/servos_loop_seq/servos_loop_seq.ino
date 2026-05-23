// servos_loop_seq — los 4 servos oscilan 0° <-> 180° en secuencia (uno a la vez).
// Mientras un servo se mueve, los otros 3 quedan sostenidos en 0°.
// V+ del PCA9685 toma 5V del USB del ESP32, por eso solo movemos uno a la vez.

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

static const uint8_t  SDA_PIN      = 21;
static const uint8_t  SCL_PIN      = 22;
static const uint32_t I2C_FREQ_HZ  = 400000;
static const uint8_t  PCA_ADDR     = 0x40;
static const float    PWM_FREQ_HZ  = 50.0f;

static const uint16_t SERVO_PULSE_MIN = 150;
static const uint16_t SERVO_PULSE_MAX = 600;

static const uint8_t  SERVO_CHANNELS[] = {0, 7, 8, 15};
static const uint8_t  NUM_SERVOS = sizeof(SERVO_CHANNELS) / sizeof(SERVO_CHANNELS[0]);

static const uint8_t  MIN_DEG       = 0;
static const uint8_t  MAX_DEG       = 180;
static const uint8_t  STEP_DEG      = 3;     // pasos chicos = inrush controlado
static const uint16_t STEP_MS       = 20;    // velocidad
static const uint16_t REST_BETWEEN  = 400;   // pausa entre extremos
static const uint16_t REST_BETWEEN_SERVOS = 500;  // pausa al pasar al siguiente servo

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
  Serial.println(F("\n=== servos_loop_seq ==="));
  Serial.printf("Canales: ");
  for (uint8_t i = 0; i < NUM_SERVOS; i++) Serial.printf("%u ", SERVO_CHANNELS[i]);
  Serial.println();
  Serial.printf("Oscilando %u° <-> %u° uno a la vez (paso %u°, %u ms)\n",
                MIN_DEG, MAX_DEG, STEP_DEG, STEP_MS);

  Wire.begin(SDA_PIN, SCL_PIN, I2C_FREQ_HZ);
  if (!pwm.begin()) {
    Serial.println(F("[ERROR] PCA9685 no responde"));
    while (true) { delay(1000); }
  }
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(PWM_FREQ_HZ);

  // Arranque seguro: todos los servos a 0°, secuencialmente
  Serial.println(F("Colocando todos en 0° al arranque..."));
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    setServoDeg(SERVO_CHANNELS[i], 0);
    delay(400);
  }
  delay(600);
  Serial.println(F("Empieza el loop secuencial."));
}

void loop() {
  for (uint8_t i = 0; i < NUM_SERVOS; i++) {
    uint8_t ch = SERVO_CHANNELS[i];

    Serial.printf("[ch%u] 0° -> 180°\n", ch);
    slowMove(ch, MIN_DEG, MAX_DEG);
    delay(REST_BETWEEN);

    Serial.printf("[ch%u] 180° -> 0°\n", ch);
    slowMove(ch, MAX_DEG, MIN_DEG);
    delay(REST_BETWEEN_SERVOS);
  }
}
