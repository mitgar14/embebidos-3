// I2C scanner para ESP32 + PCA9685
// Pines I2C ESP32 por defecto: SDA=GPIO21, SCL=GPIO22
// PCA9685 default: 0x40 (datos), 0x70 (All Call)

#include <Wire.h>

static const int SDA_PIN = 21;
static const int SCL_PIN = 22;
static const uint32_t I2C_FREQ_HZ = 100000;  // 100 kHz, conservador

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println();
  Serial.println(F("=== I2C scanner ESP32 ==="));
  Serial.printf("SDA=GPIO%d  SCL=GPIO%d  Freq=%lu Hz\n", SDA_PIN, SCL_PIN, (unsigned long)I2C_FREQ_HZ);

  Wire.begin(SDA_PIN, SCL_PIN, I2C_FREQ_HZ);
}

void loop() {
  Serial.println(F("\nEscaneando bus I2C..."));
  uint8_t found = 0;

  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();

    if (err == 0) {
      Serial.printf("  -> Dispositivo en 0x%02X", addr);
      if (addr == 0x40) Serial.print(F("  [PCA9685 datos default]"));
      if (addr == 0x70) Serial.print(F("  [PCA9685 All Call]"));
      Serial.println();
      found++;
    } else if (err == 4) {
      Serial.printf("  -> Error desconocido en 0x%02X\n", addr);
    }
  }

  if (found == 0) {
    Serial.println(F("  (ningun dispositivo respondio)"));
    Serial.println(F("  Verificar: SDA/SCL, VCC del PCA9685, GND comun"));
  } else {
    Serial.printf("Total: %u dispositivo(s) en el bus.\n", found);
  }

  delay(5000);
}
