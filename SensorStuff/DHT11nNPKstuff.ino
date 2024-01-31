#include <DHT.h>
#include <SoftwareSerial.h>

// Set the DHT sensor type (DHT11 or DHT22)
#define DHT_TYPE DHT11

// Pin to which the DHT sensor is connected
#define DHT_PIN 2

DHT dht(DHT_PIN, DHT_TYPE);

// Modbus RTU requests for reading NPK values
const byte nitro[] = {0x01,0x03, 0x00, 0x1e, 0x00, 0x01, 0xe4, 0x0c};
const byte phos[] = {0x01,0x03, 0x00, 0x1f, 0x00, 0x01, 0xb5, 0xcc};
const byte pota[] = {0x01,0x03, 0x00, 0x20, 0x00, 0x01, 0x85, 0xc0};

// A variable used to store NPK values
byte values[8];

SoftwareSerial mod(8, 9);

void setup() {
  Serial.begin(9600);
  Serial.println("DHT11 Sensor Reading:");
  dht.begin();

  // Set the baud rate for the SerialSoftware object
  mod.begin(4800);
  delay(500);
}

void loop() {
  // DHT Sensor Reading
  delay(2000); // Wait for 2 seconds between readings
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();

  // Check if any reads failed and exit early (to try again).
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("Failed to read from DHT sensor!");
    return;
  }

  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.print("°C\t");

  Serial.print("Humidity: ");
  Serial.print(humidity);
  Serial.println("%");

  // NPK Sensor Reading
  byte val1,val2,val3;
  val1 = nitrogen();
  delay(250);
  val2 = phosphorous();
  delay(250);
  val3 = potassium();
  delay(250);

  // Print values to the serial monitor
  Serial.print("Nitrogen: ");
  Serial.print(val1);
  Serial.println(" mg/kg");
  Serial.print("Phosphorous: ");
  Serial.print(val2);
  Serial.println(" mg/kg");
  Serial.print("Potassium: ");
  Serial.print(val3);
  Serial.println(" mg/kg");

  delay(2000);
}

byte nitrogen(){
  mod.write(nitro,8);
  delay(100);
  for(byte i=0;i<7;i++){
    Serial.print(mod.read(),HEX);
    values[i] = mod.read();
    Serial.print(values[i],HEX);
  }
  Serial.println();
  return values[4];
}

byte phosphorous(){
  mod.write(phos,8);
  for(byte i=0;i<7;i++){
    values[i] = mod.read();
    Serial.print(values[i],HEX);
  }
  Serial.println();
  return values[4];
}

byte potassium(){
  mod.write(pota,8);
  for(byte i=0;i<7;i++){
    values[i] = mod.read();
    Serial.print(values[i],HEX);
  }
  Serial.println();
  return values[4];
}