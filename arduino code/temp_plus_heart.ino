#include <PulseSensorPlayground.h>     // Includes the PulseSensorPlayground Library.   

// Pulse Sensor Variables
const int PulseWire = 0;       // PulseSensor PURPLE WIRE connected to ANALOG PIN 0
const int LED = LED_BUILTIN;   // The on-board Arduino LED, close to PIN 13.
int Threshold = 550;           // Threshold for detecting heartbeat
PulseSensorPlayground pulseSensor;  // PulseSensor object

// Temperature Sensor Variables
const int lm35_pin = A1;  /* LM35 O/P pin */

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Setup PulseSensor
  pulseSensor.analogInput(PulseWire);   
  pulseSensor.blinkOnPulse(LED);  // Auto-magically blink Arduino's LED with heartbeat
  pulseSensor.setThreshold(Threshold);

  // Confirm that the PulseSensor is working
  if (pulseSensor.begin()) {
    Serial.println("PulseSensor object created!");
  }
}

void loop() {
  // Read temperature data
  int temp_adc_val = analogRead(lm35_pin);  /* Read Temperature */
  float temp_val = (temp_adc_val * 4.88);   /* Convert ADC value to voltage */
  temp_val = (temp_val / 10);  /* LM35 gives output of 10mv/°C */
  
  // Print temperature data
  Serial.print("Temperature = ");
  Serial.print(temp_val);
  Serial.println(" Degree Celsius");

  // Check if heartbeat is detected
  if (pulseSensor.sawStartOfBeat()) {
    int myBPM = pulseSensor.getBeatsPerMinute();  // Get BPM
    Serial.println("♥  A HeartBeat Happened!");
    Serial.print("BPM: ");
    Serial.println(myBPM);
  }

  // Delay for readability and to avoid overwhelming the serial monitor
  delay(1000);
}
