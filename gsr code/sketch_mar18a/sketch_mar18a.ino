#include <WiFi.h>
#include <HTTPClient.h>

// --- CONFIGURATION ---
const char* ssid = "FE";          // ENTER YOUR WI-FI NAME
const char* password = "qwertyuiop";  // ENTER YOUR WI-FI PASSWORD

// IMPORTANT: Replace with your laptop's local IP address running the Flask app
const char* serverName = "http://10.204.94.57:5000/api/hardware"; 

const int GSR_PIN = 34; // ADC1 pin for clean Wi-Fi analog reading

void setup() {
  Serial.begin(115200);
  pinMode(GSR_PIN, INPUT);
  
  Serial.print("Connecting to Wi-Fi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nCONNECTED! IP Address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    // The ESP32 12-bit ADC reads values from 0 to 4095
    int gsrRaw = analogRead(GSR_PIN);
    
    // Create the JSON payload
    String httpRequestData = "{\"gsr\": " + String(gsrRaw) + "}";

    // Send the POST request
    int httpResponseCode = http.POST(httpRequestData);
    
    if (httpResponseCode > 0) {
      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
      Serial.println("Payload sent: " + httpRequestData);
    } else {
      Serial.print("Error code: ");
      Serial.println(httpResponseCode);
    }
    http.end();
  } else {
    Serial.println("WiFi Disconnected");
  }
  
  // Throttle updates to 2 Hz (every 500ms) to prevent crashing the Flask server
  delay(500); 
}