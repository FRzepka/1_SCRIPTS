#include <WiFiS3.h>
#include <ArduinoOTA.h>
#include "arduino_secrets.h"

// WLAN-Zugangsdaten in arduino_secrets.h
const char* ssid     = SECRET_SSID;
const char* password = SECRET_PASS;

// OTA-Daten
const char* otaName     = "uno-r4-wifi";  // der Name, wie er im IDE-Port erscheint
const char* otaPassword = "password";     // Default-Passwort

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // WLAN verbinden
  WiFi.begin(ssid, password);
  Serial.print("Verbinde mit WLAN");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print('.');
  }
  Serial.println("\nVerbunden, IP: " + WiFi.localIP().toString());

  // OTA-Dienst starten: IP, Hostname, Passwort, Storage
  // InternalStorage ist der Flash-Speicher des MCU
  ArduinoOTA.begin(WiFi.localIP(), otaName, otaPassword, InternalStorage);

  Serial.println("OTA bereit – wähle im IDE-Port: " + String(otaName) +
                 " at " + WiFi.localIP().toString());
}

void loop() {
  // OTA-Anfragen abarbeiten
  ArduinoOTA.poll();
}
