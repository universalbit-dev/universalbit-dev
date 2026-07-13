/*
==============================================================================
  UniversalBit Project - Fibonacci NTP Clock Firmware
  Repository Path: ESP8266/fibonacci_ntp/fibonacci_ntp.ino
==============================================================================
*/

#include <ESP8266WiFi.h>
#include <time.h>

// --- Configuration Network Profiles ---
const char* ssid             = "Guest Wifi-Name";     // Replace with your WiFi SSID
const char* password         = "Guest Password";      // Replace with your WiFi Password
const char* ntpServer        = "pool.ntp.org";

// --- Timezone Offset Calculations ---
const long gmtOffset_sec     = 3600;                  // 1 Hour offset = 3600 seconds (e.g., UTC+1)
const int daylightOffset_sec = 0;                     // Daylight savings offset in seconds

// Forward Declarations
void printLocalTime();

void setup() {
  // Boost communication baud speed to standard high-speed profile
  Serial.begin(115200);
  Serial.println("\n\n====================================");
  Serial.println("  UniversalBit NTP Clock Initializing  ");
  Serial.println("====================================");

  // Initialize Station Mode interface
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.printf("[WIFI] Connecting to target SSID: %s ", ssid);

  // Guard against permanent hangs using an explicit connection timeout gate
  int timeout_counter = 0;
  while (WiFi.status() != WL_CONNECTED && timeout_counter < 30) {
    delay(500);
    Serial.print(".");
    timeout_counter++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WIFI] Connected successfully!");
    Serial.print("[WIFI] Assigned Local IP Address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\n[WARN] Connection timed out. Running on internal RTC tracking fallback...");
  }

  // Bind the core NTP background synchronization engine service layer 
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
}

void loop() {
  // Non-blocking delay execution tracking to display time updates every second
  static unsigned long last_update = 0;
  if (millis() - last_update >= 1000) {
    last_update = millis();
    printLocalTime();
  }
}

void printLocalTime() {
  struct tm timeinfo;
  time_t now;
  
  time(&now);                       // Fetch current internal system epoch timestamp
  localtime_r(&now, &timeinfo);     // Parse epoch metrics cleanly into the structured time container

  // Verify if the system has completed its baseline synchronization handshake
  if (timeinfo.tm_year < (2016 - 1900)) { 
    Serial.println("[TIME] Syncing with pool.ntp.org servers...");
    return;
  }

  // Dynamically format and output a human-readable clean string snapshot over Serial
  char timeStringBuff[64];
  strftime(timeStringBuff, sizeof(timeStringBuff), "%A, %B %d %Y %H:%M:%S", &timeinfo);
  Serial.printf("[CLOCK] %s\n", timeStringBuff);
}
