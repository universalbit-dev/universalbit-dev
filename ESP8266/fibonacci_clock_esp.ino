// Fibonacci Clock for ESP8266/ESP32 with NTP Time Sync
// Integration of https://github.com/pchretien/fibo/blob/master/fibonacci.ino
// and ESP8266 NTP time via WiFi
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License Version 2

#include <ESP8266WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_NeoPixel.h>

// WiFi credentials
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// NTP settings
const char* ntpServerName = "pool.ntp.org";
const int timeZone = 0;     // UTC offset (hours)

WiFiUDP udp;
unsigned int localPort = 2390;      // local port to listen for UDP packets

#define NTP_PACKET_SIZE 48
byte packetBuffer[NTP_PACKET_SIZE];

// Hardware pins (adapt for ESP8266/ESP32)
#define STRIP_PIN D8
#define HOUR_PIN D3
#define MINUTE_PIN D4
#define BTN_PIN D5
#define SET_PIN D6
#define DEBOUNCE_DELAY 10
#define MAX_BUTTONS_INPUT 20
#define MAX_MODES 4
#define MAX_PALETTES 10
#define CLOCK_PIXELS 5

Adafruit_NeoPixel strip = Adafruit_NeoPixel(9, STRIP_PIN, NEO_RGB + NEO_KHZ800);

byte bits[CLOCK_PIXELS];
uint32_t black = strip.Color(0,0,0);
uint32_t colors[MAX_PALETTES][4] = {
  { strip.Color(255,255,255), strip.Color(255,10,10), strip.Color(10,255,10), strip.Color(10,10,255) },
  { strip.Color(255,255,255), strip.Color(255,10,10), strip.Color(248,222,0), strip.Color(10,10,255) },
  { strip.Color(255,255,255), strip.Color(80,40,0), strip.Color(20,200,20), strip.Color(255,100,10) },
  { strip.Color(255,255,255), strip.Color(245,100,201), strip.Color(114,247,54), strip.Color(113,235,219)},
  { strip.Color(255,255,255), strip.Color(255,123,123), strip.Color(143,255,112), strip.Color(120,120,255)},
  { strip.Color(255,255,255), strip.Color(212,49,45), strip.Color(145,210,49), strip.Color(141,95,224)},
  { strip.Color(255,255,255), strip.Color(209,62,200), strip.Color(69,232,224), strip.Color(80,70,202)},
  { strip.Color(255,255,255), strip.Color(237,20,20), strip.Color(246,243,54), strip.Color(255,126,21)},
  { strip.Color(255,255,255), strip.Color(70,35,0), strip.Color(70,122,10), strip.Color(200,182,0)},
  { strip.Color(255,255,255), strip.Color(211,34,34), strip.Color(80,151,78), strip.Color(16,24,149)}
};

boolean on = true;
byte oldHours = 0;
byte oldMinutes = 0;
int lastButtonValue[MAX_BUTTONS_INPUT];
int currentButtonValue[MAX_BUTTONS_INPUT];
int mode = 0;
int palette = 0;
byte error = 0;
byte oldError = 0;

// NTP time caching
unsigned long lastNtpSync = 0;
unsigned long cachedEpoch = 0;
const unsigned long ntpSyncInterval = 60 * 1000; // 1 min

void setupWiFi() {
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void sendNTPpacket(const char* address) {
  memset(packetBuffer, 0, NTP_PACKET_SIZE);
  packetBuffer[0] = 0b11100011;
  packetBuffer[1] = 0;
  packetBuffer[2] = 6;
  packetBuffer[3] = 0xEC;
  udp.beginPacket(address, 123);
  udp.write(packetBuffer, NTP_PACKET_SIZE);
  udp.endPacket();
}

unsigned long getNtpTime() {
  if (millis() - lastNtpSync > ntpSyncInterval || cachedEpoch == 0) {
    sendNTPpacket(ntpServerName);
    delay(1000);
    int cb = udp.parsePacket();
    if (!cb) {
      Serial.println("No NTP response");
      return cachedEpoch; // fallback to previous
    }
    udp.read(packetBuffer, NTP_PACKET_SIZE);
    unsigned long highWord = word(packetBuffer[40], packetBuffer[41]);
    unsigned long lowWord = word(packetBuffer[42], packetBuffer[43]);
    unsigned long secsSince1900 = highWord << 16 | lowWord;
    const unsigned long seventyYears = 2208988800UL;
    cachedEpoch = secsSince1900 - seventyYears + timeZone * 3600;
    lastNtpSync = millis();
  }
  return cachedEpoch + ((millis() - lastNtpSync) / 1000);
}

struct TimeHM {
  byte hour;
  byte minute;
};

TimeHM getCurrentTime() {
  unsigned long epoch = getNtpTime();
  byte hour = (epoch % 86400L) / 3600;
  byte minute = (epoch % 3600) / 60;
  return {hour, minute};
}

void setup() {
  Serial.begin(115200);
  setupWiFi();
  udp.begin(localPort);

  strip.begin();
  strip.show();

  pinMode(HOUR_PIN, INPUT);
  pinMode(MINUTE_PIN, INPUT);
  pinMode(BTN_PIN, INPUT);
  pinMode(SET_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  for(int i=0;i<4;i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
  }

  oldHours = 99;
}

void loop() {
  int set_button = debounce(SET_PIN);
  int hour_button = debounce(HOUR_PIN);
  int minute_button = debounce(MINUTE_PIN);
  int button = debounce(BTN_PIN);

  if(set_button && button && hasChanged(BTN_PIN)) {
    for(int i=0; i<100; i++)
      if(!debounce(SET_PIN) || !debounce(BTN_PIN)) break;
    if(debounce(SET_PIN) && debounce(BTN_PIN)) checkErrors();
  }
  else if(set_button && hour_button && hasChanged(HOUR_PIN)) {
    palette = (palette+1)%MAX_PALETTES;
    oldHours = 99;
    oldError = 99;
  }
  else if(set_button && minute_button && hasChanged(MINUTE_PIN)) {
    cachedEpoch = 0; // manual NTP sync
    displayCurrentTime();
  }
  else if(minute_button && hasChanged(MINUTE_PIN)) {
    toggleOnOff();
  }
  else if(hour_button && hasChanged(HOUR_PIN)) {
    palette = (palette+1)%MAX_PALETTES;
    oldHours = 99;
    oldError = 99;
  }
  else if(button && hasChanged(BTN_PIN)) {
    mode = mode + 1;
    if(mode >= MAX_MODES) mode = 0;
  }

  resetButtonValues();
  switch(mode) {
    case 0:  displayCurrentTime(); break;
    case 1:  oldHours = 99; rainbowCycle(20); break;
    case 2:  oldHours = 99; rainbow(20); break;
    case 3:  oldHours = 99; displayErrorCode(); break;
  }
}

int debounce(int pin) {
  int val = digitalRead(pin);
  if(val == lastButtonValue[pin]) {
    currentButtonValue[pin] = val;
    return val;
  }
  delay(DEBOUNCE_DELAY);
  val = digitalRead(pin);
  if(val != lastButtonValue[pin]) {
    currentButtonValue[pin] = val;
    return val;
  }
  currentButtonValue[pin] = lastButtonValue[pin];
  return lastButtonValue[pin];
}

boolean hasChanged(int pin) { return lastButtonValue[pin] != currentButtonValue[pin]; }

void resetButtonValues() {
  for(int i=0; i<MAX_BUTTONS_INPUT; i++)
    lastButtonValue[i] = currentButtonValue[i];
}

void displayCurrentTime() {
  TimeHM now = getCurrentTime();
  setTime(now.hour % 12, now.minute);
}

void setTime(byte hours, byte minutes) {
  if(oldHours == hours && oldMinutes/5 == minutes/5) return;
  oldHours = hours;
  oldMinutes = minutes;
  for(int i=0; i<CLOCK_PIXELS; i++) bits[i] = 0;
  setBits(hours, 0x01);
  setBits(minutes/5, 0x02);

  for(int i=0; i<CLOCK_PIXELS; i++) {
    setPixel(i, colors[palette][bits[i]]);
    strip.show();
  }
}

void setBits(byte value, byte offset) {
  switch(value) {
    case 1: switch(random(2)) { case 0: bits[0]|=offset; break; case 1: bits[1]|=offset; break; } break;
    case 2: switch(random(2)) { case 0: bits[2]|=offset; break; case 1: bits[0]|=offset; bits[1]|=offset; break; } break;
    case 3: switch(random(3)) { case 0: bits[3]|=offset; break; case 1: bits[0]|=offset; bits[2]|=offset; break; case 2: bits[1]|=offset; bits[2]|=offset; break; } break;
    case 4: switch(random(3)) { case 0: bits[0]|=offset; bits[3]|=offset; break; case 1: bits[1]|=offset; bits[3]|=offset; break; case 2: bits[0]|=offset; bits[1]|=offset; bits[2]|=offset; break; } break;
    case 5: switch(random(3)) { case 0: bits[4]|=offset; break; case 1: bits[2]|=offset; bits[3]|=offset; break; case 2: bits[0]|=offset; bits[1]|=offset; bits[3]|=offset; break; } break;
    case 6: switch(random(4)) { case 0: bits[0]|=offset; bits[4]|=offset; break; case 1: bits[1]|=offset; bits[4]|=offset; break; case 2: bits[0]|=offset; bits[2]|=offset; bits[3]|=offset; break; case 3: bits[1]|=offset; bits[2]|=offset; bits[3]|=offset; break; } break;
    case 7: switch(random(3)) { case 0: bits[2]|=offset; bits[4]|=offset; break; case 1: bits[0]|=offset; bits[1]|=offset; bits[4]|=offset; break; case 2: bits[0]|=offset; bits[1]|=offset; bits[2]|=offset; bits[3]|=offset; break; } break;
    case 8: switch(random(3)) { case 0: bits[3]|=offset; bits[4]|=offset; break; case 1: bits[0]|=offset; bits[2]|=offset; bits[4]|=offset; break; case 2: bits[1]|=offset; bits[2]|=offset; bits[4]|=offset; break; } break;
    case 9: switch(random(2)) { case 0: bits[0]|=offset; bits[3]|=offset; bits[4]|=offset; break; case 1: bits[1]|=offset; bits[3]|=offset; bits[4]|=offset; break; } break;
    case 10: switch(random(2)) { case 0: bits[2]|=offset; bits[3]|=offset; bits[4]|=offset; break; case 1: bits[0]|=offset; bits[1]|=offset; bits[3]|=offset; bits[4]|=offset; break; } break;
    case 11: switch(random(2)) { case 0: bits[0]|=offset; bits[2]|=offset; bits[3]|=offset; bits[4]|=offset; break; case 1: bits[1]|=offset; bits[2]|=offset; bits[3]|=offset; bits[4]|=offset; break; } break;
    case 12: bits[0]|=offset; bits[1]|=offset; bits[2]|=offset; bits[3]|=offset; bits[4]|=offset; break;
    default: break;
  }
}

void setPixel(byte pixel, uint32_t color) {
  if(!on) return;
  switch(pixel) {
    case 0: strip.setPixelColor(0, color); break;
    case 1: strip.setPixelColor(1, color); break;
    case 2: strip.setPixelColor(2, color); break;
    case 3: strip.setPixelColor(3, color); strip.setPixelColor(4, color); break;
    case 4: strip.setPixelColor(5, color); strip.setPixelColor(6, color); strip.setPixelColor(7, color); strip.setPixelColor(8, color); break;
  }
}

void rainbow(uint8_t wait) {
  uint16_t i, j;
  for(j=0; j<256; j++) {
    for(i=0; i< CLOCK_PIXELS; i++) setPixel(i, Wheel((i+j) & 255));
    strip.show();
    delay(wait);
    if(debounce(BTN_PIN) && hasChanged(BTN_PIN)) { mode = (mode+1)%MAX_MODES; resetButtonValues(); return; }
    if(debounce(MINUTE_PIN) && hasChanged(MINUTE_PIN)) { toggleOnOff(); resetButtonValues(); return; }
    resetButtonValues();
  }
}

void rainbowCycle(uint8_t wait) {
  uint16_t i, j;
  for(j=0; j<256*5; j++) {
    for(i=0; i< CLOCK_PIXELS; i++) setPixel(i, Wheel(((i * 256 / CLOCK_PIXELS) + j) & 255));
    strip.show();
    delay(wait);
    if(debounce(BTN_PIN) && hasChanged(BTN_PIN)) { mode = (mode+1)%MAX_MODES; resetButtonValues(); return; }
    if(debounce(MINUTE_PIN) && hasChanged(MINUTE_PIN)) { toggleOnOff(); resetButtonValues(); return; }
    resetButtonValues();
  }
}

uint32_t Wheel(byte WheelPos) {
  if(WheelPos < 85) return strip.Color(WheelPos * 3, 255 - WheelPos * 3, 0);
  else if(WheelPos < 170) { WheelPos -= 85; return strip.Color(255 - WheelPos * 3, 0, WheelPos * 3); }
  else { WheelPos -= 170; return strip.Color(0, WheelPos * 3, 255 - WheelPos * 3); }
}

void toggleOnOff() {
  if(on) {
    for(int i=0; i<CLOCK_PIXELS; i++) setPixel(i, black);
    strip.show();
  }
  on = !on;
  if(on) { oldHours = 99; oldError = 99; }
}

void checkErrors() {
  error = 0;
  oldError = 99;
  mode = 3;
  palette = 0;
}

void displayErrorCode() {
  if(oldError == error) return;
  oldError = error;
  for(int i=0; i<CLOCK_PIXELS; i++) bits[i] = 0;
  if(error == 0) setBits(12, 0x02);
  else setBits(error, 0x01);

  for(int i=0; i<CLOCK_PIXELS; i++) setPixel(i, colors[palette][bits[i]]);
  strip.show();
}
