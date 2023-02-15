/*
 UniversalBit ... once again
 about some copy and paste code
 European NTP Server 
 0.europe.pool.ntp.org | 1.europe.pool.ntp.org | 2.europe.pool.ntp.org | pool 3.europe.pool.ntp.org

 https://microcontrollerslab.com/current-date-time-esp8266-nodemcu-ntp-server/
 Board:Generic ESP8266 Module
 Connecting....
 *********************************
 Chip is ESP8266EX
 Features: WiFi
 Writing at 0x00030000... (100 %)
 *********************************
*/

#include <ESP8266WiFi.h>
#include "time.h"

const char* ssid       = "Guest Wifi-Name";   //Replace with your guest wifi name
const char* password   = "Guest Password";  //Replace with your guest wifi password
const char* ntpServer = "0.europe.pool.ntp.org";
//How many seconds in 1 hour
const long  gmtOffset_sec = 3600;  //Replace with your GMT offset (seconds)
//
const int   daylightOffset_sec = 0;  //Replace with your daylight offset (seconds)

void setup()
{
  Serial.begin(9600);  
  //connect to WiFi
  Serial.printf("Connecting to %s ", ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
      delay(500);
      Serial.print(".");
  }
  Serial.println("CONNECTED to WIFI");
  
  //init and get the time
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
  printLocalTime();

  //disconnect WiFi as it's no longer needed
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
}

void loop()
{
  delay(1000);
  printLocalTime();
}
void printLocalTime()
{
  time_t rawtime;
  struct tm * timeinfo;
  time (&rawtime);
  timeinfo = localtime (&rawtime);
  Serial.println(asctime(timeinfo));
  delay(1000);
}
