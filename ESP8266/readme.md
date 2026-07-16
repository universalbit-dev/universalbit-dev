![Tested on Wemos D1 Mini](https://img.shields.io/badge/Tested%20on-Wemos%20D1%20Mini-00979D?style=flat-color&logo=expressif&logoColor=white)
![Tested on WeMos D1 R3](https://img.shields.io/badge/Tested%20on-WeMos%20D1%20R3%20(ESP8266)-orange?style=flat-color&logo=espressif&logoColor=white)
![Firmware Status](https://img.shields.io/badge/Firmware-Verified%20%26%20Stable-success?style=flat-color)
# ESP8266 NTP Server Setup Guide

Welcome to the ESP8266 NTP Server project! This guide will help you set up and use the ESP8266 module to synchronize time using the Network Time Protocol (NTP).

---

## 📚 **Support and References**
- [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)
- [Disambiguation (Wikipedia)](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)
- [Bash References](https://www.gnu.org/software/bash/manual/bash.html)

---

## 🛠 **Getting Started**

### Clone the Repository
Follow these steps to clone the project and navigate to the ESP8266 directory:
```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
cd universalbit-dev/ESP8266
```

---

## 🌐 **What is ESP8266?**
The [ESP8266](https://en.wikipedia.org/wiki/ESP8266) is a low-cost Wi-Fi microchip with full TCP/IP stack and microcontroller capability, ideal for IoT projects.

---

## ⏰ **ESP8266 NTP Server**
Synchronize your device's time using the [Network Time Protocol (NTP)](https://microcontrollerslab.com/current-date-time-esp8266-nodemcu-ntp-server/). This project provides a ready-to-use sketch for creating your own NTP server.

---

## ⚠️ **Important Note**
When uploading sketches to your ESP8266, use a **USB data cable**. Some cables are designed only for charging and will not allow data transfer.

---

## 🔧 **Hardware Requirements**

### [D1 Mini](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/images/D1_Mini_ESP8266.png)
 ESP8266
The D1 Mini ESP8266 is a compact, Wi-Fi-enabled microcontroller perfect for this project.

---
## **[ESP8266 Fibonacci Clock with NTP Synchronization](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/ESP8266%20Fibonacci%20Clock.md)**

## 📜 **Project Files**
- **Sketch**:
- [esp8266_ntp.ino](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/esp8266_ntp.ino)
- [fibonacci_clock_esp.ino](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/fibonacci_clock_esp.ino)

- **⚡ How to Upload the Firmware**: 
  Instead of setting up the bulky Arduino IDE, you can compile and flash the firmware directly from your terminal using our automated, interactive deployment script. This tool handles compiler installation, hardware auto-detection, and flashing under the hood[cite: 3]:

  ```bash
  # 1. Make the script executable
  chmod +x esp8266_ntp.sh

  # 2. Run the interactive menu
  ./esp8266_ntp.sh
  ```

---

## 🚀 Explore More ESP32 & ESP8266 Projects

Looking for more inspiration? Check out these curated project collections from Random Nerd Tutorials:

- [ESP8266 Project](https://randomnerdtutorials.com/projects-esp8266/)

Discover creative ideas, detailed tutorials, and inspiring builds for your next Internet of Things project!

--- 

