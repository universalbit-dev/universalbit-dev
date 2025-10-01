# ESP8266 Fibonacci Clock with NTP Synchronization

---

## Overview

Upgrade your Fibonacci Clock with an ESP8266 microcontroller for WiFi connectivity and automatic time synchronization using the Network Time Protocol (NTP).  
No RTC module required—accurate time is maintained as long as the clock is connected to WiFi.

---

## Features

- **WiFi-Enabled:** Seamless local network connection.
- **Accurate Time:** Periodically fetches time from NTP servers.
- **Unique Display:** Visualizes time with color-coded NeoPixels in a Fibonacci layout.
- **Color Palettes:** Choose from multiple color themes.
- **Interactive Controls:** Change palettes, modes, and toggle display with buttons.

---

## Hardware Requirements

- **ESP8266 Board:** (D1 Mini or NodeMCU recommended)
- **NeoPixel Strip:** At least 9 WS2812 (or compatible) LEDs
- **Push Buttons:** For mode, palette, and display control
- **USB Data Cable:** For programming and power

**Wiring:**
- NeoPixel data pin → ESP8266 D8 (customizable in code)
- Buttons → ESP8266 D3, D4, D5, D6 (customizable in code)

---

## Software Setup

1. **Clone the Repository**
    ```bash
    git clone https://github.com/universalbit-dev/universalbit-dev.git
    cd universalbit-dev/ESP8266
    ```

2. **Install Arduino IDE & ESP8266 Support**
    - [Arduino IDE](https://www.arduino.cc/en/software)
    - [ESP8266 Board Setup Guide](https://arduino-esp8266.readthedocs.io/en/latest/installing.html)

3. **Install Required Libraries**
    - `ESP8266WiFi`
    - `WiFiUdp`
    - `Adafruit_NeoPixel`
    
    Use Arduino Library Manager to install any missing libraries.

4. **Configure WiFi**
    - Edit `fibonacci_clock_esp8266.ino`:
      ```cpp
      const char* ssid     = "YOUR_WIFI_SSID";
      const char* password = "YOUR_WIFI_PASSWORD";
      ```

5. **Upload the Sketch**
    - Open `fibonacci_clock_esp8266.ino` in Arduino IDE.
    - Select your ESP8266 board and correct port.
    - Upload using a USB data cable.

---

## Usage

- **Modes:**
    - **Time Display:** Shows current time (default).
    - **Rainbow Animation:** Cycles through colors.
    - **Error Display:** Indicates WiFi/NTP errors.

- **Button Controls:**
    - **Palette Button:** Cycles through color palettes.
    - **Mode Button:** Switches display modes.
    - **Minute/Hour Buttons:** Toggles display or forces NTP sync.

- **Time Source:**  
    - Time is fetched from NTP servers at startup and every minute.
    - If WiFi/NTP fails, the clock falls back to 00:00.

---

## Troubleshooting

- **WiFi Issues:** Verify SSID/password; use Serial Monitor for debug output.
- **No Lights:** Check NeoPixel wiring and correct pin assignment.
- **Wrong Time:** Double-check NTP server and time zone offset.
- **Unresponsive Buttons:** Confirm hardware connections and pin settings.

---

## References

- [ESP8266 NTP Example](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/esp8266_ntp.ino)
- [Adafruit NeoPixel Guide](https://learn.adafruit.com/adafruit-neopixel-uberguide)
- [Network Time Protocol (Wikipedia)](https://en.wikipedia.org/wiki/Network_Time_Protocol)

---

## Contributing

Open issues or submit pull requests to improve features or fix bugs!

---

## License

Released under GNU General Public License v2.  
See LICENSE for details.
