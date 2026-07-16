# 🌀 Fibonacci NTP Clock (`fibonacci_ntp`)

[![status](https://img.shields.io/badge/status-concept_/_untested-orange?style=flat-square)](https://github.com/universalbit-dev/universalbit-dev)

A specialized time-tracking engine that maps Network Time Protocol (NTP) data onto a geometric grid using the Fibonacci sequence. Powered by the ESP8266EX microcontroller, this project replaces traditional physical clocks with a real-time, network-synced cryptic visual puzzle.

---

## 🧭 How It Works: The Fibonacci Matrix

The clock displays time using a golden rectangle layout divided into 5 squares whose side lengths match the first 5 terms of the Fibonacci sequence: **1, 1, 2, 3, and 5**. 

### 1. The Color Encoding Schema
Instead of hands or digits, the current time is translated into an array of color configurations rendered digitally on screen:

| Cell Color | Chronological Interpretation | Operational Matrix Action |
| :--- | :--- | :--- |
| **🟥 Red** | Hour Value Component | Add cell weight to Hours total |
| **🟩 Green** | Minute Value Component | Add cell weight to Minutes total |
| **🟦 Blue** | Dual-Value Intersection | Add cell weight to **Both** Hours & Minutes |
| **⬜ White** | Null Spacer Value | Ignored entirely during value calculations |

### 2. Computing the Time Arrays
* **Hours ($1 \rightarrow 12$):** Sum the side-lengths of all **Red** and **Blue** squares.
* **Minutes ($0 \rightarrow 55$):** Sum the side-lengths of all **Green** and **Blue** squares, then **multiply the result by 5**. 
* **Precision Rounding:** Standard layout algorithms round standard system clock times down to the nearest lower multiple of 5 minutes (e.g., `10:34` becomes the Fibonacci equivalent of `10:30`).

$$\text{Final Minute Value} = \text{Calculated Minute Sum} \times 5$$

---

## 🔌 Hardware Circuit Architecture

The device transitions away from legacy, drift-heavy hardware Real-Time Clock modules (like the DS1307) by leveraging the ESP8266's integrated 802.11b/g/n Wi-Fi stack to sync directly with atomic clock stratums over the internet.

Traditional DIY Fibonacci clocks require wiring complex physical LED/wooden matrix boxes via massive copper grids or shift registers. By switching to a high-speed native SPI bus, hardware complexity drops to just a few data lines. The ESP8266 streams mathematical coordinates directly over SPI, letting the TFT controller instantly draw the geometric color grid on the glass.


```text
                +-----------------------------------+
                |             ESP8266               |
                |        (NodeMCU / D1 Mini)        |
                |                                   |
                |   [Wi-Fi] <---> Internet (NTP)    |
                |                                   |
                |       SPI Hardware Bus Interface  |
                +-----------------------------------+
                          |   |   |   |   |
                         SCK MOSI CS  DC RST
                          |   |   |   |   |
                +-----------------------------------+
                |           LOLIN TFT-2.4           |
                |        (ILI9341 Color Glass)      |
                |                                   |
                |   🔴 🟩 🟦 Dynamic Geometric Grid |
                +-----------------------------------+

```

### Pinout Mapping Configuration

| LOLIN TFT-2.4 Pin | D1 Mini (ESP8266) Pin | Logical GPIO Assignment | Purpose |
| :--- | :--- | :--- | :--- |
| **3V** | **3V3** | 3.3V Power Rail | Screen Logic Power |
| **LED** | **3V3** | Backlight Power | Constant Backlight Activation |
| **GND** | **G** | Ground | Common System Ground |
| **SCK** | **D5** | `GPIO14` | Native Hardware SPI Clock |
| **SI** (MOSI) | **D7** | `GPIO13` | Native Hardware SPI Data Input |
| **SO** (MISO) | **D6** | `GPIO12` | SPI Data Output (Optional) |
| **CS** | **D8** | `GPIO15` | Hardware Chip Select Line |
| **DC** (Data/Cmd)| **D3** | `GPIO0` | Display Command Bus Toggle |
| **RST** (Reset) | **RST** | Physical Reset Link | Shared Hardware System Reset |

---

## 🧠 The NTP-to-Sequence Translation Engine

The system completely bypasses the need for an external physical Real-Time Clock (RTC) chip. The execution cycle operates using a lightweight, automated parsing loop:

```text
[Atomic NTP Server] 📡 Network Time Packets
         ↓ (Wi-Fi Background Sync)
    [ESP8266 MCU]    💻 Formats Unix Epoch to HH:MM / Runs Fibonacci Matrix Math
         ↓ (High-Speed SPI Bus)
  [LOLIN TFT-2.4]    🔴 🟢 🔵 Refreshes Cryptic Geometric Color Arrays

```

### Display Driver Code Hook

The accompanying sketch utilizes the performance-optimized `TFT_eSPI` or `Adafruit_ILI9341` library suites. To map this to your hardware platform, use the following initialization parameters:

```cpp
#define TFT_CS    D8  // Hardware Pin GPIO15
#define TFT_DC    D3  // Hardware Pin GPIO0
#define TFT_RST   -1  // Tied directly to MCU Reset to save GPIO lines

// Initialize the display engine driver
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

```

---

## ⚡ Firmware Optimization Layer

The accompanying `fibonacci_ntp.ino` production binary features several structural performance enhancements:

* 🟢 **Asynchronous Network Guards:** Replaced legacy infinite blocking loops during setup with a definitive timeout sequence to prevent hard hangs if an access point drops offline.
* 🟢 **Active Drift Suppression:** Retains a persistent background connection to local regional `pool.ntp.org` servers to automatically update the software RTC without animation stutter.
* 🟢 **Non-Blocking Logic:** Swapped out execution-pausing `delay()` statements within the primary loop for timestamp tracking via `millis()` state checks to preserve fluid visual rendering.
* 🟢 **High-Speed Serial Communication:** Standardized to a profile speed of `115200` baud to match native ESP8266 boot parameters and clear out data pipe parsing overhead.

---

## 🔗 Navigation Links

* 🏠 [Return to Main Project Root](https://github.com/universalbit-dev/universalbit-dev/tree/main/ESP8266)
* 💻 [View Core Firmware Sketch File](https://github.com/universalbit-dev/universalbit-dev/blob/main/ESP8266/fibonacci_ntp/fibonacci_ntp.ino)

```

```
