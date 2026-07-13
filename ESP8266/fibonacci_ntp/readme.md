[![status](https://img.shields.io/badge/status-concept_/_untested-orange?style=flat-square)](https://github.com/universalbit-dev/universalbit-dev)
# 📡 Fibonacci NTP Engine

This subsystem handles the core networking logic and matrix rendering for the futuristic Fibonacci Chronometer. By combining the Wi-Fi processing power of the **ESP8266** with a dedicated **LOLIN TFT-2.4** display, this project replaces physical clock components with a real-time network-synced cryptic visual puzzle.

---

## 🛠️ Hardware Integration Architecture

Instead of managing complex discrete wiring loops for physical panels, this build utilizes the high-speed native **SPI bus** of the Wemos D1 Mini (ESP8266) to digitally render the entire geometric layout directly onto the TFT display panel.

> 💡 **Why SPI Architecture?** Traditional DIY Fibonacci clocks require wiring complex physical LED/wooden matrix boxes via massive copper grids or shift registers. By switching to a high-speed native SPI bus, hardware complexity drops to just a few data lines. The ESP8266 streams mathematical coordinates directly over SPI, letting the TFT controller instantly draw the geometric color grid on the glass.

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
| **DC** (Data/Command)| **D3** | `GPIO0` | Display Command Bus Toggle |
| **RST** (Reset) | **RST** | Physical Reset Link | Shared Hardware System Reset |

> 📁 **Hardware Schematic References:**
> * [D1 Mini ESP8266 Pinout Schematic]()

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

### The Visual Core Calculation Logic

The clock face is drawn dynamically on the TFT screen as a 5-block grid scaling proportionally to the Fibonacci sequence: **1, 1, 2, 3, and 5**.

* 🔴 **Red Grid Units:** Contribute their sequence value directly to the **Hour Sum**.
* 🟢 **Green Grid Units:** Contribute their sequence value directly to the **Minute Sum**.
* 🔵 **Blue Grid Units:** Dual contribution — values are added to **both** the Hour and Minute totals.
* ⚪ **Unlit Units:** Ignored during the active calculation cycle.

$$\text{Final Minute Value} = \text{Calculated Minute Sum} \times 5$$

---

## 💻 Firmware Engine Setup

The accompanying sketch utilizes the performance-optimized `TFT_eSPI` or `Adafruit_ILI9341` library suites. To flash this to your ESP8266 platform, ensure your initialization header flags are set as follows:

```cpp
#define TFT_CS    D8  // Hardware Pin GPIO15
#define TFT_DC    D3  // Hardware Pin GPIO0
#define TFT_RST   -1  // Tied directly to MCU Reset to save GPIO lines

// Initialize the display engine driver
Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

```

### Navigation Links

* 🏠 [Return to Main Project Root]()
* 💻 [View Core Firmware Sketch File]()

```

```
