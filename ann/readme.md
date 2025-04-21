# ArduinoANN: An Artificial Neural Network for Arduino

This project implements an Artificial Neural Network (ANN) for the Arduino platform. It demonstrates how neural networks can be used to learn and predict based on predefined input and output patterns.

---

## Project Overview

### Key Features
- **Platform**: Arduino Nano, Arduino Uno, ESP8266, and ESP32.
- **Neural Network**: A feedforward ANN trained using backpropagation.
- **Inputs and Outputs**:
  - The input patterns are based on the Fibonacci sequence.
  - Target outputs are hypothetical binary data.

For more details on the ANN implementation, refer to the source file: `ArduinoANN.ino`.

---

## Clone the Project

To get started, clone the repository:

```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
cd universalbit-dev/ann/
```

---

## Hardware Requirements

### **Arduino Nano**
An artificial neural network implemented on the Arduino Nano.

![Arduino Nano ANN](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/gif/arduino-nano_ANN.gif)

#### Wiring Multiple Arduino Nano Boards
| Four Arduino Nano Boards                       | Example Setup                              |
| ---------------------------------------------- | ------------------------------------------ |
| ![arduino_ann](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/arduino_ann.JPG) | ![arduino_ann_02](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/arduino_ann_02.JPG) |

---

### **ESP8266 and ESP32**
The project also supports ESP8266 and ESP32 microcontrollers.

![ESP32 ANN](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/gif/esp32_ANN.gif)

#### Wiring ESP8266 (WiFi) and Two Arduino Nano Boards
- [ESP8266](https://en.wikipedia.org/wiki/ESP8266)
- [ESP32](https://en.wikipedia.org/wiki/ESP32)

![D1 Mini + Arduino Nano](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/D1_Mini_ArduinoNano_Logic_Converter.png)

*Note*: Use a [Logic Converter (3.3V ↔ 5V)](https://forum.arduino.cc/t/logic-level-converter/1136803/9) for safe communication between ESP8266 and Arduino Nano.

---

## Important Notes

- **USB Data Cable**: Be cautious to use only a USB data cable. Some USB cables are designed for charging only and will not allow code uploads.
- **Upload Sketch**: Follow [this guide](https://support.arduino.cc/hc/en-us/articles/4733418441116-Upload-a-sketch-in-Arduino-IDE) to upload sketches to your Arduino.

---

## Resources

Here are some additional resources to help you get started:

- [Arduino Nano](https://en.wikipedia.org/wiki/Arduino_Nano) Wiki
- [Arduino Uno](https://en.wikipedia.org/wiki/Arduino_Uno) Wiki
- [Arduino Projects](https://randomnerdtutorials.com/projects-esp32/)
- [MicroPython on ESP32](https://randomnerdtutorials.com/getting-started-micropython-esp32-esp8266/)

---

## About the Code

The `ArduinoANN.ino` file implements the following:

1. **Header and Includes**:
   - `math.h` library is used for mathematical operations.

2. **Network Configuration**:
   - Configurations such as the number of nodes, learning rate, and momentum are predefined.
   - Input patterns (Fibonacci sequence) and target outputs are stored in arrays.

3. **Setup Function**:
   - Initializes serial communication.
   - Seeds the random number generator for weight initialization.

4. **Main Loop**:
   - Trains the neural network using backpropagation.
   - Adjusts weights to minimize errors.
   - Monitors progress every 1000 training cycles.

5. **toTerminal Function**:
   - Displays the state of the neural network, including input patterns, target outputs, and computed outputs.

---
