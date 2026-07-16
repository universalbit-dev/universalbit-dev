# Arduino Light AI: On-Device ANN & Reinforcement Learning

This repository implements lightweight Machine Learning ("Light AI" or TinyML) algorithms running natively and entirely offline on highly constrained microcontrollers. It showcases how neural networks, pattern classifiers, and reinforcement learning can execute directly on edge silicon without cloud dependencies.

---

## 🧠 The Light AI Models

### 1. 7-Segment Digit Recognizer (`ArduinoANN.ino`)

* **Paradigm**: Supervised Feedforward Artificial Neural Network (ANN) trained on-chip via backpropagation.


* **Task**: 7-segment display digit recognition (mapping physical segments to digits 0–9).


* **Architecture**: 7 input nodes, 8 hidden nodes, and 4 binary output nodes.


* **Mathematical Model**: Calculates activations using sigmoid functions and performs localized weight updates with adjustable Learning Rate and Momentum.



### 2. Fibonacci Sequence Classifier (`ArduinoANNFIB.ino`)

* **Paradigm**: Supervised Feedforward Neural Network using on-chip backpropagation.
* **Task**: Recognizes and classifies sliding-window patterns of the Fibonacci sequence.
* **Data Handling**: Uses 7 sequential input values (with automatic 8-bit `byte` wrap-around for values exceeding 255, such as 377 becoming `121` and 610 becoming `98`) mapped to 4-bit classifications.
* **Optimization**: Configured to converge under success thresholds using raw on-device matrix math.
  
### 3. Reinforcement Learning Agent (`ArduinoRL.ino`)

* **Paradigm**: Tabular Q-learning.
* **Task**: Autonomous pathfinding in a 1D grid world of 6 states (0 to 5).
* **Algorithm**: Utilizes an $\epsilon$-greedy strategy to balance exploration and exploitation, dynamically updating a localized State-Action value matrix (Q-Table) via the Bellman optimality equation.
---

## 🛠️ Clone & Quick Start

To get started, clone the repository and navigate into the project directory:

```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
cd universalbit-dev/ann/

```

### Deploying Firmware via CLI (No IDE Required)

You can compile and flash either model using the local sandbox script `arduino_ai_uploader.sh`.

```bash
# Make the deployment script executable
chmod +x arduino_ai_uploader.sh

# Run the automated uploader (Handles serial ports and chip auto-probing)
sudo ./arduino_ai_uploader.sh

```

#### What the Script Does:

1. **Dynamic Scanning**: Automatically scans your workspace directory and lists all `.ino` files (dynamically parsing `ArduinoANN.ino`, `ArduinoANNFIB.ino`, and `ArduinoRL.ino` into the menu selection).
2. **Local Sandboxing**: Installs a localized, sandboxed instance of `arduino-cli` inside your home directory so your system packages remain unmodified.
3. **Serial Discovery**: Scans active USB lines to auto-detect your active connection port (e.g., `/dev/ttyUSB1`).
4. **Hardware Probing**: Queries the hardware signature to determine if the connected board is an AVR (Nano/Uno), ESP8266, or ESP32 chip.
5. **Clean Builds**: Isolates compiler tasks inside a temporary `/tmp` environment to bypass multiple-definition or artifact conflicts.

---

## 🔌 Hardware Setup & Requirements

The firmware is fully compatible with standard AVR (Nano/Uno) and 32-bit Expressif (ESP8266/ESP32) boards.

### **Arduino Nano**

* Executes on-device backpropagation and tabular Q-value updates inside a restricted 2 KB of SRAM.



### **ESP8266 and ESP32**

* Leverages fast 32-bit CPU clocks and spacious flash and RAM partitions to complete training iterations exponentially faster.

---

## ⚡ Important Operational Notes

* **USB Data Cable**: Be sure to use a dedicated **USB Data Cable**. Standard charging-only cables omit internal data lines and cannot compile/upload sketches or output serial logs.


* **Old Bootloader Fallback**: Many common Arduino Nano clone boards use the Atmega328P Old Bootloader. If your upload times out, simply select **Option 2** (Arduino Nano Old Bootloader) in the command-line script menu.



---

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/LICENSE) file for the full terms and conditions.
