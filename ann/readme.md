# Arduino Light AI: On-Device ANN & Reinforcement Learning

This repository implements lightweight Machine Learning ("Light AI" or TinyML) algorithms running natively and entirely offline on highly constrained microcontrollers. It showcases how both neural networks and reinforcement learning can execute directly on edge silicon.

---

## 📂 Repository Structure

Your workspace contains the following files:
*   **`ArduinoANN.ino`**: The feedforward Artificial Neural Network classifier.
*   **`ArduinoRL.ino`**: The autonomous tabular Q-learning reinforcement learning agent.
*   **`arduino_ai_uploader.sh`**: The automated multi-board compiler and firmware flasher.
---

## 🧠 The Light AI Models

### 1. Artificial Neural Network (`ArduinoANN.ino`)
*   **Paradigm**: Supervised Feedforward Artificial Neural Network (ANN) trained on-chip via backpropagation.
*   **Task**: 7-segment display digit recognition (mapping physical segments to digits 0–9).
*   **Architecture**: 7 input nodes, 8 hidden nodes, and 4 binary output nodes.
*   **Mathematical Model**: Calculates activations using the sigmoid function and performs localized weight updates with adjustable Learning Rate and Momentum.

### 2. Reinforcement Learning Agent (`ArduinoRL.ino`)
*   **Paradigm**: Tabular Q-learning.
*   **Task**: Autonomous pathfinding in a 1D grid world of 6 states (0 to 5).
*   **Algorithm**: Utilizes an epsilon-greedy strategy to balance exploration and exploitation, dynamically updating a localized State-Action value matrix (Q-Table) via the Bellman optimality equation.
*   **Goal**: Reach State 5 (Goal State) while dynamically minimizing steps taken per episode.

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

1. Dynamically scans and lists the `.ino` sketches in your folder so you can choose which one to build.
2. Installs a localized, sandboxed instance of `arduino-cli` inside your home directory.
3. Scans active USB lines to auto-detect your active port (e.g., `/dev/ttyUSB1`).
4. Probes the hardware signature to check if it's an AVR (Nano/Uno), ESP8266, or ESP32 chip.
5. Isolates the build environment in `/tmp` to prevent compilation/multiple-definition bugs.

---

## 🔌 Hardware Setup & Requirements

The firmware is fully compatible with standard AVR (Nano/Uno) and 32-bit Expressif (ESP8266/ESP32) boards.

### **Arduino Nano**

Running on-device backpropagation and Q-value iteration inside only 2 KB of SRAM.

---

### **ESP8266 and ESP32**
Leveraging higher CPU clock speeds and larger flash/RAM capacities.
---

## ⚡ Important Operational Notes

* **USB Data Cable**: Be sure to use a dedicated **USB Data Cable**. Standard charging-only cables omit internal data lines and cannot compile/upload sketches or output serial logs.
* **Old Bootloader Fallback**: Many common Arduino Nano clone boards use the Atmega328P Old Bootloader. If your upload times out, simply select **Option 2** (Arduino Nano Old Bootloader) in the command-line script.

---

## ⚖️ License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/LICENSE) file for the full terms and conditions.

```
