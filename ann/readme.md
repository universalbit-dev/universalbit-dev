##### [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support) -- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation) -- [Bash Reference Manual](https://www.gnu.org/software/bash/manual/html_node/index.html) -- [Join Mastodon](https://mastodon.social/invite/wTHp2hSD) -- [Website](https://www.universalbit.it/) -- [Content Delivery Network](https://www.universalbitcdn.it/)

-- copilot explain

The ArduinoANN.ino file implements an artificial neural network (ANN) for the Arduino platform. Here's a breakdown of its structure and functionality:

1. **Header and Includes**:
   - The file begins with a comment header describing the project.
   - It includes the `math.h` library for mathematical functions.

2. **Network Configuration**:
   - Constants define the configuration of the neural network, including the number of patterns, input nodes, hidden nodes, output nodes, learning rate, momentum, initial weight maximum, and success threshold.
   - Arrays `Input` and `Target` are defined to hold input patterns and their corresponding target outputs.

3. **Variables**:
   - Various variables are declared for indices, training cycles, errors, accumulators, and weights.
   - Arrays for hidden layer activations, output activations, weight changes, and deltas are also declared.

4. **Setup Function**:
   - Initializes the serial communication.
   - Seeds the random number generator.
   - Prepares the randomized indices for training patterns.

5. **Loop Function**:
   - **Initialization**:
     - Initializes hidden and output weights with random values.
     - Initializes the weight change arrays to zero.
   - **Training**:
     - The main training loop runs for a large number of cycles.
     - Randomizes the order of training patterns.
     - For each pattern, computes hidden and output layer activations.
     - Calculates output errors and backpropagates these errors to adjust weights.
     - Every 1000 cycles, it reports the current state to the terminal.
     - Terminates training if the error rate falls below the success threshold.

6. **toTerminal Function**:
   - Outputs the current state of the network to the terminal, including input patterns, target outputs, and the computed outputs.

This code essentially trains a simple feedforward neural network using backpropagation on predefined patterns, adjusting weights to minimize the error between actual and target outputs.

---

* clone project:
```bash
git clone https://github.com/universalbit-dev/universalbit-dev.git
cd universalbit-dev/ann/

```

* ##### [Arduino Nano](https://en.wikipedia.org/wiki/Arduino_Nano) - An artificial neural network for the Arduino --
<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/gif/arduino-nano_ANN.gif" width="auto"></img>

| Wire Four Arduino Nano                           |                             |
| ----------------------------------- | ----------------------------------- |
| ![arduino_ann](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/arduino_ann.JPG) | ![arduino_ann_02](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/arduino_ann_02.JPG) |


<img src="https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/gif/esp32_ANN.gif" width="auto"></img>

* [Arduino Nano -- wiki --](https://en.wikipedia.org/wiki/Arduino_Nano) || [Arduino Uno -- wiki --](https://en.wikipedia.org/wiki/Arduino_Uno)
  
---

---

#### Be careful to use only a USB data cable,some usb cables are only used to recharge and do not allow the upload of code.

## Wire ESP8266(Wifi) and Two Arduino Nano
* [ESP8266](https://en.wikipedia.org/wiki/ESP8266)
* [ESP32](https://en.wikipedia.org/wiki/ESP32)

---

###### D1 Mini ESP8266     - [Logic Converter 3.3 | 5V](https://forum.arduino.cc/t/logic-level-converter/1136803/9)   - Two Arduino Nano

[![D1_Mini_Arduino](https://github.com/universalbit-dev/universalbit-dev/blob/main/ann/img/D1_Mini_ArduinoNano_Logic_Converter.png)](https://github.com/universalbit-dev/universalbit-dev/tree/main/ann)

---

[Upload a sketch](https://support.arduino.cc/hc/en-us/articles/4733418441116-Upload-a-sketch-in-Arduino-IDE)

### Resources:
* [Arduino Projects](https://randomnerdtutorials.com/projects-esp32/)
* [MicroPython on ESP32](https://randomnerdtutorials.com/getting-started-micropython-esp32-esp8266/)
---
