# Arduino Nano Neural Network Project

Welcome to the **Arduino Nano Neural Network Project**! This guide provides detailed instructions and information on setting up a neural network using an Arduino Nano and ENC28J60 Ethernet module.

---

## Table of Contents
1. [Support and References](#support-and-references)
2. [Project Overview](#project-overview)
3. [Required Hardware](#required-hardware)
4. [Code Explanation](#code-explanation)
5. [Additional Resources](#additional-resources)
6. [Stay Tuned](#stay-tuned)

---

## Support and References

Support the **UniversalBit Project** and explore related topics:
- [Support UniversalBit Project](https://github.com/universalbit-dev/universalbit-dev/tree/main/support)
- [Disambiguation](https://en.wikipedia.org/wiki/Wikipedia:Disambiguation)
- [Hacker Culture](https://en.wikipedia.org/wiki/Hacker_culture)

---

## Project Overview

This project demonstrates how to set up and train an artificial neural network using an Arduino Nano. The network is configured to process inputs, calculate outputs, and optimize weights using backpropagation.

Key features include:
- **MAC Address Configuration**: Customize the network using a unique MAC address.
- **Network Training**: Use training patterns to achieve a low error rate.
- **Real-time Monitoring**: Display network training progress via the serial terminal.

---

## Required Hardware

To build this project, you need the following:
1. **Arduino Nano**
2. **Mini ENC28J60 Ethernet Module**

![Arduino Nano and Ethernet Module](https://github.com/universalbit-dev/universalbit-dev/blob/main/hacking/images/DSCF2499.JPG)

---

## Code Explanation

### Overview
The provided Arduino script includes the following:
- **Network Configuration**: Set up IP addresses, MAC addresses, and server details.
- **Neural Network Architecture**:
  - Input Nodes: 7
  - Hidden Nodes: 8
  - Output Nodes: 4
- **Training Parameters**:
  - Learning Rate: 0.3
  - Momentum: 0.9
  - Success Threshold: 0.0004

### Example MAC Address
```c++
byte mac[] = { 0xDE, 0xAF, 0xCF, 0xEF, 0xFE, 0xBD };
```

### Training Data
Input and target patterns for training:
```c++
const byte Input[PatternCount][InputNodes] = {
  { 1, 1, 1, 1, 1, 1, 0 },  // 0
  { 0, 1, 1, 0, 0, 0, 0 },  // 1
  { 1, 1, 0, 1, 1, 0, 1 },  // 2
  ...
};

const byte Target[PatternCount][OutputNodes] = {
  { 0, 0, 0, 0 },
  { 0, 0, 0, 1 },
  { 0, 0, 1, 0 },
  ...
};
```

### Training Loop
The training loop randomizes input patterns, calculates errors, and updates weights using backpropagation:
```c++
for (TrainingCycle = 1; TrainingCycle < MAX_CYCLES; TrainingCycle++) {
  // Randomize input patterns
  // Compute activations
  // Backpropagate errors
  // Update weights
}
```

### Serial Monitoring
Monitor training progress and outputs via the serial terminal:
```c++
void toTerminal() {
  Serial.print("Training Pattern: ");
  Serial.print(p);
  Serial.print(" Output: ");
  for (i = 0; i < OutputNodes; i++) {
    Serial.print(Output[i], 5);
    Serial.print(" ");
  }
}
```

### Full Code
For the full implementation, refer to the script provided in this repository.

---

## Additional Resources

Explore related references:
1. [Ethernet Client Library](https://www.arduino.cc/reference/en/libraries/ethernet/ethernetclient/)
2. [Ethernet Reference Documentation](https://www.arduino.cc/reference/en/libraries/ethernet/)
3. [Arduino Neural Network Guide](https://robotics.hobbizine.com/arduinoann.html)

---

## Stay Tuned

Stay updated with the latest developments in this project!

![Voice Cloning Example](https://github.com/universalbit-dev/universalbit-dev/blob/main/hacking/images/voice_cloned.JPG)

---
