# Convolutional Neural Network (CNN) Overview

This repository contains key components for creating and training Convolutional Neural Networks (CNNs) and implementing Deep Q-Learning (DQN) using JavaScript.

## Files Overview

### [convnet.js](https://github.com/universalbit-dev/universalbit-dev/blob/main/convolutional_neural_network/convnet.js)

The `convnet.js` file is a library for building and training CNNs. Key components include:
- **Utility Functions**: Functions for generating random numbers, creating zero arrays, and other array utilities.
- **Volume Class (`Vol`)**: Represents a 3D volume of numbers, essential for holding data, weights, and gradients.
- **Neural Network Layers**: Includes layers like `ConvLayer`, `FullyConnLayer`, `ReluLayer`, and more.
- **Network Class (`Net`)**: Manages a sequence of layers and handles forward and backward propagation.
- **Trainer Classes**: Implements training algorithms like SGD to optimize the network.

For further details, visit the [convnet.js file](https://github.com/universalbit-dev/universalbit-dev/blob/main/convolutional_neural_network/convnet.js).

---

### [deepqlearn.js](https://github.com/universalbit-dev/universalbit-dev/blob/main/convolutional_neural_network/deepqlearn.js)

The `deepqlearn.js` file is part of a reinforcement learning library utilizing Deep Q-Learning (DQN). Key components include:
- **Experience Class**: Stores state, action, reward, and next state for each interaction.
- **Brain Class**: Implements the core DQN algorithm, managing the neural network, experience replay, and epsilon-greedy policy.
- **Temporal Difference Learning**: Uses past experiences and replay memory for learning.

For further details, visit the [deepqlearn.js file](https://github.com/universalbit-dev/universalbit-dev/blob/main/convolutional_neural_network/deepqlearn.js).

---

## Deep Q-Learning (DQN) Algorithm

Deep Q-Learning (DQN) is a reinforcement learning approach that combines Q-Learning with deep neural networks. Here are the key components:

1. **Q-Learning Basics**:
   - Q-Learning estimates the cumulative reward for taking an action in a given state and following a policy thereafter.

2. **Deep Neural Network**:
   - A neural network approximates the Q-value function, taking a state \(s\) as input and outputting Q-values for actions.

3. **Experience Replay**:
   - Stores past experiences \((s, a, r, s')\) in a replay buffer.
   - Randomly samples mini-batches to break correlations and improve learning.

4. **Target Network**:
   - A separate target network computes target Q-values, updated periodically to stabilize training.

5. **Epsilon-Greedy Policy**:
   - Balances exploration and exploitation:
     - Random actions (\(\epsilon\)) for exploration.
     - Best actions (\(1-\epsilon\)) for exploitation.

6. **Loss Function**:
   - Minimizes the Mean Squared Error (MSE) between predicted and target Q-values:
   
DQN is widely used in applications like robotics, gaming, and autonomous decision-making.

---

## Additional Resources

- [Cheatsheet on Convolutional Neural Networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)
- [Convnet.js Trainers](https://cs.stanford.edu/people/karpathy/convnetjs/demo/trainers.html)
- [Convnet.js Library](https://cs.stanford.edu/people/karpathy/convnetjs/)
- [Convnet.js Documentation](https://cs.stanford.edu/people/karpathy/convnetjs/docs.html)
- [Convnet.js Npm Package](https://www.npmjs.com/package/convnet)

---

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/universalbit-dev/universalbit-dev.git
   ```

2. Navigate to the `convolutional_neural_network` directory:
   ```bash
   cd universalbit-dev/convolutional_neural_network
   ```

3. Explore the `convnet.js` and `deepqlearn.js` files for implementation details.

---

## Contribution

Contributions are welcome! If you would like to improve this project, please follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---
