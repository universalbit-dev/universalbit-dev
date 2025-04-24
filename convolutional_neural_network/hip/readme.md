# Convolutional Neural Network (HIP)

This directory contains the implementation of neural network components using HIP (Heterogeneous Interface for Portability) for GPU acceleration. These components are designed for high-performance computation in convolutional neural networks (CNNs).

## Overview

The implementation leverages HIP to utilize GPU hardware for efficient matrix operations, activation functions, and other key processes used in training and running CNNs. HIP provides a portable interface that supports both AMD and NVIDIA GPUs.

### Key Features
- **ReLU Activation**: Implements the Rectified Linear Unit (ReLU) function for introducing non-linearity in neural networks.
- **GPU Acceleration**: All operations in this module are optimized for parallel execution on GPUs.
- **Portability**: The code is written with HIP, making it compatible with AMD and NVIDIA GPU architectures.

## Files

- **`relu_activation.cpp`**: Contains the GPU kernel and host-side function for applying the ReLU activation function. This file demonstrates efficient GPU-based computation for neural networks.

## Usage

1. **Prerequisites**:
   - HIP runtime installed. For installation instructions, refer to the [HIP documentation](https://github.com/ROCm-Developer-Tools/HIP).
   - A compatible GPU (AMD or NVIDIA).

2. **Compiling the Code**:
   To compile the code, use a HIP-compatible compiler such as `hipcc`. For example:
   ```bash
   hipcc relu_activation.cpp -o relu_activation
