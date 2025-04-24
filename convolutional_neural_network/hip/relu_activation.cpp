/*
 * File: relu_activation.cpp
 * Description: Implementation of the ReLU (Rectified Linear Unit) activation function using HIP (Heterogeneous Interface for Portability).
 * Repository: universalbit-dev/universalbit-dev
 * Author: universalbit-dev
 * Date: April 24, 2025
 * 
 * This file implements the ReLU activation function for neural networks, leveraging HIP for GPU acceleration.
 * The ReLU function introduces non-linearity by setting all negative input values to zero while keeping positive values unchanged.
 * 
 * Usage:
 * - The `relu_activation` kernel runs on the GPU to apply the ReLU function element-wise across an input array.
 * - The `apply_relu` function configures and launches the kernel from the host.
 * 
 * License:
 * Refer to the repository's LICENSE file for licensing information.
 */

__global__ void relu_activation(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Host code
void apply_relu(float* device_input, float* device_output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu_activation<<<blocksPerGrid, threadsPerBlock>>>(device_input, device_output, size);
    hipDeviceSynchronize();
}
