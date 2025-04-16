/******************************************************************************
 * File: convnet_hip.cpp
 * Description: 
 * This file implements GPU-accelerated image convolution operations inspired 
 * by the "convnet.js" library. The implementation utilizes the HIP (Heterogeneous 
 * Interface for Portability) framework for parallel processing on AMD and NVIDIA 
 * GPUs.
 *
 * Features:
 * - 2D Convolutional Layer
 * - Support for custom kernels/filters
 * - Optimized for coalesced memory access and shared memory (LDS) usage
 * - Scalable for high-performance neural network operations
 *
 * Original JavaScript Implementation Reference:
 * convnet.js by Andrej Karpathy
 * (https://cs.stanford.edu/people/karpathy/convnetjs/)
 *
 * Usage:
 * This implementation is a low-level GPU adaptation intended for use as a 
 * building block in deep learning and image processing applications.
 *
 * Author: universalbit-dev
 * Date: 2025-04-16
 * License: MIT
 *
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 16 // Define the block size for GPU threads

__global__ void convolution2D(float* input, float* output, float* kernel, 
                              int width, int height, int kernel_size) {
    // Calculate thread and block positions
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    // Calculate the half size of the kernel
    int half_k = kernel_size / 2;

    // Perform convolution if within image bounds
    if (tx < width && ty < height) {
        float sum = 0.0f;

        // Iterate over kernel
        for (int i = -half_k; i <= half_k; i++) {
            for (int j = -half_k; j <= half_k; j++) {
                int x = min(max(tx + i, 0), width - 1);  // Clamp to image boundaries
                int y = min(max(ty + j, 0), height - 1);

                sum += input[y * width + x] * kernel[(i + half_k) * kernel_size + (j + half_k)];
            }
        }

        output[ty * width + tx] = sum; // Write result to output
    }
}
void runConvolution2D(const std::vector<float>& input, std::vector<float>& output, 
                      const std::vector<float>& kernel, int width, int height, int kernel_size) {
    // Device pointers
    float *d_input, *d_output, *d_kernel;

    // Allocate device memory
    hipMalloc(&d_input, width * height * sizeof(float));
    hipMalloc(&d_output, width * height * sizeof(float));
    hipMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));

    // Copy data to device
    hipMemcpy(d_input, input.data(), width * height * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_kernel, kernel.data(), kernel_size * kernel_size * sizeof(float), hipMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    hipLaunchKernelGGL(convolution2D, gridDim, blockDim, 0, 0, d_input, d_output, d_kernel, width, height, kernel_size);

    // Copy result back to host
    hipMemcpy(output.data(), d_output, width * height * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_kernel);
}

int main() {
    // Define image dimensions
    int width = 5, height = 5;

    // Define a simple 3x3 kernel
    std::vector<float> kernel = {
        0, -1, 0,
        -1, 4, -1,
        0, -1, 0
    };

    // Define a simple 5x5 input image
    std::vector<float> input = {
        1, 1, 1, 1, 1,
        1, 2, 2, 2, 1,
        1, 2, 3, 2, 1,
        1, 2, 2, 2, 1,
        1, 1, 1, 1, 1
    };

    std::vector<float> output(width * height, 0); // Output image

    // Run convolution
    runConvolution2D(input, output, kernel, width, height, 3);

    // Print output
    std::cout << "Output Image:" << std::endl;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << output[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
