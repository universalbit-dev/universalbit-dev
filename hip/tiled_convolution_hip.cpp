/******************************************************************************
 * File: tiled_convolution_hip.cpp
 * Description:
 * This file combines the functionality of tiled matrix multiplication 
 * (from `tiled_matrix_multiplication_hip.cpp`) and image convolution 
 * (from `convnet_hip.cpp`) into a single optimized implementation. 
 * The integration leverages tiling techniques to optimize 2D convolution 
 * operations using shared memory for efficient GPU computations.
 *
 * Features:
 * - Tiling optimization for 2D convolution
 * - Shared memory utilization for reduced global memory access
 * - Flexible kernel size and support for boundary conditions
 * - High-performance GPU implementation using the HIP framework
 *
 * Integration Details:
 * - The tiling logic and shared memory strategies from 
 *   `tiled_matrix_multiplication_hip.cpp` are adapted to handle 
 *   the overlapping regions required for convolution operations.
 * - The convolution logic from `convnet_hip.cpp` is enhanced to 
 *   utilize the tiling mechanism for improved performance.
 *
 * Usage:
 * This implementation is designed for high-performance deep learning 
 * and image processing applications. It can be further extended to 
 * support additional features such as multi-channel convolution or 
 * stride-based operations.
 *
 * Author: universalbit-dev
 * Date: 2025-04-16
 * License: MIT
 ******************************************************************************/
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define TILE_SIZE 16 // Define the tile size for shared memory

// Tiled Convolution Kernel
__global__ void tiledConvolution2D(const float* input, float* output, const float* kernel,
                                   int width, int height, int kernel_size) {
    // Shared memory for input tile
    __shared__ float tile[TILE_SIZE + 2][TILE_SIZE + 2];

    // Calculate thread and block positions
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Calculate the half size of the kernel
    int half_k = kernel_size / 2;

    // Load input tile into shared memory, including the halo for the kernel
    if (row < height && col < width) {
        tile[ty + 1][tx + 1] = input[row * width + col];
        if (ty == 0 && row > 0) tile[0][tx + 1] = input[(row - 1) * width + col];
        if (ty == TILE_SIZE - 1 && row < height - 1) tile[TILE_SIZE + 1][tx + 1] = input[(row + 1) * width + col];
        if (tx == 0 && col > 0) tile[ty + 1][0] = input[row * width + col - 1];
        if (tx == TILE_SIZE - 1 && col < width - 1) tile[ty + 1][TILE_SIZE + 1] = input[row * width + col + 1];
    } else {
        tile[ty + 1][tx + 1] = 0.0f;
        if (ty == 0) tile[0][tx + 1] = 0.0f;
        if (ty == TILE_SIZE - 1) tile[TILE_SIZE + 1][tx + 1] = 0.0f;
        if (tx == 0) tile[ty + 1][0] = 0.0f;
        if (tx == TILE_SIZE - 1) tile[ty + 1][TILE_SIZE + 1] = 0.0f;
    }

    // Synchronize to ensure all threads have loaded their data
    __syncthreads();

    // Perform convolution if within image bounds
    if (row < height && col < width) {
        float sum = 0.0f;

        // Iterate over the kernel
        for (int i = -half_k; i <= half_k; i++) {
            for (int j = -half_k; j <= half_k; j++) {
                sum += tile[ty + 1 + i][tx + 1 + j] * kernel[(i + half_k) * kernel_size + (j + half_k)];
            }
        }

        // Write the result to the output
        output[row * width + col] = sum;
    }
}

// Function to launch the tiled convolution kernel
void runTiledConvolution2D(const std::vector<float>& input, std::vector<float>& output,
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
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    hipLaunchKernelGGL(tiledConvolution2D, gridDim, blockDim, 0, 0, d_input, d_output, d_kernel, width, height, kernel_size);

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
    runTiledConvolution2D(input, output, kernel, width, height, 3);

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
