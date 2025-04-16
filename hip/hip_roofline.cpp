/**
 * @file hip_roofline.cpp
 * @brief Demonstrates matrix reduction sum using HIP (Heterogeneous Interface for Portability).
 *
 * This program performs a reduction operation on a large matrix using HIP. 
 * It utilizes GPU parallelism to compute the sum of matrix elements efficiently. 
 * The kernel leverages shared memory for optimization and coalesced memory access for better performance.
 *
 * Key Features:
 * - Initializes a matrix with random values.
 * - Uses shared memory and reduction within blocks to compute partial sums.
 * - Optimized for coalesced read access for better memory performance.
 * - Measures kernel execution time in milliseconds.
 *
 * Constants:
 * - `N`: Matrix dimension (N x N elements).
 * - `BLOCK_SIZE`: Number of threads per block.
 *
 * HIP Kernel:
 * - `matrixReduction`: Reduces a matrix by summing elements using shared memory and block-wise reduction.
 *
 * Functions:
 * - `initMatrix`: Initializes the host matrix with random values.
 *
 * Usage:
 * - Compile with HIP-enabled environment.
 * - Runs on GPUs that support HIP.
 *
 * @dependencies HIP runtime library.
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Constants for matrix dimensions and block size
#define N 1024
#define BLOCK_SIZE 32

// HIP kernel for reduction sum
__global__ void matrixReduction(const float* __restrict__ input, float* output, int numElements) {
    __shared__ float sharedData[BLOCK_SIZE];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory with coalesced reads
    sharedData[tid] = (globalIdx < numElements) ? input[globalIdx] : 0;
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of reduction to output
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

// Helper function to initialize matrix
void initMatrix(std::vector<float>& matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int numElements = N * N;
    size_t size = numElements * sizeof(float);

    // Host matrices
    std::vector<float> h_input(numElements);
    std::vector<float> h_output(N, 0);

    // Initialize input matrix
    initMatrix(h_input, numElements);

    // Device matrices
    float *d_input, *d_output;
    hipMalloc(&d_input, size);
    hipMalloc(&d_output, N * sizeof(float));

    // Copy input to device
    hipMemcpy(d_input, h_input.data(), size, hipMemcpyHostToDevice);

    // Launch kernel
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((numElements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    hipLaunchKernelGGL(matrixReduction, gridSize, blockSize, 0, 0, d_input, d_output, numElements);
    hipEventRecord(stop);

    // Copy result back to host
    hipMemcpy(h_output.data(), d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    // Calculate elapsed time
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Reduction completed in " << milliseconds << " ms." << std::endl;

    // Cleanup
    hipFree(d_input);
    hipFree(d_output);

    return 0;
}
