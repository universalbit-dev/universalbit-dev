/**
 * @file tiled_matrix_multiplication_hip.cpp
 * @brief HIP implementation of a tiled matrix multiplication kernel.
 *
 * This implementation is inspired by the design philosophy of underscore.js, 
 * particularly its modular and reusable methods. The code demonstrates 
 * concepts such as tiling for coalesced reads/writes, and LDS (local data share) 
 * optimization for efficient GPU computations.
 * 
 * Features:
 * - Tiled approach for efficient matrix multiplication.
 * - Coalesced memory access for global memory reads/writes.
 * - Shared memory (LDS) utilization for reduced global memory access.
 * - Modular design with reusable matrix multiplication logic.
 * 
 * Author: GitHub Copilot
 * Date: 2025-04-16
 */

#include <hip/hip_runtime.h>
#include <iostream>

// Define the TILE_SIZE for the shared memory block
#define TILE_SIZE 16

// HIP kernel for tiled matrix multiplication
__global__ void matrixMulTiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory for tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulator for the result
    float value = 0;

    // Loop over all tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load data into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Perform computation on the tile
        for (int i = 0; i < TILE_SIZE; i++) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        // Synchronize to ensure all threads are done with this tile
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Helper function to launch the kernel
void matrixMultiply(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, M * K * sizeof(float));
    hipMalloc(&d_B, K * N * sizeof(float));
    hipMalloc(&d_C, M * N * sizeof(float));

    // Copy data to device
    hipMemcpy(d_A, A, M * K * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, K * N * sizeof(float), hipMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    hipLaunchKernelGGL(matrixMulTiled, gridDim, blockDim, 0, 0, d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    hipMemcpy(C, d_C, M * N * sizeof(float), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}

int main() {
    // Matrix dimensions
    int M = 64, N = 64, K = 64;

    // Allocate and initialize host memory
    float *A = new float[M * K];
    float *B = new float[K * N];
    float *C = new float[M * N];

    for (int i = 0; i < M * K; i++) A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) B[i] = 1.0f;

    // Perform matrix multiplication
    matrixMultiply(A, B, C, M, N, K);

    // Print a portion of the result
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free host memory
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
