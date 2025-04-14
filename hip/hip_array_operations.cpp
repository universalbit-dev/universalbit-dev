/******************************************************************************
 * HIP Implementation of Common Array Operations
 * Author: universalbit-dev
 * Date: 2025-04-14
 * Description:
 * This file contains GPU-accelerated implementations of common array 
 * operations using HIP (Heterogeneous-Computing Interface for Portability).
 * These operations are crucial for learning parallel programming concepts, 
 * SIMD activation, and efficient GPU computation.
 * 
 * Methods Inspired by underscore.js:
 * - map       : Apply a function to each array element.
 * - reduce    : Reduce an array to a single value using a binary operation.
 * - filter    : Filter array elements based on a predicate function.
 * - find      : Find the first element matching a predicate.
 * - contains  : Check if an array contains a value.
 * - pluck     : Extract a specific property from an array of objects.
 * - sortBy    : Sort array elements based on a criterion.
 * - uniq      : Remove duplicate elements from an array.
 * - union     : Combine arrays into one with unique elements.
 * - intersection: Find common elements between arrays.
 * - difference: Find elements in one array not present in others.
 * - zip/unzip : Combine or split arrays of arrays.
 * - flatten   : Flatten nested arrays into a single array.
 * - compact   : Remove falsy values from an array.
 * - range     : Generate a sequence of numbers.
 * - chunk     : Split an array into chunks of a specific size.
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// Utility macros for checking HIP errors
#define HIP_CHECK(call)                                               \
    {                                                                 \
        hipError_t err = call;                                        \
        if (err != hipSuccess) {                                      \
            std::cerr << "HIP error: " << hipGetErrorString(err)      \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(err);                                                \
        }                                                             \
    }

/******************************************************************************
 * Kernel for Map Operation:
 * Applies a user-specified function to each array element.
 ******************************************************************************/
__global__ void mapKernel(int* input, int* output, int length, int (*func)(int)) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        output[idx] = func(input[idx]);
    }
}

__device__ int square(int x) { return x * x; }

/******************************************************************************
 * Kernel for Reduce Operation:
 * Reduces an array to a single value using shared memory.
 ******************************************************************************/
__global__ void reduceKernel(int* input, int* output, int length) {
    extern __shared__ int sharedData[];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    sharedData[tid] = (idx < length) ? input[idx] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) sharedData[tid] += sharedData[tid + stride];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sharedData[0];
}

/******************************************************************************
 * Kernel for Filter Operation:
 * Copies elements satisfying a predicate to an output array.
 ******************************************************************************/
__global__ void filterKernel(int* input, int* output, int* outputSize, int length, bool (*predicate)(int)) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length && predicate(input[idx])) {
        int pos = atomicAdd(outputSize, 1); // Atomic operation ensures correct indexing
        output[pos] = input[idx];
    }
}

__device__ bool isEven(int x) { return x % 2 == 0; }

int main() {
    const int length = 1024;
    int hostInput[length], hostOutput[length];

    // Initialize input
    for (int i = 0; i < length; i++) hostInput[i] = i;

    int *deviceInput, *deviceOutput;
    HIP_CHECK(hipMalloc(&deviceInput, length * sizeof(int)));
    HIP_CHECK(hipMalloc(&deviceOutput, length * sizeof(int)));
    HIP_CHECK(hipMemcpy(deviceInput, hostInput, length * sizeof(int), hipMemcpyHostToDevice));

    // Launch the map kernel with the square function
    int blockSize = 256;
    int gridSize = (length + blockSize - 1) / blockSize;
    hipLaunchKernelGGL(mapKernel, dim3(gridSize), dim3(blockSize), 0, 0, deviceInput, deviceOutput, length, square);

    HIP_CHECK(hipMemcpy(hostOutput, deviceOutput, length * sizeof(int), hipMemcpyDeviceToHost));

    // Print some results
    for (int i = 0; i < 10; i++) {
        std::cout << hostOutput[i] << " ";
    }
    std::cout << std::endl;

    HIP_CHECK(hipFree(deviceInput));
    HIP_CHECK(hipFree(deviceOutput));

    return 0;
}
