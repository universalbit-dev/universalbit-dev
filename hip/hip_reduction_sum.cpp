/**
 * @file hip_reduction_sum.cpp
 * @brief HIP implementation of `_.reduce` and `_.each` functions inspired by Underscore.js.
 *
 * This program provides a GPU-accelerated implementation of the `_.reduce` and `_.each` 
 * functions from the Underscore.js library using HIP. The following functionalities are included:
 *
 * 1. **`reduce_kernel`**:
 *    - Mimics the behavior of `_.reduce` by performing a parallel reduction to compute the sum of an array.
 *    - Utilizes shared memory (LDS) for efficient intermediate computations and optimized memory access.
 *
 * 2. **`each_kernel`**:
 *    - Analogous to `_.each`, applies a specified operation to every element of the input array in parallel.
 *    - Demonstrates how a simple operation (e.g., adding a constant) can be applied to GPU memory.
 *
 * The code provided aligns closely with the functional behavior of Underscore.js's `_.reduce` and `_.each`, 
 * but it's designed to leverage GPU parallelism for performance optimization.
 *
 * @usage
 * 1. Compile the program using:
 *    ```bash
 *    hipcc hip_reduction_sum.cpp -o hip_reduction_sum
 *    ```
 * 2. Run the executable:
 *    ```bash
 *    ./hip_reduction_sum
 *    ```
 * 
 * @example
 * Example output:
 * ```
 * Sum of array: 55
 * Array after adding 10 to each element: 11 12 13 14 15 16 17 18 19 20
 * ```
 *
 * @dependencies
 * - HIP runtime library
 *
 * @note
 * - Ensure that the HIP runtime is properly installed on your system.
 * - This program demonstrates the translation of JavaScript utility functions to GPU-accelerated equivalents.
 *
 * @author universalbit-dev
 * @date 2025-04-16
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate

// Kernel for parallel reduction using shared memory (LDS)
__global__ void reduce_kernel(const int* d_input, int* d_output, int n) {
    extern __shared__ int sdata[]; // Shared memory for partial sums
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load input into shared memory
    sdata[tid] = (idx < n) ? d_input[idx] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's result to the global memory
    if (tid == 0) {
        d_output[blockIdx.x] = sdata[0];
    }
}

// Kernel for applying a function to each element of an array (akin to `_.each`)
__global__ void each_kernel(int* d_array, int n, int operation) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Example operation: Add a constant value (operation) to each element
        d_array[idx] += operation;
    }
}

// Host function for reduction
int reduce(const std::vector<int>& input) {
    int n = input.size();
    int blockSize = 256; // Threads per block
    int numBlocks = (n + blockSize - 1) / blockSize;

    int* d_input;
    int* d_partialSums;
    hipMalloc(&d_input, n * sizeof(int));
    hipMalloc(&d_partialSums, numBlocks * sizeof(int));

    hipMemcpy(d_input, input.data(), n * sizeof(int), hipMemcpyHostToDevice);

    // Launch reduction kernel
    reduce_kernel<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_input, d_partialSums, n);

    // Copy partial sums back to host
    std::vector<int> partialSums(numBlocks);
    hipMemcpy(partialSums.data(), d_partialSums, numBlocks * sizeof(int), hipMemcpyDeviceToHost);

    // Final reduction on host
    int totalSum = std::accumulate(partialSums.begin(), partialSums.end(), 0);

    hipFree(d_input);
    hipFree(d_partialSums);

    return totalSum;
}

// Host function for applying a function to each element (akin to `_.each`)
void each(std::vector<int>& array, int operation) {
    int n = array.size();
    int* d_array;

    hipMalloc(&d_array, n * sizeof(int));
    hipMemcpy(d_array, array.data(), n * sizeof(int), hipMemcpyHostToDevice);

    // Launch kernel to apply operation to each element
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    each_kernel<<<numBlocks, blockSize>>>(d_array, n, operation);

    // Copy results back to host
    hipMemcpy(array.data(), d_array, n * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_array);
}

int main() {
    // Example input data
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // 1. Perform reduction (sum of array)
    int sum = reduce(input);
    std::cout << "Sum of array: " << sum << std::endl;

    // 2. Apply an operation to each element (e.g., add 10)
    each(input, 10);
    std::cout << "Array after adding 10 to each element: ";
    for (int val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
