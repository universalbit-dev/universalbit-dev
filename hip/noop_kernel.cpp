/**
 * @file noop_kernel.cpp
 * @brief HIP implementation of a no-operation (noop) kernel.
 *
 * This file contains a HIP kernel equivalent to the `_noop` function in underscore.js.
 * The kernel performs no operations and serves as a placeholder or demonstration of
 * a minimal HIP kernel implementation.
 *
 * @author universalbit-dev
 * @date 2025-04-14
 */

#include <hip/hip_runtime.h>
#include <iostream>

// A noop kernel that does nothing
__global__ void noopKernel() {
    // No operation here
}

int main() {
    // Launch the noop kernel
    int threadsPerBlock = 1;
    int blocksPerGrid = 1;

    hipLaunchKernelGGL(noopKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0);

    // Synchronize to ensure kernel execution is complete
    hipDeviceSynchronize();

    std::cout << "Noop kernel executed." << std::endl;

    return 0;
}
