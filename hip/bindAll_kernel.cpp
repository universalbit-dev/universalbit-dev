/**
 * @file bindAll_kernel.cpp
 * @brief HIP example of a conceptual translation of _.bindAll in Underscore.js.
 *
 * This example demonstrates how to bind methods to an object-like structure in HIP,
 * ensuring that the correct context is maintained when executing GPU kernels.
 *
 * Relation to SIMD (Single Instruction, Multiple Data), Wavefronts, and Compute Units (CUs):
 * 
 * Wavefronts:
 *     On AMD GPUs (e.g., R9 290x), threads are grouped into wavefronts. A wavefront typically consists of 64 threads.
 *     If fewer than 64 threads are launched (e.g., threadsPerBlock = 1), the wavefront is underutilized. Only 1 lane of the SIMD unit is active, and the remaining lanes are idle.
 * 
 * Compute Units (CUs):
 *     The AMD R9 290x has 44 Compute Units (CUs). To fully utilize the GPU's processing power, the kernel launch configuration should:
 *     - Use threadsPerBlock = 64 to match the wavefront size.
 *     - Use blocksPerGrid = 44 to utilize all CUs (one block per CU).
 *     - If the workload permits, blocksPerGrid can be increased further to improve GPU occupancy, especially for memory-bound workloads.
 * 
 * Optimization and Efficiency:
 * 
 *     Using threadsPerBlock = 64 and blocksPerGrid = 44 ensures that:
 *         - Each wavefront is fully occupied (64 active threads).
 *         - All 44 CUs are utilized simultaneously.
 *     For better performance:
 *         - Increase threadsPerBlock to match the wavefront size (e.g., 64 threads for AMD GPUs).
 *         - Launch multiple blocks (blocksPerGrid >= 44) to utilize multiple CUs efficiently.
 *
 * Updated Launch Configuration:
 *     int threadsPerBlock = 64;   // Matches the wavefront size
 *     int blocksPerGrid = 44;     // Matches the number of Compute Units (CUs) on R9 290x
 *     hipLaunchKernelGGL(bindAllKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, deviceObj, 42);
 *
 * Kernel Execution Cycle Explanation:
 *     With threadsPerBlock = 64 and blocksPerGrid = 44:
 *         - The GPU can execute 44 wavefronts simultaneously (one per CU).
 *         - Each wavefront will execute in 1 cycle, assuming no memory or resource stalls.
 *     If the workload requires more wavefronts than available CUs, additional wavefronts will be executed in subsequent cycles.
 * 
 * @author universalbit-dev
 * @date 2025-04-14
 */

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Define a structure to represent the object with methods
struct Object {
    int value;

    // Method to initialize value
    __device__ void initialize(int val) {
        value = val;
    }

    // Method to double the value
    __device__ void doubleValue() {
        value *= 2;
    }

    // Method to print the value (executed on the CPU for demonstration)
    void print() const {
        std::cout << "Value: " << value << std::endl;
    }
};

// Define a kernel to bind and invoke methods on the object
__global__ void bindAllKernel(Object* obj, int initValue) {
    obj->initialize(initValue); // Bind and call initialize
    obj->doubleValue();         // Bind and call doubleValue
}

int main() {
    // Create an instance of Object on the host
    Object hostObj;

    // Allocate memory for the object on the device
    Object* deviceObj;
    hipMalloc(&deviceObj, sizeof(Object));

    // Copy the host object to the device
    hipMemcpy(deviceObj, &hostObj, sizeof(Object), hipMemcpyHostToDevice);

    // Launch the kernel to bind methods and execute them
    int threadsPerBlock = 64;   // Matches the wavefront size
    int blocksPerGrid = 44;     // Matches the number of Compute Units (CUs) on R9 290x
    hipLaunchKernelGGL(bindAllKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, deviceObj, 42);

    // Copy the modified object back to the host
    hipMemcpy(&hostObj, deviceObj, sizeof(Object), hipMemcpyDeviceToHost);

    // Print the result
    hostObj.print();

    // Free device memory
    hipFree(deviceObj);

    return 0;
}

/**
 * AMD R9 290 Hardware Specifications and SIMD Activation Limitations:
 * 
 * 1. Compute Units (CUs):
 *    - The R9 290 features 44 Compute Units (CUs), each capable of executing one wavefront (64 threads) at a time.
 * 
 * 2. SIMD Units:
 *    - Each CU contains multiple SIMD (Single Instruction, Multiple Data) units. However, the effective utilization of these units depends on the number of active threads per wavefront.
 *    - If fewer than 64 threads are launched per block, a portion of the SIMD lanes will remain idle, leading to underutilization of GPU resources.
 * 
 * 3. Wavefront Size:
 *    - The R9 290 uses a wavefront size of 64 threads.
 *    - To fully activate all SIMD lanes within a wavefront, ensure that threadsPerBlock is set to 64 or a multiple of 64.
 * 
 * 4. Memory-Bound Workloads:
 *    - For workloads that are memory-bound rather than compute-bound, increasing blocksPerGrid beyond the number of CUs (44) can help improve GPU occupancy.
 * 
 * Note:
 *    - These limitations are inherent to the hardware design of the R9 290 and must be accounted for when writing HIP kernels to maximize GPU efficiency.
 */
