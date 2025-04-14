/**
 * @file bindAll_kernel.cpp
 * @brief HIP example of a conceptual translation of _.bindAll in Underscore.js.
 *
 * This example demonstrates how to bind methods to an object-like structure in HIP,
 * ensuring that the correct context is maintained when executing GPU kernels.
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
    int threadsPerBlock = 1;
    int blocksPerGrid = 1;
    hipLaunchKernelGGL(bindAllKernel, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, deviceObj, 42);

    // Copy the modified object back to the host
    hipMemcpy(&hostObj, deviceObj, sizeof(Object), hipMemcpyDeviceToHost);

    // Print the result
    hostObj.print();

    // Free device memory
    hipFree(deviceObj);

    return 0;
}
