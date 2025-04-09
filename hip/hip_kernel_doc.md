# HIP Kernel Documentation: `vector_add`

This document explains the purpose, functionality, and implementation details of the HIP kernel `vector_add` and how it works in conjunction with the HIP runtime.

---

### **Overview**
The `vector_add` kernel performs **element-wise addition** of two input vectors (arrays) `A` and `B` and stores the result in an output vector `C`. It is designed to run on a GPU using the HIP (Heterogeneous-Compute Interface for Portability) runtime. HIP enables developers to write portable GPU code that can run on both AMD and NVIDIA GPUs.

---

### **Code Explanation**

#### **Header Inclusion**
```cpp
#include <hip/hip_runtime.h>
```
- The `hip/hip_runtime.h` header is included to provide access to HIP runtime APIs and macros required for GPU programming. This header defines the necessary constructs for launching GPU kernels, managing memory, and handling execution.

---

#### **Kernel Function Definition**
```cpp
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    ...
}
```
- **`__global__`**:
  - This specifier designates `vector_add` as a **kernel function**. Kernel functions are executed on the device (GPU) but are called from the host (CPU).
- **Parameters**:
  - `const float* A`: Pointer to the first input vector. The `const` qualifier ensures the data in `A` is not modified by the kernel.
  - `const float* B`: Pointer to the second input vector. Similarly, this is read-only within the kernel.
  - `float* C`: Pointer to the output vector where the results of the addition will be stored.
  - `int N`: The total number of elements in the vectors. This is used to ensure boundary checking so that threads do not access out-of-bounds memory.

---

#### **Thread Index Calculation**
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
- **Threading Model in HIP**:
  - GPU computation is parallelized using **threads**, which are grouped into **blocks**, and blocks are organized into a **grid**.
- **Thread Identification**:
  - `threadIdx.x`: The thread's index **within its block**.
  - `blockDim.x`: The total number of threads **per block**.
  - `blockIdx.x`: The index of the block **within the grid**.
  - The formula `blockIdx.x * blockDim.x + threadIdx.x` calculates the **global thread ID** across all blocks and threads. This ensures each thread computes a unique element of the vectors.

---

#### **Boundary Check**
```cpp
if (i < N) {
    ...
}
```
- **Purpose**:
  - The number of threads launched on the GPU may exceed the size of the vectors (`N`). This check ensures that threads with a global ID `i` outside the valid range `[0, N-1]` do not execute the addition operation, preventing **out-of-bounds memory access**, which could lead to undefined behavior or crashes.

---

#### **Element-wise Vector Addition**
```cpp
C[i] = A[i] + B[i];
```
- Each thread performs the addition operation for a single element of the vectors:
  - It reads the `i`-th element of vectors `A` and `B`.
  - It computes their sum.
  - It writes the result to the `i`-th position of the output vector `C`.

---

### **Kernel Execution Context**

To execute the `vector_add` kernel, it must be launched from the host (CPU) with appropriate **grid** and **block dimensions**. Below is a step-by-step example demonstrating how to launch the kernel.

---

#### **Host Code Example**
```cpp
#include <hip/hip_runtime.h>
#include <iostream>

// Vector size
int N = 1024;

// Define block and grid dimensions
dim3 blockDim(256); // 256 threads per block
dim3 gridDim((N + blockDim.x - 1) / blockDim.x); // Number of blocks needed

// Allocate memory for vectors on the host
float* A = new float[N];
float* B = new float[N];
float* C = new float[N];

// Initialize input vectors
for (int i = 0; i < N; i++) {
    A[i] = i;
    B[i] = 2 * i;
}

// Allocate memory on the device
float *d_A, *d_B, *d_C;
hipMalloc(&d_A, N * sizeof(float));
hipMalloc(&d_B, N * sizeof(float));
hipMalloc(&d_C, N * sizeof(float));

// Copy data from host to device
hipMemcpy(d_A, A, N * sizeof(float), hipMemcpyHostToDevice);
hipMemcpy(d_B, B, N * sizeof(float), hipMemcpyHostToDevice);

// Launch the kernel
hipLaunchKernelGGL(vector_add, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N);

// Copy result back from device to host
hipMemcpy(C, d_C, N * sizeof(float), hipMemcpyDeviceToHost);

// Validate results
for (int i = 0; i < N; i++) {
    if (C[i] != A[i] + B[i]) {
        std::cerr << "Error at index " << i << std::endl;
        break;
    }
}

// Free memory on device and host
hipFree(d_A);
hipFree(d_B);
hipFree(d_C);
delete[] A;
delete[] B;
delete[] C;
```

---

### **Kernel Launch Parameters**
1. **Grid and Block Dimensions**:
   - `blockDim`: Specifies the number of threads per block. In this example, each block contains 256 threads.
   - `gridDim`: Specifies the number of blocks in the grid. It is calculated as `(N + blockDim.x - 1) / blockDim.x` to ensure all elements are covered, even if `N` is not a multiple of `blockDim.x`.

2. **Memory Transfers**:
   - `hipMalloc`: Allocates memory on the device (GPU).
   - `hipMemcpy`: Copies data between host and device.

3. **Kernel Launch Syntax**:
   - `hipLaunchKernelGGL(vector_add, gridDim, blockDim, 0, 0, d_A, d_B, d_C, N);`
     - The first two parameters specify the grid and block dimensions.
     - The next two parameters specify shared memory size and stream (set to `0` here).
     - The remaining parameters are arguments to the kernel function.

---

### **Performance Considerations**
1. **Thread Utilization**:
   - Ensure the number of threads launched matches or exceeds the size of the vectors (`N`).
   - Use appropriate block and grid dimensions to maximize GPU occupancy.

2. **Memory Coalescing**:
   - Accessing consecutive elements in global memory (e.g., `A[i]` and `B[i]`) ensures coalesced memory access, which improves performance.

3. **Boundary Checking**:
   - Always include a boundary check (`if (i < N)`) to avoid invalid memory accesses, especially when `N` is not a multiple of `blockDim.x`.

---

### **Key Takeaways**
- The `vector_add` kernel is a simple example of **data parallelism** implemented using HIP.
- It demonstrates fundamental concepts of GPU programming, such as thread hierarchy, memory management, and kernel execution.
- This kernel can be extended or modified to perform more complex operations while adhering to similar principles.
