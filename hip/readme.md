# HIP (Heterogeneous-Compute Interface for Portability) Programming Model

This document introduces the **HIP programming model**, a framework developed by AMD to enable portable GPU programming across AMD and NVIDIA GPUs. The guide explains HIP's key features, applications, programming mechanics, and installation steps.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Applications](#applications)
4. [How the HIP Model Works](#how-the-hip-model-works)
5. [Why Use HIP?](#why-use-hip)
6. [Setup and Installation](#setup-and-installation)
   - [Ubuntu Update and Upgrade](#ubuntu-update-and-upgrade)
   - [AMD Repository Setup](#amd-repository-setup)
7. [Resources and Links](#resources-and-links)

---

## Introduction

The **HIP (Heterogeneous-Compute Interface for Portability)** model is a programming framework developed by AMD. HIP allows developers to write portable GPU-accelerated code that works seamlessly on both AMD and NVIDIA GPUs. It leverages AMD's ROCm (Radeon Open Compute) platform and provides high performance, portability, and open-source flexibility.

[Learn more about the HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html).

---

## Key Features

1. **Portability**:
   - Write GPU-accelerated code that works on both AMD and NVIDIA GPUs.
   - Use different backends, such as ROCm for AMD GPUs or CUDA for NVIDIA GPUs.

2. **CUDA-like Syntax**:
   - HIP uses CUDA-like syntax, making it easier for developers familiar with CUDA.
   - Most CUDA APIs have direct counterparts in HIP.

3. **Performance**:
   - HIP provides high-performance GPU programming capabilities by leveraging AMD GPUs through the ROCm platform.

4. **Open-source**:
   - HIP is part of the open-source ROCm ecosystem, allowing developers to contribute and customize.

---

## Applications

- **High-Performance Computing (HPC)**:
  - Ideal for scientific computing, simulations, and other performance-intensive workloads.

- **Machine Learning and AI**:
  - Frameworks like TensorFlow and PyTorch use HIP for AMD GPU support.

- **Graphics and Simulation**:
  - Used in rendering engines like Blender and applications requiring GPU acceleration.

---

## How the HIP Model Works

1. **Programming with HIP**:
   - Developers write GPU kernels and host code using HIP APIs and syntax.
   - Example:
     ```cpp
     #include <hip/hip_runtime.h>

     __global__ void vector_add(const float* A, const float* B, float* C, int N) {
         int i = blockIdx.x * blockDim.x + threadIdx.x;
         if (i < N) {
             C[i] = A[i] + B[i];
         }
     }
     ```

   [Detailed documentation on HIP kernels](https://github.com/universalbit-dev/universalbit-dev/blob/main/hip/hip_kernel_doc.md).

2. **Compilation**:
   - The HIP compiler translates the code to the appropriate backend (e.g., ROCm for AMD or CUDA for NVIDIA).

3. **Execution on GPUs**:
   - Compiled code is executed on the target GPU, leveraging its hardware capabilities.

---

## Why Use HIP?

- Maintain a single codebase that targets both AMD and NVIDIA GPUs.
- Leverage open-source GPU programming tools and avoid vendor lock-in.
- Take advantage of AMD's ROCm ecosystem for high-performance GPU computing.

---

## Setup and Installation

### Ubuntu Update and Upgrade
Ensure your system is up to date:
```bash
sudo apt-get update && sudo apt-get upgrade
```

### AMD Repository Setup
1. Add the AMD repository:
   ```bash
   wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.3.60304-1_all.deb
   sudo chmod a+x amdgpu-install_6.3.60304-1_all.deb
   ```

2. Install the package using `gdebi`:
   ```bash
   sudo apt install gdebi
   sudo gdebi ./amdgpu-install_6.3.60304-1_all.deb
   ```

3. Install HIP:
   ```bash
   sudo amdgpu-install --usecase=hip
   ```

### Additional Resources for Setup
- [Blender HIP Rendering Documentation](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html)
- [AMD Radeon Software Installation Instructions](https://amdgpu-install.readthedocs.io/en/latest/)

---

## Resources and Links

- [Introduction to HIP Programming Model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)
- [Enhance HArmadillium Project (Experimental)](https://github.com/universalbit-dev/HArmadillium?tab=readme-ov-file)
- [AMD HIP Kernel Documentation](https://github.com/universalbit-dev/universalbit-dev/blob/main/hip/hip_kernel_doc.md)
- [Ubuntu Update and Upgrade Guide](https://help.ubuntu.com/community/UbuntuUpdates)
