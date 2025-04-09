[Introduction to the HIP programming model](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html)

The **HIP (Heterogeneous-Compute Interface for Portability)** model is a programming framework developed by AMD. It is designed to enable developers to write portable code for both AMD and NVIDIA GPUs with minimal changes. Here's a brief explanation of the HIP model:

---

### **Key Features of the HIP Model**
1. **Portability**:
   - HIP allows developers to write GPU-accelerated code in a way that works seamlessly on AMD GPUs and NVIDIA GPUs.
   - Code written in HIP can be compiled with different backends, such as ROCm for AMD GPUs or CUDA for NVIDIA GPUs.

2. **CUDA-like Syntax**:
   - HIP uses a CUDA-like syntax, making it easy for developers who are familiar with CUDA to transition their code to HIP.
   - Most CUDA APIs have a direct counterpart in HIP, simplifying the porting process.

3. **Performance**:
   - HIP provides high-performance GPU programming capabilities, leveraging the full power of AMD GPUs through the ROCm platform.

4. **Open-source**:
   - HIP is part of AMD's ROCm (Radeon Open Compute) platform, which is open-source, allowing developers to contribute and customize their GPU programming environment.

---

### **Applications of HIP**
- **High-Performance Computing (HPC)**:
  - HIP is often used in scientific computing, simulations, and other performance-intensive workloads.
- **Machine Learning and AI**:
  - Many frameworks, such as TensorFlow and PyTorch, use HIP as part of their AMD GPU backends.
- **Graphics and Simulation**:
  - HIP is used in rendering engines like Blender and other applications requiring GPU acceleration.

---

### **How the HIP Model Works**
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

2. **Compilation**:
   - The HIP compiler translates the code to the appropriate backend (e.g., ROCm for AMD or CUDA for NVIDIA).

3. **Execution on GPUs**:
   - The compiled code is executed on the target GPU, taking advantage of its hardware capabilities.

---

### **Why Use HIP?**
- To maintain a single codebase that can target both AMD and NVIDIA GPUs.
- To take advantage of open-source GPU programming tools and avoid vendor lock-in.
- To leverage AMD's ROCm ecosystem for high-performance GPU computing.

---

[Enhance HArmadillium Project -- Experimental --](https://github.com/universalbit-dev/HArmadillium?tab=readme-ov-file)
---
Ubuntu 24.04 LTS noble
* kernel version: 6.8.0-35-lowlatency
---
* Update and Upgrade Ubuntu Os
```bash
sudo apt-get update && sudo apt-get upgrade
```

* [AMD Repo](https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/)

```bash
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.3.60304-1_all.deb
sudo chmod a+x amdgpu-install_6.3.60304-1_all.deb 
```
[gdebi](https://wiki.ubuntu-it.org/AmministrazioneSistema/InstallareProgrammi/Gdebi)
```bash
sudo apt install gdebi
sudo gdebi ./amdgpu-install_6.3.60304-1_all.deb
```


```bash
sudo amdgpu-install --usecase=hip
```

* [Blender HIP](https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html)
* [AMD Radeon Software Instructions](https://amdgpu-install.readthedocs.io/en/latest/)
