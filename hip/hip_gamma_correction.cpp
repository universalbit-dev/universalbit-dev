/******************************************************************************
 * Project: Image Gamma Correction Using HIP
 * File: hip_gamma_correction.cpp
 * 
 * Description:
 * This program demonstrates the use of HIP to perform gamma correction on
 * grayscale image pixels using GPU parallel computation.
 * 
 * Key Features:
 * - Implements a custom `Pixel` struct for storing image intensity values.
 * - Utilizes HIP for efficient parallel gamma correction.
 * - Measures kernel execution time using HIP events for performance analysis.
 * 
 * Compilation Instructions:
 * Compile the program using the HIP compiler:
 * 
 *     hipcc hip_gamma_correction.cpp -o hip_gamma_correction
 * 
 * Execution:
 * Run the compiled binary as follows:
 * 
 *     ./hip_gamma_correction
 * 
 * Notes:
 * - The program assumes grayscale image data, but can be extended for RGB.
 * - Optimize GPU utilization by tuning the number of threads and blocks.
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

// Define a struct for a grayscale pixel
struct Pixel {
    float intensity; // Pixel intensity value (0.0 to 1.0)

    __device__ __host__ float applyGamma(float gamma) const {
        return powf(intensity, gamma);
    }
};

// Kernel function to apply gamma correction
__global__ void applyGammaCorrection(Pixel* pixels, float* correctedPixels, int numPixels, float gamma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPixels) {
        correctedPixels[idx] = pixels[idx].applyGamma(gamma);
    }
}

int main() {
    const int numPixels = 1024 * 1024; // Example: 1 megapixel
    const int threadsPerBlock = 256;
    const int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    const float gamma = 2.2f; // Example gamma value

    // Host memory allocation
    Pixel* hostPixels = new Pixel[numPixels];
    float* hostCorrectedPixels = new float[numPixels];

    // Initialize pixels with random intensity values
    for (int i = 0; i < numPixels; ++i) {
        hostPixels[i].intensity = static_cast<float>(rand()) / RAND_MAX; // Random value between 0.0 and 1.0
    }

    // Device memory allocation
    Pixel* devicePixels;
    float* deviceCorrectedPixels;
    hipMalloc(&devicePixels, numPixels * sizeof(Pixel));
    hipMalloc(&deviceCorrectedPixels, numPixels * sizeof(float));

    // Copy data from host to device
    hipMemcpy(devicePixels, hostPixels, numPixels * sizeof(Pixel), hipMemcpyHostToDevice);

    // Create HIP events for timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Record the start event
    hipEventRecord(start, 0);

    // Launch kernel
    applyGammaCorrection<<<blocks, threadsPerBlock>>>(devicePixels, deviceCorrectedPixels, numPixels, gamma);

    // Record the stop event
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    // Copy results back to host
    hipMemcpy(hostCorrectedPixels, deviceCorrectedPixels, numPixels * sizeof(float), hipMemcpyDeviceToHost);

    // Print some corrected pixel values
    for (int i = 0; i < 10; ++i) {
        std::cout << "Original intensity: " << hostPixels[i].intensity
                  << " | Corrected intensity: " << hostCorrectedPixels[i] << std::endl;
    }

    // Free memory
    delete[] hostPixels;
    delete[] hostCorrectedPixels;
    hipFree(devicePixels);
    hipFree(deviceCorrectedPixels);

    // Destroy HIP events
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
