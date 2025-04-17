/******************************************************************************
 * File: hip_social_sparse_data_fft.cpp
 * Description:
 * This file demonstrates the use of Discrete Fourier Transform (DFT) for 
 * large-scale data processing using the rocFFT library. It focuses on GPU-accelerated
 * DFT computation, which is crucial for applications like signal processing, 
 * spectral analysis, and other transformations in large datasets.
 *
 * Implementation:
 * This file uses rocFFT for DFT/FFT computation. The example demonstrates
 * the computation of the DFT for a signal and supports extensions for
 * batched FFT and multidimensional FFT.
 *
 * Usage:
 * The example provided shows how to:
 * - Set up and execute a DFT computation on GPU.
 * - Retrieve the results from GPU to the host.
 *
 * Author: universalbit-dev
 * Date: 2025-04-17
 * License: MIT
 ******************************************************************************/

#include <rocfft.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Function to perform DFT using rocFFT
void performFFT(const std::vector<float>& input, std::vector<float>& output, size_t n) {
    // Create rocFFT plan
    rocfft_plan plan = nullptr;
    rocfft_plan_create(&plan, 
                       rocfft_placement_notinplace, 
                       rocfft_transform_type_real_forward, 
                       rocfft_precision_single, 
                       1, &n, 1);

    // Allocate GPU memory
    float* d_input;
    float* d_output;
    hipMalloc(&d_input, n * sizeof(float));
    hipMalloc(&d_output, n * sizeof(float));

    // Copy data from host to GPU
    hipMemcpy(d_input, input.data(), n * sizeof(float), hipMemcpyHostToDevice);

    // Execute the FFT
    rocfft_execute(plan, (void**)&d_input, (void**)&d_output, nullptr);

    // Copy result back to host
    hipMemcpy(output.data(), d_output, n * sizeof(float), hipMemcpyDeviceToHost);

    // Clean up
    rocfft_plan_destroy(plan);
    hipFree(d_input);
    hipFree(d_output);
}

// Main function to demonstrate FFT
int main() {
    // Example input signal
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t n = input.size();

    // Output vector to store FFT result
    std::vector<float> output(n, 0.0f);

    // Perform FFT
    std::cout << "Performing FFT on input signal..." << std::endl;
    performFFT(input, output, n);

    // Print FFT result
    std::cout << "FFT Result:" << std::endl;
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
