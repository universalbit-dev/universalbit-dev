/******************************************************************************
 * File: hip_social_sparse_data_fft.cpp
 * Description:
 * This file demonstrates the use of Discrete Fourier Transform (DFT) for 
 * large-scale data processing using the rocFFT library. It focuses on GPU-accelerated
 * DFT computation, which is crucial for applications like signal processing, 
 * spectral analysis, and other transformations in large datasets.
 *
 * Implementation:
 * The file uses rocFFT for DFT/FFT computation. Features include:
 * - Single and batched FFT for multiple signals.
 * - Error handling for GPU memory allocation and rocFFT operations.
 * - Profiling for measuring execution time.
 *
 * How to Use:
 * 1. Prepare an input signal as a `std::vector<float>` containing real-valued data.
 * 2. Optionally, specify the signal size or batch size via command-line arguments:
 *    `./hip_social_sparse_data_fft [signal_size] [batch_size]`
 * 3. Compile the program with the following command:
 *    `hipcc hip_social_sparse_data_fft.cpp -lrocfft -o hip_social_sparse_data_fft`
 * 4. Run the compiled program to observe the DFT results for the example input signal.
 *
 * Example Input:
 * Input Signal (per batch): {1.0f, 2.0f, 3.0f, 4.0f}
 * Batch Size: 2
 *
 * Example Output:
 * FFT Execution Time: 1 ms
 * FFT Result (Batch 1): 10 -2 2 -2 
 * FFT Result (Batch 2): 10 -2 2 -2
 *
 * Author: universalbit-dev
 * Date: 2025-04-17
 * License: MIT
 ******************************************************************************/

#include <rocfft.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>

// Function to perform batched FFT using rocFFT
void performBatchedFFT(const std::vector<float>& input, std::vector<float>& output, size_t signal_size, size_t batch_size) {
    rocfft_status status;

    // Create rocFFT plan for batched FFT
    rocfft_plan plan = nullptr;
    size_t lengths[1] = {signal_size};
    status = rocfft_plan_create(&plan, 
                                rocfft_placement_notinplace, 
                                rocfft_transform_type_real_forward, 
                                rocfft_precision_single, 
                                1, lengths, batch_size);
    if (status != rocfft_status_success) {
        std::cerr << "Error: Failed to create rocFFT plan!" << std::endl;
        return;
    }

    // Allocate GPU memory
    float* d_input;
    float* d_output;
    if (hipMalloc(&d_input, signal_size * batch_size * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_output, signal_size * batch_size * sizeof(float)) != hipSuccess) {
        std::cerr << "Error: Failed to allocate GPU memory!" << std::endl;
        rocfft_plan_destroy(plan);
        return;
    }

    // Copy data from host to GPU
    hipMemcpy(d_input, input.data(), signal_size * batch_size * sizeof(float), hipMemcpyHostToDevice);

    // Execute the FFT
    rocfft_execution_info exec_info;
    rocfft_execution_info_create(&exec_info);

    auto start = std::chrono::high_resolution_clock::now();
    rocfft_execute(plan, (void**)&d_input, (void**)&d_output, exec_info);
    auto end = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    hipMemcpy(output.data(), d_output, signal_size * batch_size * sizeof(float), hipMemcpyDeviceToHost);

    // Print timing
    std::cout << "FFT Execution Time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
              << " ms" << std::endl;

    // Clean up
    rocfft_execution_info_destroy(exec_info);
    rocfft_plan_destroy(plan);
    hipFree(d_input);
    hipFree(d_output);
}

int main(int argc, char** argv) {
    // Parse command-line arguments for signal size and batch size
    size_t signal_size = (argc > 1) ? std::stoi(argv[1]) : 4; // Default signal size: 4
    size_t batch_size = (argc > 2) ? std::stoi(argv[2]) : 1; // Default batch size: 1

    // Validate input
    if (signal_size < 1 || batch_size < 1) {
        std::cerr << "Error: Signal size and batch size must be positive integers." << std::endl;
        return 1;
    }

    // Prepare input signal
    std::vector<float> input(signal_size * batch_size, 0.0f);
    for (size_t i = 0; i < signal_size * batch_size; ++i) {
        input[i] = static_cast<float>((i % signal_size) + 1); // Example: {1.0, 2.0, ..., signal_size}
    }

    // Prepare output vector
    std::vector<float> output(signal_size * batch_size, 0.0f);

    // Perform batched FFT
    performBatchedFFT(input, output, signal_size, batch_size);

    // Print FFT results for each batch
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::cout << "FFT Result (Batch " << batch + 1 << "): ";
        for (size_t i = 0; i < signal_size; ++i) {
            std::cout << output[batch * signal_size + i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
