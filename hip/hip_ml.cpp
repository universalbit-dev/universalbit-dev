/*
 * File: hip_ml.cpp
 * Description: Demonstrates the benefits of GPU processing using PyTorch and TensorFlow frameworks.
 *              This example compares execution times for machine learning inference on CPU and GPU.
 * 
 * Key Features:
 *  - Uses PyTorch C++ API to load and execute a pre-trained model.
 *  - Uses TensorFlow C++ API for basic operations and performance comparisons.
 *  - Measures and outputs execution times for both CPU and GPU processing.
 * 
 * Author: universalbit-dev
 * Date: 2025-04-17
 * Repository: https://github.com/universalbit-dev/universalbit-dev
 *
 * Notes:
 *  - Ensure PyTorch and TensorFlow libraries are properly linked during compilation.
 *  - GPU support requires appropriate CUDA and cuDNN installations.
 *  - Replace `model.pt` with the path to your PyTorch model file.
 *
 */
 
#include <iostream>
#include <chrono>

// Include PyTorch headers
#include <torch/script.h> // PyTorch C++ API

// Include TensorFlow headers
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"

void demonstrate_pytorch() {
    // Load a PyTorch model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("model.pt"); // Replace with your model path
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the PyTorch model\n";
        return;
    }

    // Create a sample input tensor
    torch::Tensor input = torch::rand({1, 3, 224, 224}); // Example: 1 image, 3 channels, 224x224

    // Measure CPU execution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    auto output_cpu = model.forward({input});
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "PyTorch CPU Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count()
              << " ms\n";

    // Measure GPU execution
    input = input.to(torch::kCUDA);
    model.to(torch::kCUDA);
    auto start_gpu = std::chrono::high_resolution_clock::now();
    auto output_gpu = model.forward({input});
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "PyTorch GPU Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()
              << " ms\n";
}

void demonstrate_tensorflow() {
    using namespace tensorflow;
    Scope root = Scope::NewRootScope();

    // Define a simple computation graph (example)
    auto A = ops::Const(root, { {3.f, 2.f}, {1.f, 0.f} });
    auto B = ops::Const(root, { {1.f, 0.f}, {0.f, 1.f} });
    auto matmul = ops::MatMul(root.WithOpName("MatMul"), A, B);

    // Create a session
    ClientSession session(root);

    // Measure CPU execution
    std::vector<Tensor> outputs_cpu;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    session.Run({matmul}, &outputs_cpu);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "TensorFlow CPU Time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count()
              << " ms\n";

    // Note: TensorFlow GPU usage requires additional setup for GPU-specific execution
    // Refer to TensorFlow documentation for enabling GPU in C++ API
}

int main() {
    std::cout << "Demonstrating PyTorch:\n";
    demonstrate_pytorch();

    std::cout << "\nDemonstrating TensorFlow:\n";
    demonstrate_tensorflow();

    return 0;
}
