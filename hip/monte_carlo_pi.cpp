/******************************************************************************
 * Project: Monte Carlo Simulation for Estimating Pi (HIP Version)
 * File: monte_carlo_pi.cpp
 * 
 * Description:
 * This program estimates the value of Pi (π) using the Monte Carlo method. 
 * It leverages GPU parallelism with HIP to distribute the simulation workload 
 * across thousands of threads. Random points are generated within a unit 
 * square, and the ratio of points inside a quarter-circle to the total number 
 * of points is used to approximate π.
 * 
 * Features:
 * - HIP implementation for GPU parallel computation.
 * - Scalable to a large number of points for higher accuracy.
 * 
 * How It Works:
 * 1. Points are randomly generated in a 2D Cartesian coordinate system within 
 *    the range [-1, 1] for both x and y.
 * 2. Each thread calculates how many points fall within a unit circle 
 *    (x² + y² ≤ 1).
 * 3. The estimated value of π is calculated as 4 times the ratio of points 
 *    inside the circle to the total number of points.
 * 
 * How to Compile:
 * Use the HIP compiler to compile this program:
 * hipcc monte_carlo_pi_hip.cpp -o monte_carlo_pi_hip
 * 
 * How to Run:
 * ./monte_carlo_pi
 * 
 * Example Output:
 * Estimated value of Pi: 3.14159
 * 
 * Notes:
 * - The accuracy of the result improves as the number of points increases.
 * - Adjust the number of threads and blocks for optimal GPU utilization.
 * 
 * Repository: github.com/universalbit-dev/
 * Date: April 2025
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <iostream>
#include <random>

// Kernel function to calculate points inside the circle
__global__ void calculateInsideCircle(int *insideCircle, int numPointsPerThread, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    int localCount = 0;

    // Initialize random number generator
    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    for (int i = 0; i < numPointsPerThread; ++i) {
        // Generate random points (x, y) in the range [-1, 1]
        float x = curand_uniform(&state) * 2.0f - 1.0f;
        float y = curand_uniform(&state) * 2.0f - 1.0f;

        // Check if the point is inside the circle
        if (x * x + y * y <= 1.0f) {
            ++localCount;
        }
    }

    // Store the count of points inside the circle
    insideCircle[idx] = localCount;
}

int main() {
    const int numPoints = 1000000; // Total number of points
    const int threadsPerBlock = 256; // Number of threads per block
    const int blocks = 256; // Number of blocks
    const int totalThreads = threadsPerBlock * blocks;
    const int numPointsPerThread = numPoints / totalThreads;

    // Allocate memory for results on host and device
    int *insideCircleHost = new int[totalThreads];
    int *insideCircleDevice;
    hipMalloc(&insideCircleDevice, totalThreads * sizeof(int));

    // Launch the kernel
    calculateInsideCircle<<<blocks, threadsPerBlock>>>(insideCircleDevice, numPointsPerThread, time(NULL));

    // Copy results back to host
    hipMemcpy(insideCircleHost, insideCircleDevice, totalThreads * sizeof(int), hipMemcpyDeviceToHost);

    // Calculate the total number of points inside the circle
    int totalInsideCircle = 0;
    for (int i = 0; i < totalThreads; ++i) {
        totalInsideCircle += insideCircleHost[i];
    }

    // Estimate the value of Pi
    double piEstimate = 4.0 * totalInsideCircle / numPoints;
    std::cout << "Estimated value of Pi: " << piEstimate << std::endl;

    // Free memory
    delete[] insideCircleHost;
    hipFree(insideCircleDevice);

    return 0;
}
