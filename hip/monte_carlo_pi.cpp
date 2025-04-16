/******************************************************************************
 * Project: Monte Carlo Simulation for Estimating Pi (HIP Version)
 * File: monte_carlo_pi.cpp
 * 
 * Description:
 * This program estimates the value of Pi (π) using the Monte Carlo method. 
 * It leverages GPU parallelism with HIP to distribute the simulation workload 
 * across thousands of threads. Random points are generated within a unit 
 * square using the RocRand library, and the ratio of points inside a 
 * quarter-circle to the total number of points is used to approximate π.
 * 
 * Features:
 * - HIP implementation for GPU parallel computation.
 * - RocRand library integration for generating random numbers on the GPU.
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
 * hipcc monte_carlo_pi.cpp -o monte_carlo_pi -lrocrand
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
 * - RocRand library must be installed to use this program.
 * 
 * Repository: github.com/universalbit-dev/
 * Date: April 2025
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <iostream>

// Kernel function to calculate points inside the circle
__global__ void calculateInsideCircle(int *insideCircle, const float *randomX, const float *randomY, int numPointsPerThread) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    int localCount = 0;

    for (int i = 0; i < numPointsPerThread; ++i) {
        // Get random points (x, y)
        float x = randomX[idx * numPointsPerThread + i];
        float y = randomY[idx * numPointsPerThread + i];

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

    // Allocate memory for random numbers on the device
    float *randomXDevice, *randomYDevice;
    hipMalloc(&randomXDevice, numPoints * sizeof(float));
    hipMalloc(&randomYDevice, numPoints * sizeof(float));

    // Create a RocRand generator
    rocrand_generator generator;
    rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);

    // Seed the generator
    rocrand_set_seed(generator, time(NULL));

    // Generate random numbers for x and y coordinates
    rocrand_generate_uniform(generator, randomXDevice, numPoints);
    rocrand_generate_uniform(generator, randomYDevice, numPoints);

    // Launch the kernel
    calculateInsideCircle<<<blocks, threadsPerBlock>>>(insideCircleDevice, randomXDevice, randomYDevice, numPointsPerThread);

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
    hipFree(randomXDevice);
    hipFree(randomYDevice);

    // Cleanup RocRand generator
    rocrand_destroy_generator(generator);

    return 0;
}
