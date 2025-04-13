/******************************************************************************
 * Project: Distance Calculation Using HIP
 * File: calculate_distance_from_origin.cpp
 * 
 * Description:
 * This program demonstrates the use of HIP (Heterogeneous-Compute Interface 
 * for Portability) to calculate the distance of 2D points from the origin 
 * using GPU parallel computation.
 * 
 * Key Features:
 * - Implements a custom `Point` struct for 2D point representation.
 * - Utilizes HIP for efficient parallel distance computation on the GPU.
 * - Measures kernel execution time using HIP events for performance analysis.
 * 
 * Workflow:
 * 1. Points are initialized on the host with random values in the range [-1, 1].
 * 2. Data is transferred to the GPU for parallel distance computation.
 * 3. Kernel execution time is measured and results are copied back to the host.
 * 4. The program prints example outputs and cleans up allocated resources.
 * 
 * Compilation Instructions:
 * Compile the program using the HIP compiler:
 * 
 *     hipcc calculate_distance_from_origin.cpp -o calculate_distance_from_origin
 * 
 * Execution:
 * Run the compiled binary as follows:
 * 
 *     ./calculate_distance_from_origin
 * 
 * Example Output:
 * Distance of point (0.5, -0.3) from origin: 0.583095
 * Distance of point (-0.7, 0.7) from origin: 0.989949
 * 
 * Notes:
 * - The program can be extended to handle 3D points or other distance metrics.
 * - Optimize GPU utilization by tuning the number of threads and blocks.
 * - HIP event profiling is used to measure kernel execution performance.
 * 
 * Repository: https://github.com/universalbit-dev/universalbit-dev
 * 
 * Date: April 2025
 ******************************************************************************/

#include <hip/hip_runtime.h>
#include <iostream>
#include <cmath>

// Define a struct for a 2D point
struct Point {
    float x;
    float y;

    __device__ __host__ float distanceFromOrigin() const {
        return sqrtf(x * x + y * y);
    }
};

// Kernel function to calculate distances
__global__ void calculateDistances(Point *points, float *distances, int numPoints) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        distances[idx] = points[idx].distanceFromOrigin();
    }
}

int main() {
    const int numPoints = 1000;
    const int threadsPerBlock = 256;
    const int blocks = (numPoints + threadsPerBlock - 1) / threadsPerBlock;

    // Host memory allocation
    Point *hostPoints = new Point[numPoints];
    float *hostDistances = new float[numPoints];

    // Initialize points with random values
    for (int i = 0; i < numPoints; ++i) {
        hostPoints[i] = {static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f,
                         static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f};
    }

    // Device memory allocation
    Point *devicePoints;
    float *deviceDistances;
    hipMalloc(&devicePoints, numPoints * sizeof(Point));
    hipMalloc(&deviceDistances, numPoints * sizeof(float));

    // Copy data from host to device
    hipMemcpy(devicePoints, hostPoints, numPoints * sizeof(Point), hipMemcpyHostToDevice);

    // Create HIP events for timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Record the start event
    hipEventRecord(start, 0);

    // Launch kernel
    calculateDistances<<<blocks, threadsPerBlock>>>(devicePoints, deviceDistances, numPoints);

    // Record the stop event
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    hipEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

    // Copy results back to host
    hipMemcpy(hostDistances, deviceDistances, numPoints * sizeof(float), hipMemcpyDeviceToHost);

    // Print some distances
    for (int i = 0; i < 10; ++i) {
        std::cout << "Distance of point (" << hostPoints[i].x << ", " << hostPoints[i].y
                  << ") from origin: " << hostDistances[i] << std::endl;
    }

    // Free memory
    delete[] hostPoints;
    delete[] hostDistances;
    hipFree(devicePoints);
    hipFree(deviceDistances);

    // Destroy HIP events
    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
