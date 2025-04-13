/******************************************************************************
 * Project: Distance Calculation Using HIP
 * File: calculate_distance_from_origin.cpp
 * 
 * Description:
 * This program calculates the distance of 2D points from the origin using 
 * HIP (Heterogeneous-Compute Interface for Portability). Points are represented 
 * using a struct, and the distance is calculated in parallel on the GPU.
 * 
 * Features:
 * - HIP implementation for GPU parallel computation.
 * - Uses a custom struct for better organization of point data.
 * 
 * How It Works:
 * 1. Points are initialized on the host with random values in the range [-1, 1].
 * 2. The kernel calculates the distance of each point from the origin in parallel.
 * 3. Results are copied back to the host for further processing.
 * 
 * How to Compile:
 * Use the HIP compiler to compile this program:
 * hipcc calculate_distance_from_origin.cpp -o calculate_distance_from_origin
 * 
 * How to Run:
 * ./calculate_distance_from_origin
 * 
 * Example Output:
 * Distance of point (0.5, -0.3) from origin: 0.583095
 * Distance of point (-0.7, 0.7) from origin: 0.989949
 * 
 * Notes:
 * - The program can be extended to handle 3D points or other distance metrics.
 * - Adjust the number of threads and blocks for optimal GPU utilization.
 * 
 * Repository: github.com/universalbit-dev/
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

    // Launch kernel
    calculateDistances<<<blocks, threadsPerBlock>>>(devicePoints, deviceDistances, numPoints);

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

    return 0;
}
