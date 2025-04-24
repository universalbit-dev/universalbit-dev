__global__ void relu_activation(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// Host code
void apply_relu(float* device_input, float* device_output, int size) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    relu_activation<<<blocksPerGrid, threadsPerBlock>>>(device_input, device_output, size);
    hipDeviceSynchronize();
}

