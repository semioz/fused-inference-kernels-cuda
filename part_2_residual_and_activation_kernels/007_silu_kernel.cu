#include "../model.cuh"

__global__ void silu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}
