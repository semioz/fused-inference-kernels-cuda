#include "../model.cuh"

__global__ void swiglu_kernel(
    const float* gate,
    const float* up,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    float g = gate[i];
    float silu = g / (1.0f + expf(-g));

    out[i] = silu * up[i];
}
