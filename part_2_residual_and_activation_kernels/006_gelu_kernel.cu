#include "../model.cuh"

__global__ void gelu_kernel(
    const float* x,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        float v3 = v * v * v;

        float inner =
            sqrtf(2.0f / M_PI) *
            (v + 0.044715f * v3);

        out[i] =
            0.5f * v * (1.0f + tanhf(inner));
    }
}
