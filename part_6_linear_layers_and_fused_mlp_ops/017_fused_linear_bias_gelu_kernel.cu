#include "../model.cuh"

__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
    // TODO: fuse matmul, bias add, and GELU tanh approx into one kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= M * N) {
        return;
    }

    int m = idx / N;
    int n = idx % N;

    // linear projection
    float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        sum += x[m*K + k] * weight[n*K + k];
    }

    sum += bias[n];

    // gelu
    float v3 = sum * sum * sum;
    float inner  = 0.79788456f * (sum + 0.044715f * v3);
    float gelu = 0.5f * sum * (1.0f + tanhf(inner));

    out[idx] = gelu;
}
