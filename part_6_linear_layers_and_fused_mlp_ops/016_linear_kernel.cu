#include "../model.cuh"

__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // TODO: compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = M * N;

    if (idx >= total) {
        return;
    }

    // which input row
    int m = idx / N;

    // which output feature
    int n = idx % N;

    // accumulator for dot prod
    float sum = 0.0f;

    // dot prod
    for (int k = 0; k < K; k++) {
        sum += x[m * K + k] * weight[n * K + k];
    }

    // optional bias
    if (bias != nullptr) {
        sum += bias[n];
    }

    // store outputs
    out[idx] = sum;
}
