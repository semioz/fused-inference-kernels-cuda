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

__global__ void linear_tiled_kernel(const float* x, const float* weight,
                                    const float* bias, float* out,
                                    int M, int N, int K) {
    constexpr int TILE = 16;
    __shared__ float a[TILE][TILE];
    __shared__ float b[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + threadIdx.x;
        a[threadIdx.y][threadIdx.x] = (row < M && k < K) ? x[row * K + k] : 0.0f;
        b[threadIdx.y][threadIdx.x] = (blockIdx.x * TILE + threadIdx.y < N && k < K)
            ? weight[(blockIdx.x * TILE + threadIdx.y) * K + k] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE; ++i) {
            sum += a[threadIdx.y][i] * b[threadIdx.x][i];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        out[row * N + col] = sum + (bias ? bias[col] : 0.0f);
    }
}
