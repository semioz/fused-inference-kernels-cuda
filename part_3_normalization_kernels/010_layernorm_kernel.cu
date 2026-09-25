#include "../model.cuh"

__global__ void layernorm_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int n,
    float eps
) {
    const float* x_row = x + blockIdx.x * n;
    float* out_row = out + blockIdx.x * n;

    __shared__ float shared[32];
    __shared__ float mean;
    __shared__ float inv_std;

    // 1. Compute mean
    float sum = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += x_row[i];
    }

    float total_sum = block_reduce_sum(sum, shared);

    if (threadIdx.x == 0) {
        mean = total_sum / n;
    }

    __syncthreads();

    // 2. Compute variance
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float diff = x_row[i] - mean;
        sum_sq += diff * diff;
    }

    float total_sum_sq = block_reduce_sum(sum_sq, shared);

    if (threadIdx.x == 0) {
        float variance = total_sum_sq / n;
        inv_std = rsqrtf(variance + eps);
    }

    __syncthreads();

    // 3. Normalize + scale + bias
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] =
            (x_row[i] - mean) * inv_std * weight[i] + bias[i];
    }
}
