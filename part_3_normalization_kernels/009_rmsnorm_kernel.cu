#include "../model.cuh"

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    const float* x_row = x + blockIdx.x * n;
    float* out_row = out + blockIdx.x * n;

    __shared__ float shared[32];

    float sum_sq = 0.0f;

    // each thread computes a partial sum of squares
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum_sq += x_row[i] * x_row[i];
    }

    // sum partial results across whole block
    float total_sum_sq = block_reduce_sum(sum_sq, shared);

    // thread 0 computes inverse RMS and broadcasts it
    if (threadIdx.x == 0) {
        shared[0] = rsqrtf(total_sum_sq / n + eps);
    }

    __syncthreads();

    float inv_rms = shared[0];

    // normalize and apply learned weight
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * inv_rms * weight[i];
    }
}
