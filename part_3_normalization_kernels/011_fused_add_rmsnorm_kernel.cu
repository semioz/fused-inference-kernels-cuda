#include "../model.cuh"

__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    // selecting the row handled by this block
    const float* x_row = x + blockIdx.x * n;
    const float* residual_row = residual + blockIdx.x * n;
    float* out_row = out + blockIdx.x * n;
    float* residual_out_row = residual_out + blockIdx.x * n;

    // shared scalar used to broadcast inv_rms to the whole block
    __shared__ float inv_rms;

    __shared__ float shared[32];

    // each thread computes part of the sum of squares
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float r = x_row[i] + residual_row[i];

        residual_out_row[i] = r;

        sum_sq += r * r;
    }

    // combine every thread's partial sum
    float total_sum_sq = block_reduce_sum(sum_sq, shared);

    // thread 0 computes 1 / RMS
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(total_sum_sq / n + eps);
    }

    __syncthreads();

    // normalize residual_out and apply learned weight
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = residual_out_row[i] * inv_rms * weight[i];
    }
}
