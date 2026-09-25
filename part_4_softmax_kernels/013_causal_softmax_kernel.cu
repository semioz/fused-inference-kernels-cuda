#include "../model.cuh"

__global__ void causal_softmax_kernel(
    const float* x,
    float* out,
    int rows,
    int cols
) {
    int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    __shared__ float shared[32];
    __shared__ float row_max;
    __shared__ float row_sum;

    // 1. Find max only over causal-valid positions: col <= row
    float local_max = -INFINITY;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i <= row) {
            local_max = fmaxf(local_max, x_row[i]);
        }
    }

    float max_val = block_reduce_max(local_max, shared);

    if (threadIdx.x == 0) {
        row_max = max_val;
    }

    __syncthreads();

    // sum exp(x - max) only over valid positions
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i <= row) {
            local_sum += expf(x_row[i] - row_max);
        }
    }

    float sum_val = block_reduce_sum(local_sum, shared);

    if (threadIdx.x == 0) {
        row_sum = sum_val;
    }

    __syncthreads();

    // softmax for valid positions, zero for future positions
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i <= row) {
            out_row[i] = expf(x_row[i] - row_max) / row_sum;
        } else {
            out_row[i] = 0.0f;
        }
    }
}
