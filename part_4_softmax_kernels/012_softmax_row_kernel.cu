#include "../model.cuh"

__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;

    if (row >= rows) {
        return;
    }

    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    __shared__ float shared[32];
    __shared__ float row_max;
    __shared__ float row_sum;

    // find the maximum value in this row
    float local_max = -INFINITY;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x_row[i]);
    }

    float max_val = block_reduce_max(local_max, shared);

    // thread 0 broadcasts the row max
    if (threadIdx.x == 0) {
        row_max = max_val;
    }

    __syncthreads();

    // compute exp(x-max) and sum them
    float local_sum = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_sum += expf(x_row[i] - row_max);
    }

    float sum_val = block_reduce_sum(local_sum, shared);

    if (threadIdx.x == 0) {
        row_sum = sum_val;
    }

    __syncthreads();

    // normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}
