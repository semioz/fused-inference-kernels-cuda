#include "../model.cuh"

__device__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 1. Each warp sums its own 32 values
    float warp_max = warp_reduce_max(val);

    // 2. Only the first thread of each warp writes the result to shared memory
    // lane = the thread's index within the warp.
    if (lane == 0) {
        shared[warp_id] = warp_max;
    }

    // so all warps finish writing before any read.
    __syncthreads();

    /// # of warps in block
    int num_warps = (blockDim.x + 31) / 32;

    // 3. The first warp sums all warp results
    if (warp_id == 0) {
        float x = (lane < num_warps) ? shared[lane] : -INFINITY;
        float total = warp_reduce_max(x);

        if (lane == 0) {
            return total;
        }
    }

    return -INFINITY;
}
