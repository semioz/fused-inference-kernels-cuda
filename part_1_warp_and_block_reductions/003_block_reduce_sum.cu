#include "../model.cuh"

__device__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 1. Each warp sums its own 32 values
    float warp_sum = warp_reduce_sum(val);

    // 2. Only the first thread of each warp writes the result to shared memory
    // lane = the thread's index within the warp.
    if (lane == 0) {
        shared[warp_id] = warp_sum;
    }

    // so all warps finish writing before any read.
    __syncthreads();

    /// # of warps in block
    int num_warps = (blockDim.x + 31) / 32;

    // 3. The first warp sums all warp results
    if (warp_id == 0) {
        /*
        shared[0] = sum of warp 0
        shared[1] = sum of warp 1
        shared[7] = sum of warp 7
        */
        float x = (lane < num_warps) ? shared[lane] : 0.0f;
        float total = warp_reduce_sum(x);

        if (lane == 0) {
            return total;
        }
    }

    return 0.0f;
}
