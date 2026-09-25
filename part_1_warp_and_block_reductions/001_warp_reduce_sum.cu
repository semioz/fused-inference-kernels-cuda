#include "../model.cuh"

__device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(mask, val, offset);
    }

    return val;
}
