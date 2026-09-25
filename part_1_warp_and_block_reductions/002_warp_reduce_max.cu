#include "../model.cuh"

__device__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, val, offset);
        val = fmaxf(val, other);
    }

    return val;
}
