#include "../model.cuh"

__global__ void rope_kernel(
    float* q,
    float* k,
    const float* cos_table,
    const float* sin_table,
    int seq_len,
    int n_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // RoPE rotates dimensions in pairs: (0,1), (2,3), ...
    int half = head_dim / 2;

    // total number of pairs across all tokens and heads
    int total = seq_len * n_heads * half;

    if (idx >= total) {
        return;
    }

    // which pair inside the head?
    int pair_i = idx % half;

    // remove the pair dimension
    int tmp = idx / half;

    // which attention head?
    int h = tmp % n_heads;

    // which token position?
    int t = tmp / n_heads;

    // start of q[t, h, :]
    int base = (t * n_heads + h) * head_dim;

    // indices of the pair being rotated
    int even = base + 2 * pair_i;
    int odd  = even + 1;

    // sin/cos tables are [seq_len, half]
    int table_idx = t * half + pair_i;

    // read position-dependent rotation values
    float c = cos_table[table_idx];
    float s = sin_table[table_idx];

    // load Q pair before overwriting it
    float q0 = q[even];
    float q1 = q[odd];

    // rotate Q
    q[even] = q0 * c - q1 * s;
    q[odd]  = q0 * s + q1 * c;

    // load K pair
    float k0 = k[even];
    float k1 = k[odd];

    // rotate K using the same sin/cos
    k[even] = k0 * c - k1 * s;
    k[odd]  = k0 * s + k1 * c;
}
