"""
Fused LLM Inference Kernels in CUDA

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(mask, val, offset);
    }

    return val;
}

# Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(mask, val, offset);
        val = fmaxf(val, other);
    }

    return val;
}

# Step 3 - block_reduce_sum
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

# Step 4 - block_reduce_max
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

# Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual, float* out, int n) {
  // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (; i < n; i += blockDim.x * gridDim.x) {
    out[i] = x[i] + residual[i];
  }
}

# Step 6 - gelu_kernel
__global__ void gelu_kernel(
    const float* x,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        float v3 = v * v * v;

        float inner =
            sqrtf(2.0f / M_PI) *
            (v + 0.044715f * v3);

        out[i] =
            0.5f * v * (1.0f + tanhf(inner));
    }
}

# Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        float v = x[i];
        out[i] = v / (1.0f + expf(-v));
    }
}

# Step 8 - swiglu_kernel
__global__ void swiglu_kernel(
    const float* gate,
    const float* up,
    float* out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) {
        return;
    }

    float g = gate[i];
    float silu = g / (1.0f + expf(-g));

    out[i] = silu * up[i];
}

# Step 9 - rmsnorm_kernel (not yet solved)
# TODO: implement

# Step 10 - layernorm_kernel (not yet solved)
# TODO: implement

# Step 11 - fused_add_rmsnorm_kernel (not yet solved)
# TODO: implement

# Step 12 - softmax_row_kernel (not yet solved)
# TODO: implement

# Step 13 - causal_softmax_kernel (not yet solved)
# TODO: implement

# Step 14 - embedding_lookup_kernel (not yet solved)
# TODO: implement

# Step 15 - rope_kernel (not yet solved)
# TODO: implement

# Step 16 - linear_kernel (not yet solved)
# TODO: implement

# Step 17 - fused_linear_bias_gelu_kernel (not yet solved)
# TODO: implement

# Step 18 - mlp_swiglu_forward (not yet solved)
# TODO: implement

# Step 19 - rmsnorm_residual_block (not yet solved)
# TODO: implement

# Step 20 - run_transformer_ffn (not yet solved)
# TODO: implement

