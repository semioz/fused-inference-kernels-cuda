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

# Step 9 - rmsnorm_kernel
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

# Step 10 - layernorm_kernel
__global__ void layernorm_kernel(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int n,
    float eps
) {
    const float* x_row = x + blockIdx.x * n;
    float* out_row = out + blockIdx.x * n;

    __shared__ float shared[32];
    __shared__ float mean;
    __shared__ float inv_std;

    // 1. Compute mean
    float sum = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        sum += x_row[i];
    }

    float total_sum = block_reduce_sum(sum, shared);

    if (threadIdx.x == 0) {
        mean = total_sum / n;
    }

    __syncthreads();

    // 2. Compute variance
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float diff = x_row[i] - mean;
        sum_sq += diff * diff;
    }

    float total_sum_sq = block_reduce_sum(sum_sq, shared);

    if (threadIdx.x == 0) {
        float variance = total_sum_sq / n;
        inv_std = rsqrtf(variance + eps);
    }

    __syncthreads();

    // 3. Normalize + scale + bias
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] =
            (x_row[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

# Step 11 - fused_add_rmsnorm_kernel
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

# Step 12 - softmax_row_kernel
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

# Step 13 - causal_softmax_kernel
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

