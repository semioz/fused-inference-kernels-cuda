#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

__device__ float warp_reduce_sum(float val);
__device__ float warp_reduce_max(float val);
__device__ float block_reduce_sum(float val, float* shared);
__device__ float block_reduce_max(float val, float* shared);

__global__ void add_residual_kernel(const float* x, const float* residual, float* out, int n);
__global__ void gelu_kernel(const float* x, float* out, int n);
__global__ void silu_kernel(const float* x, float* out, int n);
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n);

__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps);
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps);
__global__ void fused_add_rmsnorm_kernel(const float* x, const float* residual, const float* weight, float* out, float* residual_out, int n, float eps);

__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols);
__global__ void causal_softmax_kernel(const float* x, float* out, int rows, int cols);

__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim);
__global__ void rope_kernel(float* q, float* k, const float* cos_table, const float* sin_table, int seq_len, int n_heads, int head_dim);

__global__ void linear_kernel(const float* x, const float* weight, const float* bias, float* out, int M, int N, int K);
__global__ void fused_linear_bias_gelu_kernel(const float* x, const float* weight, const float* bias, float* out, int M, int N, int K);
void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up, const float* w_down, float* out, int M, int hidden_dim, int intermediate_dim);

void rmsnorm_residual_block(const float* x, const float* residual, const float* weight, float* out, float* residual_out, int rows, int n, float eps);
void run_transformer_ffn(const float* x, const float* residual, const float* norm_weight, const float* w_gate, const float* w_up, const float* w_down, float* out, int M, int hidden_dim, int intermediate_dim, float eps);
