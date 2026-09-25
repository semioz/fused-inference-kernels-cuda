#include "../model.cuh"

void run_transformer_ffn(
    const float* x,
    const float* residual,
    const float* norm_weight,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* out,
    int M,
    int hidden_dim,
    int intermediate_dim,
    float eps
) {
    // temp residual stream: r = x + residual
    float* d_residual_out;

    // temporary normalized activations
    float* d_norm;

    size_t bytes = M * hidden_dim * sizeof(float);

    cudaMalloc(&d_residual_out, bytes);
    cudaMalloc(&d_norm, bytes);

    // 1. r = x + residual
    // 2. x_norm = RMSNorm(r)
    rmsnorm_residual_block(
        x,
        residual,
        norm_weight,
        d_norm,
        d_residual_out,
        M,
        hidden_dim,
        eps
    );

    // 3. out = SwiGLU_MLP(x_norm)
    mlp_swiglu_forward(
        d_norm,
        w_gate,
        w_up,
        w_down,
        out,
        M,
        hidden_dim,
        intermediate_dim
    );

    // 4. out = residual_out + out
    int total = M * hidden_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    add_residual_kernel<<<blocks, threads>>>(
        d_residual_out,
        out,
        out,
        total
    );

    cudaDeviceSynchronize();

    cudaFree(d_residual_out);
    cudaFree(d_norm);
}
