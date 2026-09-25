#include "../model.cuh"

void mlp_swiglu_forward(
    const float* x,
    const float* w_gate,
    const float* w_up,
    const float* w_down,
    float* out,
    int M,
    int hidden_dim,
    int intermediate_dim
) {
    float* d_gate;
    float* d_up;
    float* d_act;

    size_t temp_bytes =
        M * intermediate_dim * sizeof(float);

    cudaMalloc(&d_gate, temp_bytes);
    cudaMalloc(&d_up, temp_bytes);
    cudaMalloc(&d_act, temp_bytes);

    int threads = 256;

    // gate/up outputs have shape [M, intermediate_dim]
    int gate_up_elems = M * intermediate_dim;
    int blocks = (gate_up_elems + threads - 1) / threads;

    // 1. gate = x @ w_gate^T
    // M rows, N=intermediate_dim outputs, K=hidden_dim inputs
    linear_kernel<<<blocks, threads>>>(
        x,
        w_gate,
        nullptr,
        d_gate,
        M,
        intermediate_dim,
        hidden_dim
    );

    // 2. up = x @ w_up^T
    linear_kernel<<<blocks, threads>>>(
        x,
        w_up,
        nullptr,
        d_up,
        M,
        intermediate_dim,
        hidden_dim
    );

    // 3. act = SiLU(gate) * up
    swiglu_kernel<<<blocks, threads>>>(
        d_gate,
        d_up,
        d_act,
        M * intermediate_dim
    );

    // down output has shape [M, hidden_dim]
    int output_elems = M * hidden_dim;
    blocks = (output_elems + threads - 1) / threads;

    // 4. out = act @ w_down^T
    // M rows, N=hidden_dim outputs, K=intermediate_dim inputs
    linear_kernel<<<blocks, threads>>>(
        d_act,
        w_down,
        nullptr,
        out,
        M,
        hidden_dim,
        intermediate_dim
    );

    cudaDeviceSynchronize();

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_act);
}
