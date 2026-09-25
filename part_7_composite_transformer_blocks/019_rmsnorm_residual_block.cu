#include "../model.cuh"

void rmsnorm_residual_block(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int rows,
    int n,
    float eps
) {
    // TODO: launch fused_add_rmsnorm_kernel for the pre-norm residual+RMSNorm block
    int threads = 256;

    fused_add_rmsnorm_kernel<<<rows, threads>>>(
        x,
        residual,
        weight,
        out,
        residual_out,
        n,
        eps
    );
}
