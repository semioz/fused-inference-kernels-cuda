#include "../model.cuh"

__global__ void add_residual_kernel(const float* x, const float* residual, float* out, int n) {
  // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  for (; i < n; i += blockDim.x * gridDim.x) {
    out[i] = x[i] + residual[i];
  }
}
