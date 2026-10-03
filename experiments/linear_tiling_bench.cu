#include "../model.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::exit(1); \
    } \
} while (0)

void launch(bool tiled, const float* x, const float* weight, const float* bias,
            float* out, int M, int N, int K) {
    if (tiled) {
        linear_tiled_kernel<<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(
            x, weight, bias, out, M, N, K);
    } else {
        linear_kernel<<<(M * N + 255) / 256, 256>>>(x, weight, bias, out, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
}

float time_ms(bool tiled, const float* x, const float* weight, const float* bias,
              float* out, int M, int N, int K) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < 2; ++i) launch(tiled, x, weight, bias, out, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    for (int trial = 0; trial < 3; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < 10; ++i) launch(tiled, x, weight, bias, out, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        times.push_back(elapsed / 10);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(times.begin(), times.end());
    return times[1];
}

void run_case(int M, int N, int K, bool use_bias) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<float> x(static_cast<size_t>(M) * K), weight(static_cast<size_t>(N) * K);
    std::vector<float> bias(N), baseline(static_cast<size_t>(M) * N), tiled(baseline.size());
    for (float& v : x) v = dist(rng);
    for (float& v : weight) v = dist(rng);
    for (float& v : bias) v = dist(rng);

    float *dx, *dw, *db, *d_baseline, *d_tiled;
    CUDA_CHECK(cudaMalloc(&dx, x.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dw, weight.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, bias.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_baseline, baseline.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tiled, tiled.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dw, weight.data(), weight.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    const float* device_bias = use_bias ? db : nullptr;

    launch(false, dx, dw, device_bias, d_baseline, M, N, K);
    launch(true, dx, dw, device_bias, d_tiled, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(baseline.data(), d_baseline, baseline.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tiled.data(), d_tiled, tiled.size() * sizeof(float), cudaMemcpyDeviceToHost));

    float max_diff = 0;
    for (size_t i = 0; i < tiled.size(); ++i) {
        if (!std::isfinite(tiled[i]) || std::abs(tiled[i] - baseline[i]) > 2e-3f + 1e-3f * std::abs(baseline[i])) {
            std::fprintf(stderr, "Mismatch M=%d N=%d K=%d at %zu: baseline=%f tiled=%f\n",
                         M, N, K, i, baseline[i], tiled[i]);
            std::exit(1);
        }
        max_diff = std::max(max_diff, std::abs(tiled[i] - baseline[i]));
    }
    for (int sample = 0; sample < 16; ++sample) {
        int m = (static_cast<long long>(sample) * 7919) % M;
        int n = (static_cast<long long>(sample) * 9973) % N;
        double expected = use_bias ? bias[n] : 0;
        for (int k = 0; k < K; ++k) expected += static_cast<double>(x[m * K + k]) * weight[n * K + k];
        float actual = tiled[static_cast<size_t>(m) * N + n];
        if (std::abs(actual - expected) > 2e-3 + 1e-3 * std::abs(expected)) {
            std::fprintf(stderr, "CPU reference mismatch at (%d,%d): expected=%f actual=%f\n", m, n, expected, actual);
            std::exit(1);
        }
    }
    std::printf("M=%d N=%d K=%d bias=%d: verified, max |difference|=%g\n", M, N, K, use_bias, max_diff);
    if (M >= 256) {
        float old_ms = time_ms(false, dx, dw, device_bias, d_baseline, M, N, K);
        float new_ms = time_ms(true, dx, dw, device_bias, d_tiled, M, N, K);
        double flops = 2.0 * M * N * K;
        std::printf("  naive %.3f ms (%.1f GFLOP/s), tiled %.3f ms (%.1f GFLOP/s), speedup %.2fx\n",
                    old_ms, flops / (old_ms * 1e6), new_ms, flops / (new_ms * 1e6), old_ms / new_ms);
    }
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(d_baseline));
    CUDA_CHECK(cudaFree(d_tiled));
}

int main() {
    run_case(17, 23, 19, false);
    run_case(17, 23, 19, true);
    run_case(256, 1024, 1024, true);
    run_case(4096, 1024, 1024, true);
}
