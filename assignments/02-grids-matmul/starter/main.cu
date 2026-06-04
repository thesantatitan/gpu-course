#include "cuda_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE 16

__global__ void matmul_basic_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: If row/col are in bounds, compute C[row, col].
    (void)a;
    (void)b;
    (void)c;
    (void)n;
    (void)row;
    (void)col;
}

__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    // TODO: Load one A tile and one B tile per phase, synchronize,
    // accumulate the dot product, then write C[row, col] if in bounds.
    (void)a;
    (void)b;
    (void)c;
    (void)n;
    (void)row;
    (void)col;
    (void)a_tile;
    (void)b_tile;
}

static void cpu_matmul(const std::vector<float>& a,
                       const std::vector<float>& b,
                       std::vector<float>& c,
                       int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += a[row * n + k] * b[k * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

static void fill_matrix(std::vector<float>& x) {
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(0.01f * static_cast<float>(i % 251)) * 0.5f;
    }
}

int main(int argc, char** argv) {
    require_cuda_device();

    int n = (argc > 1) ? std::atoi(argv[1]) : 128;
    if (n <= 0) {
        std::fprintf(stderr, "Usage: %s [n]\n", argv[0]);
        return 2;
    }

    size_t elems = static_cast<size_t>(n) * n;
    size_t bytes = elems * sizeof(float);
    std::vector<float> h_a(elems), h_b(elems), h_basic(elems, -999.0f),
        h_tiled(elems, -999.0f), h_ref(elems);
    fill_matrix(h_a);
    fill_matrix(h_b);
    cpu_matmul(h_a, h_b, h_ref, n);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(div_up(n, TILE), div_up(n, TILE));

    GpuTimer timer;
    timer.tic();
    matmul_basic_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    float basic_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_basic.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_c, 0, bytes));
    timer.tic();
    matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    float tiled_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    float basic_err = max_abs_diff(h_basic, h_ref);
    float tiled_err = max_abs_diff(h_tiled, h_ref);
    bool basic_ok = basic_err < 1e-3f;
    bool tiled_ok = tiled_err < 1e-3f;

    std::printf(
        "{\"lab\":\"matmul\",\"ok\":%s,\"n\":%d,\"basic_ms\":%.6f,"
        "\"tiled_ms\":%.6f,\"basic_err\":%.8g,\"tiled_err\":%.8g}\n",
        (basic_ok && tiled_ok) ? "true" : "false", n, basic_ms, tiled_ms,
        basic_err, tiled_err);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return (basic_ok && tiled_ok) ? 0 : 1;
}

