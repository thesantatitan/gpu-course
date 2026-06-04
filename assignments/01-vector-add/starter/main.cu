#include "cuda_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i<n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    require_cuda_device();

    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int block_size = (argc > 2) ? std::atoi(argv[2]) : 256;
    if (n <= 0 || block_size <= 0) {
        std::fprintf(stderr, "Usage: %s [n] [block_size]\n", argv[0]);
        return 2;
    }

    std::vector<float> h_a(n), h_b(n), h_c(n, -999.0f), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = std::sin(0.001f * i);
        h_b[i] = std::cos(0.002f * i);
        h_ref[i] = h_a[i] + h_b[i];
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(block_size);
    dim3 grid(div_up(n, block_size));

    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    GpuTimer timer;
    timer.tic();
    vector_add_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    float gpu_ms = timer.toc_ms();

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    float err = max_abs_diff(h_c, h_ref);
    bool ok = err < 1e-5f;
    std::printf(
        "{\"lab\":\"vector_add\",\"ok\":%s,\"n\":%d,\"block_size\":%d,"
        "\"gpu_ms\":%.6f,\"max_abs_error\":%.8g}\n",
        ok ? "true" : "false", n, block_size, gpu_ms, err);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return ok ? 0 : 1;
}
