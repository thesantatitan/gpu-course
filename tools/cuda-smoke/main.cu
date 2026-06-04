#include "cuda_utils.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void add_one_kernel(const float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[i] + 1.0f;
    }
}

int main(int argc, char** argv) {
    require_cuda_device();

    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int block_size = (argc > 2) ? std::atoi(argv[2]) : 256;
    if (n <= 0 || block_size <= 0) {
        std::fprintf(stderr, "Usage: %s [n] [block_size]\n", argv[0]);
        return 2;
    }

    std::vector<float> h_input(n), h_output(n, -999.0f), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(i % 17);
        h_ref[i] = h_input[i] + 1.0f;
    }

    float *d_input = nullptr, *d_output = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(block_size);
    dim3 grid(div_up(n, block_size));

    GpuTimer timer;
    timer.tic();
    add_one_kernel<<<grid, block>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    float gpu_ms = timer.toc_ms();

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    float err = max_abs_diff(h_output, h_ref);
    bool ok = err < 1e-6f;
    std::printf(
        "{\"lab\":\"cuda_smoke\",\"ok\":%s,\"n\":%d,\"block_size\":%d,"
        "\"gpu_ms\":%.6f,\"max_abs_error\":%.8g}\n",
        ok ? "true" : "false", n, block_size, gpu_ms, err);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return ok ? 0 : 1;
}

