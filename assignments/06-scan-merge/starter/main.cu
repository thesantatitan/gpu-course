#include "cuda_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void block_exclusive_scan_kernel(const float* input,
                                            float* output,
                                            float* block_sums,
                                            int n) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // TODO: Implement an exclusive scan within each block.
    // For the first pass, assume gridDim.x == 1.
    // Then extend by writing block_sums[blockIdx.x] and adding scanned block offsets.
    (void)input;
    (void)output;
    (void)block_sums;
    (void)n;
    (void)scratch;
    (void)tid;
    (void)i;
}

static void cpu_exclusive_scan(const std::vector<float>& input,
                               std::vector<float>& output) {
    float running = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = running;
        running += input[i];
    }
}

int main(int argc, char** argv) {
    require_cuda_device();

    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int block_size = (argc > 2) ? std::atoi(argv[2]) : 1024;
    if (n <= 0 || block_size <= 0 || block_size > 1024) {
        std::fprintf(stderr, "Usage: %s [n] [block_size <= 1024]\n", argv[0]);
        return 2;
    }

    std::vector<float> h_input(n), h_output(n, -999.0f), h_ref(n);
    for (int i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>((i % 5) + 1);
    }
    cpu_exclusive_scan(h_input, h_ref);

    int grid_size = div_up(n, block_size);
    float *d_input = nullptr, *d_output = nullptr, *d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<size_t>(grid_size) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer.tic();
    block_exclusive_scan_kernel<<<grid_size, block_size,
                                  static_cast<size_t>(block_size) * sizeof(float)>>>(
        d_input, d_output, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
    float scan_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float err = max_abs_diff(h_output, h_ref);
    bool ok = err < 1e-4f;
    std::printf(
        "{\"lab\":\"scan\",\"ok\":%s,\"n\":%d,\"block_size\":%d,"
        "\"grid_size\":%d,\"scan_ms\":%.6f,\"max_abs_error\":%.8g}\n",
        ok ? "true" : "false", n, block_size, grid_size, scan_ms, err);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_block_sums));
    return ok ? 0 : 1;
}

