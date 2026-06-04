#include "cuda_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void histogram_atomic_kernel(const unsigned int* data,
                                        unsigned int* bins,
                                        int n,
                                        int num_bins) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: If i is in bounds, atomically increment bins[data[i]].
    (void)data;
    (void)bins;
    (void)n;
    (void)num_bins;
    (void)i;
}

__global__ void reduce_sum_kernel(const float* input,
                                  float* block_sums,
                                  int n) {
    extern __shared__ float scratch[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // TODO: Load one value or 0 into scratch, reduce within the block,
    // and have thread 0 write block_sums[blockIdx.x].
    (void)input;
    (void)block_sums;
    (void)n;
    (void)scratch;
    (void)tid;
    (void)i;
}

static void cpu_histogram(const std::vector<unsigned int>& data,
                          std::vector<unsigned int>& bins) {
    for (unsigned int x : data) {
        bins[x] += 1;
    }
}

static double cpu_sum(const std::vector<float>& input) {
    double sum = 0.0;
    for (float x : input) {
        sum += x;
    }
    return sum;
}

int main(int argc, char** argv) {
    require_cuda_device();

    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int num_bins = (argc > 2) ? std::atoi(argv[2]) : 256;
    int block_size = (argc > 3) ? std::atoi(argv[3]) : 256;
    if (n <= 0 || num_bins <= 0 || block_size <= 0) {
        std::fprintf(stderr, "Usage: %s [n] [num_bins] [block_size]\n", argv[0]);
        return 2;
    }

    std::vector<unsigned int> h_data(n), h_bins(num_bins, 0), h_ref_bins(num_bins, 0);
    std::vector<float> h_values(n);
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<unsigned int>((i * 17 + (i >> 3)) % num_bins);
        h_values[i] = 1.0f / static_cast<float>(1 + (i % 13));
    }
    cpu_histogram(h_data, h_ref_bins);
    double ref_sum = cpu_sum(h_values);

    unsigned int *d_data = nullptr, *d_bins = nullptr;
    float *d_values = nullptr, *d_block_sums = nullptr;
    int grid_size = div_up(n, block_size);
    CUDA_CHECK(cudaMalloc(&d_data, static_cast<size_t>(n) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_bins, static_cast<size_t>(num_bins) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_values, static_cast<size_t>(n) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<size_t>(grid_size) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), static_cast<size_t>(n) * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), static_cast<size_t>(n) * sizeof(float),
                          cudaMemcpyHostToDevice));

    GpuTimer timer;
    CUDA_CHECK(cudaMemset(d_bins, 0, static_cast<size_t>(num_bins) * sizeof(unsigned int)));
    timer.tic();
    histogram_atomic_kernel<<<grid_size, block_size>>>(d_data, d_bins, n, num_bins);
    CUDA_CHECK(cudaGetLastError());
    float hist_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_bins.data(), d_bins, static_cast<size_t>(num_bins) * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    bool hist_ok = true;
    for (int i = 0; i < num_bins; ++i) {
        hist_ok = hist_ok && (h_bins[i] == h_ref_bins[i]);
    }

    CUDA_CHECK(cudaMemset(d_block_sums, 0, static_cast<size_t>(grid_size) * sizeof(float)));
    timer.tic();
    reduce_sum_kernel<<<grid_size, block_size, static_cast<size_t>(block_size) * sizeof(float)>>>(
        d_values, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
    float reduce_ms = timer.toc_ms();

    std::vector<float> h_block_sums(grid_size);
    CUDA_CHECK(cudaMemcpy(h_block_sums.data(), d_block_sums,
                          static_cast<size_t>(grid_size) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double gpu_sum = cpu_sum(h_block_sums);
    double sum_err = std::fabs(gpu_sum - ref_sum);
    bool reduce_ok = sum_err < 1e-2;

    std::printf(
        "{\"lab\":\"hist_reduce\",\"ok\":%s,\"n\":%d,\"num_bins\":%d,"
        "\"block_size\":%d,\"hist_ms\":%.6f,\"reduce_ms\":%.6f,"
        "\"hist_ok\":%s,\"reduce_ok\":%s,\"sum_error\":%.8g}\n",
        (hist_ok && reduce_ok) ? "true" : "false", n, num_bins, block_size,
        hist_ms, reduce_ms, hist_ok ? "true" : "false",
        reduce_ok ? "true" : "false", sum_err);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_bins));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_block_sums));
    return (hist_ok && reduce_ok) ? 0 : 1;
}

