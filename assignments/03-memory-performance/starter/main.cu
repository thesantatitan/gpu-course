#include "cuda_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE 32

__global__ void transpose_naive_kernel(const float* input,
                                       float* output,
                                       int width,
                                       int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: If x/y are in bounds, write output[x * height + y] = input[y * width + x].
    (void)input;
    (void)output;
    (void)width;
    (void)height;
    (void)x;
    (void)y;
}

__global__ void transpose_tiled_kernel(const float* input,
                                       float* output,
                                       int width,
                                       int height) {
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // TODO: Load input into shared memory, synchronize, then write the transposed tile.
    (void)input;
    (void)output;
    (void)width;
    (void)height;
    (void)x;
    (void)y;
    (void)tile;
}

static void cpu_transpose(const std::vector<float>& input,
                          std::vector<float>& output,
                          int width,
                          int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            output[x * height + y] = input[y * width + x];
        }
    }
}

int main(int argc, char** argv) {
    require_cuda_device();

    int width = (argc > 1) ? std::atoi(argv[1]) : 1024;
    int height = (argc > 2) ? std::atoi(argv[2]) : 1024;
    if (width <= 0 || height <= 0) {
        std::fprintf(stderr, "Usage: %s [width] [height]\n", argv[0]);
        return 2;
    }

    size_t elems = static_cast<size_t>(width) * height;
    size_t bytes = elems * sizeof(float);
    std::vector<float> h_input(elems), h_ref(elems), h_naive(elems, -999.0f),
        h_tiled(elems, -999.0f);
    for (size_t i = 0; i < elems; ++i) {
        h_input[i] = std::sin(0.003f * static_cast<float>(i));
    }
    cpu_transpose(h_input, h_ref, width, height);

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(div_up(width, TILE), div_up(height, TILE));

    transpose_naive_kernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    GpuTimer timer;
    timer.tic();
    transpose_naive_kernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    float naive_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_naive.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_output, 0, bytes));
    timer.tic();
    transpose_tiled_kernel<<<grid, block>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());
    float tiled_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    float naive_err = max_abs_diff(h_naive, h_ref);
    float tiled_err = max_abs_diff(h_tiled, h_ref);
    bool naive_ok = naive_err < 1e-6f;
    bool tiled_ok = tiled_err < 1e-6f;
    double gb = static_cast<double>(2 * bytes) / 1.0e9;

    std::printf(
        "{\"lab\":\"transpose\",\"ok\":%s,\"width\":%d,\"height\":%d,"
        "\"naive_ms\":%.6f,\"tiled_ms\":%.6f,\"naive_gbps\":%.3f,"
        "\"tiled_gbps\":%.3f,\"naive_err\":%.8g,\"tiled_err\":%.8g}\n",
        (naive_ok && tiled_ok) ? "true" : "false", width, height, naive_ms,
        tiled_ms, gb / (naive_ms / 1000.0), gb / (tiled_ms / 1000.0),
        naive_err, tiled_err);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return (naive_ok && tiled_ok) ? 0 : 1;
}
