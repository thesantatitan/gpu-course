#include "cuda_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define BLOCK 16
#define MAX_MASK_WIDTH 9

__constant__ float c_mask[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

__global__ void conv2d_basic_kernel(const float* input,
                                    const float* mask,
                                    float* output,
                                    int width,
                                    int height,
                                    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: Compute zero-padded 2D convolution using mask from global memory.
    (void)input;
    (void)mask;
    (void)output;
    (void)width;
    (void)height;
    (void)radius;
    (void)x;
    (void)y;
}

__global__ void conv2d_constant_kernel(const float* input,
                                       float* output,
                                       int width,
                                       int height,
                                       int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: Same result as basic kernel, but read the mask from c_mask.
    (void)input;
    (void)output;
    (void)width;
    (void)height;
    (void)radius;
    (void)x;
    (void)y;
}

static void cpu_conv2d(const std::vector<float>& input,
                       const std::vector<float>& mask,
                       std::vector<float>& output,
                       int width,
                       int height,
                       int radius) {
    int mask_width = 2 * radius + 1;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    int ix = x + dx;
                    int iy = y + dy;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int mask_idx = (dy + radius) * mask_width + (dx + radius);
                        sum += input[iy * width + ix] * mask[mask_idx];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

int main(int argc, char** argv) {
    require_cuda_device();

    int width = (argc > 1) ? std::atoi(argv[1]) : 512;
    int height = (argc > 2) ? std::atoi(argv[2]) : 512;
    int radius = (argc > 3) ? std::atoi(argv[3]) : 1;
    int mask_width = 2 * radius + 1;
    if (width <= 0 || height <= 0 || radius < 0 || mask_width > MAX_MASK_WIDTH) {
        std::fprintf(stderr, "Usage: %s [width] [height] [radius <= 4]\n", argv[0]);
        return 2;
    }

    size_t elems = static_cast<size_t>(width) * height;
    size_t image_bytes = elems * sizeof(float);
    size_t mask_elems = static_cast<size_t>(mask_width) * mask_width;
    size_t mask_bytes = mask_elems * sizeof(float);

    std::vector<float> h_input(elems), h_ref(elems), h_basic(elems, -999.0f),
        h_const(elems, -999.0f), h_mask(mask_elems);
    for (size_t i = 0; i < elems; ++i) {
        h_input[i] = std::sin(0.01f * static_cast<float>(i % 997));
    }
    for (size_t i = 0; i < mask_elems; ++i) {
        h_mask[i] = 1.0f / static_cast<float>(mask_elems);
    }
    cpu_conv2d(h_input, h_mask, h_ref, width, height, radius);

    float *d_input = nullptr, *d_mask = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, image_bytes));
    CUDA_CHECK(cudaMalloc(&d_mask, mask_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, image_bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), image_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), mask_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask.data(), mask_bytes));

    dim3 block(BLOCK, BLOCK);
    dim3 grid(div_up(width, BLOCK), div_up(height, BLOCK));

    GpuTimer timer;
    timer.tic();
    conv2d_basic_kernel<<<grid, block>>>(d_input, d_mask, d_output, width, height, radius);
    CUDA_CHECK(cudaGetLastError());
    float basic_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_basic.data(), d_output, image_bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(d_output, 0, image_bytes));
    timer.tic();
    conv2d_constant_kernel<<<grid, block>>>(d_input, d_output, width, height, radius);
    CUDA_CHECK(cudaGetLastError());
    float const_ms = timer.toc_ms();
    CUDA_CHECK(cudaMemcpy(h_const.data(), d_output, image_bytes, cudaMemcpyDeviceToHost));

    float basic_err = max_abs_diff(h_basic, h_ref);
    float const_err = max_abs_diff(h_const, h_ref);
    bool basic_ok = basic_err < 1e-4f;
    bool const_ok = const_err < 1e-4f;

    std::printf(
        "{\"lab\":\"conv2d\",\"ok\":%s,\"width\":%d,\"height\":%d,\"radius\":%d,"
        "\"basic_ms\":%.6f,\"constant_ms\":%.6f,\"basic_err\":%.8g,"
        "\"constant_err\":%.8g}\n",
        (basic_ok && const_ok) ? "true" : "false", width, height, radius,
        basic_ms, const_ms, basic_err, const_err);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_mask));
    CUDA_CHECK(cudaFree(d_output));
    return (basic_ok && const_ok) ? 0 : 1;
}

