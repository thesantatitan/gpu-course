#include "cuda_utils.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define TILE 16

__global__ void matmul_basic_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float ans = 0.0f;
    if(row<n && col<n){
        for(int i=0;i<n;i++){
            ans += a[row*n + i]*b[i*n + col];
        }
        c[row*n + col] = ans;
    }

}

__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if(row<n && col<n){
        c[row*n + col] = 0;
        float ans = 0.0f;
        for(int ph=0;ph<(n/TILE);ph++){
            a_tile[threadIdx.y][threadIdx.x] = a[row*n + (ph*TILE + threadIdx.x)];
            b_tile[threadIdx.y][threadIdx.x] = b[(ph*TILE + threadIdx.y)*n + col];
            __syncthreads();
            
            for(int i=0;i<TILE;i++){
                ans += a_tile[threadIdx.y][i]*b_tile[i][threadIdx.x];
            }
            
            __syncthreads();
        }
        c[row*n + col] = ans;
    }
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

static float cpu_matmul_element(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int row,
                                int col,
                                int n) {
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum += a[row * n + k] * b[k * n + col];
    }
    return sum;
}

static float sampled_matmul_error(const std::vector<float>& a,
                                  const std::vector<float>& b,
                                  const std::vector<float>& candidate,
                                  int n,
                                  int samples) {
    float worst = 0.0f;
    for (int s = 0; s < samples; ++s) {
        int row = 0;
        int col = 0;
        if (s == 1) {
            col = n - 1;
        } else if (s == 2) {
            row = n - 1;
        } else if (s == 3) {
            row = n - 1;
            col = n - 1;
        } else if (s >= 4) {
            row = (s * 131 + 17) % n;
            col = (s * 197 + 31) % n;
        }

        float ref = cpu_matmul_element(a, b, row, col, n);
        float diff = std::fabs(candidate[row * n + col] - ref);
        worst = std::max(worst, diff);
    }
    return worst;
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
        h_tiled(elems, -999.0f);
    fill_matrix(h_a);
    fill_matrix(h_b);

    bool full_check = n <= 512;
    int check_samples = full_check ? n * n : 128;
    std::vector<float> h_ref;
    if (full_check) {
        h_ref.resize(elems);
        cpu_matmul(h_a, h_b, h_ref, n);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid(div_up(n, TILE), div_up(n, TILE));

    matmul_basic_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    matmul_tiled_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

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

    float basic_err = full_check
        ? max_abs_diff(h_basic, h_ref)
        : sampled_matmul_error(h_a, h_b, h_basic, n, check_samples);
    float tiled_err = full_check
        ? max_abs_diff(h_tiled, h_ref)
        : sampled_matmul_error(h_a, h_b, h_tiled, n, check_samples);
    bool basic_ok = basic_err < 1e-3f;
    bool tiled_ok = tiled_err < 1e-3f;
    double ops = 2.0 * static_cast<double>(n) * n * n;

    std::printf(
        "{\"lab\":\"matmul\",\"ok\":%s,\"n\":%d,\"check_mode\":\"%s\","
        "\"check_samples\":%d,\"basic_ms\":%.6f,\"tiled_ms\":%.6f,"
        "\"basic_gflops\":%.3f,\"tiled_gflops\":%.3f,"
        "\"basic_err\":%.8g,\"tiled_err\":%.8g}\n",
        (basic_ok && tiled_ok) ? "true" : "false", n,
        full_check ? "full" : "sampled", check_samples, basic_ms, tiled_ms,
        ops / (basic_ms / 1000.0) / 1.0e9,
        ops / (tiled_ms / 1000.0) / 1.0e9, basic_err, tiled_err);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return (basic_ok && tiled_ok) ? 0 : 1;
}
