#include <iostream>
#include <unistd.h>
#include <string.h>
#include <chrono>

#include "utils.h"
#include "gemm.h"

constexpr int WARMUP = 1e3;
constexpr int ITER = 1e5;

int main() {
    uint8_t* A = new uint8_t[M * K * 4];
    int8_t* B  = new int8_t[N * K * 4];

    int32_t* C     = new int32_t[M * N];
    int32_t* C_ref = new int32_t[M * N];

    init_buffer(A, M, K * 4);
    init_buffer(B, N, K * 4);

    // Warm up
    std::cout << "Warming up..." << std::endl;
    for (int o = 0; o < WARMUP; o ++) {
        memset(C_ref, 0, sizeof(uint32_t) * M * N);
        naive_gemm(A, B, C_ref, M, N, K);
    }

    // naive_gemm
    std::cout << "Running naive_gemm..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();

    for (int o = 0; o < ITER; o ++) {
        memset(C_ref, 0, sizeof(uint32_t) * M * N);
        naive_gemm(A, B, C_ref, M, N, K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // optimized_gemm
    std::cout << "Running optimized_gemm..." << std::endl;
    start = std::chrono::high_resolution_clock::now();

    for (int o = 0; o < ITER; o ++) {
        memset(C, 0, sizeof(uint32_t) * M * N);
        optimized_gemm(A, B, C, M, N, K);
    }

    end = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // verify the ans
    check_result(C, C_ref, M, N);

    std::cout << "Baseline: " << duration1.count()  << " ms" << std::endl;
    std::cout << "Optimized: " << duration2.count() << " ms" << std::endl;
    std::cout << "Speedup: " << (double)duration1.count() / duration2.count() << "x" << std::endl;
    
    return 0;
}