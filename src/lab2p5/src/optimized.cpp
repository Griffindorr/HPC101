#include "gemm.h"

#include <riscv_vector.h>

void optimized_gemm(uint8_t* A, int8_t* B, int32_t* C, int m, int n, int k) {
    // Start your RVV journey here!
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int kk = 0; kk < k * 4; kk++) {
                C[i * n + j] += A[i * k * 4 + kk] * B[j * k * 4 + kk];
            }
        }
    }
}
