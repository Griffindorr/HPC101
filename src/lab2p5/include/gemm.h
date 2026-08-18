#ifndef GEMM_H
#define GEMM_H

#include <cstdint>

#define M 12
#define K 32
#define N 32

void naive_gemm(uint8_t* A, int8_t* B, int32_t* C, int m, int n, int k);
void optimized_gemm(uint8_t* A, int8_t* B, int32_t* C, int m, int n, int k);

#endif