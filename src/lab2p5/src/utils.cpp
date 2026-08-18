#include <iostream>
#include <unistd.h>

#include "utils.h"

void init_buffer(uint8_t* buffer, int rows, int cols) {
    srand(time(0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            buffer[i * cols + j] = rand() % 256;
        }
    }
}

void init_buffer(int8_t* buffer, int rows, int cols) {
    srand(time(0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            buffer[i * cols + j] = rand() % 256 - 128;
        }
    }
}

void print_buffer(uint8_t* buffer, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << (int)buffer[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_buffer(int8_t* buffer, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << (int)buffer[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void print_buffer(uint32_t* buffer, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << buffer[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void check_result(int32_t* C, int32_t* C_ans, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i * N + j] != C_ans[i * N + j]) {
                std::cerr << "Error at (" << i << ", " << j << "): " << C[i * N + j] << " != " << C_ans[i * N + j] << std::endl;
                exit(1);
            }
        }
    }
    std::cout << "Result is correct!" << std::endl;
}

void check_result(uint8_t* B_reshape, uint8_t* B, int N, int K) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K * 4; j++) {
            if (B_reshape[i * K * 4 + j] != B[i * K * 4 + j]) {
                std::cerr << "Error at (" << i << ", " << j << "): " << (int)B_reshape[i * K * 4 + j] << " != " << (int)B[i * K * 4 + j] << std::endl;
                return;
            }
        }
    }
    std::cout << "Result is correct!" << std::endl;
}

void check_result(int8_t* B_reshape, int8_t* B, int N, int K) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K * 4; j++) {
            if (B_reshape[i * K * 4 + j] != B[i * K * 4 + j]) {
                std::cerr << "Error at (" << i << ", " << j << "): " << (int)B_reshape[i * K * 4 + j] << " != " << (int)B[i * K * 4 + j] << std::endl;
                return;
            }
        }
    }
    std::cout << "Result is correct!" << std::endl;
}