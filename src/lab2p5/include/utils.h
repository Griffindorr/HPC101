#ifndef _UTILS_H_
#define _UTILS_H_

#include <cstdint>

void init_buffer(uint8_t* buffer, int rows, int cols);
void init_buffer(int8_t* buffer, int rows, int cols);

void print_buffer(uint8_t* buffer, int rows, int cols);
void print_buffer(int8_t* buffer, int rows, int cols);
void print_buffer(uint32_t* buffer, int rows, int cols);

void check_result(int32_t* A, int32_t* C_ref, int M, int N);
void check_result(uint8_t* A, uint8_t* A_ref, int N, int K);
void check_result(int8_t* A, int8_t* A_ref, int N, int K);


#endif