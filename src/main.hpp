#ifndef MAIN
#define MAIN

#include <iostream>
#include <cstring>
#include <cstdint>
#include <omp.h>

// matmul implementations
void matmul_base(double* x, double* w, double* xout, int array_size, int registers, int reruns);
void matmul_sse(double* x, double* w, double* xout, int array_size, int registers, int reruns);
void matmul_avx2(double* x, double* w, double* xout, int array_size, int registers, int reruns);

#endif