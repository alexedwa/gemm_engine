#ifndef MAIN
#define MAIN

#include <iostream>
#include <cstring>
#include <cstdint>
#include <omp.h>

#include <emmintrin.h> // sse
#include <immintrin.h> // avx2

// matmul implementations
void matmul_base(double* x, double* w, double* xout, int array_size, int registers, int reruns);

#endif