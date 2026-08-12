#ifndef MAIN
#define MAIN

#include <iostream>
#include <cstring>
#include <cstdint>
#include <omp.h>

// matmul implementations
void matmul_base(double* x, double* w, double* xout, int array_size, int registers, int reruns);
void matmul_unoptimised(double* x, double* w, double* xout, int array_size, int registers, int reruns);
void matmul_sse(double* x, double* w, double* xout, int array_size, int registers, int cache_line[4], int reruns);
void matmul_avx2(double* x, double* w, double* xout, int array_size, int registers, int cache_line[4], int reruns);

// util 
void initialise(double* x, double* w, int size);
void correctness_check(double* xout, double* xout_check, int size);
void get_cache_size(int cache_size[4]);

#endif