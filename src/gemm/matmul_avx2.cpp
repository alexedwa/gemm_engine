#include <immintrin.h> // avx2
#include "../main.hpp"

void avx2_simd(double* x, double* w, double* xout, int array_size){
    __m256d v_x, v_w, v_xout; // 3 (6 registers used)
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int i = 0; i < array_size; ++i){
        for(int k = 0; k < array_size; ++k){
            v_x = _mm256_set1_pd(x[i * array_size + k]);
            int j = 0;
            for(; j + 3 < array_size; j += 4){
                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm256_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm256_fmadd_pd(v_x, v_w, v_xout);
                _mm256_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}



void avx2_rb4(double* x, double* w, double* xout, int array_size){
    __m256d v_x, v_x2, v_x3, v_x4, v_w, v_xout; // 6 (12 * 2 registers used)
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));
    
    for(int i = 0; i < array_size; ++i){
        int k = 0;
        for(; k + 3 < array_size; k += 4){
            v_x = _mm256_set1_pd(x[i * array_size + (k + 0)]);
            v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);

            int j = 0;
            for(; j + 3 < array_size; j += 4){
                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm256_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm256_fmadd_pd(v_x, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x2, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x3, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x4, v_w, v_xout);

                _mm256_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void avx2_rb6(double* x, double* w, double* xout, int array_size){
    __m256d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_w, v_xout; // 14 (28 * 2 registers used)
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));
    
    for(int i = 0; i < array_size; ++i){
        int k = 0;
        for(; k + 5 < array_size; k += 6){
            v_x = _mm256_set1_pd(x[i * array_size + k]);
            v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);
            v_x5 = _mm256_set1_pd(x[i * array_size + (k + 4)]);
            v_x6 = _mm256_set1_pd(x[i * array_size + (k + 5)]);


            int j = 0;
            for(; j + 3 < array_size; j += 4){
                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm256_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm256_fmadd_pd(v_x, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x2, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x3, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x4, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x5, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x6, v_w, v_xout);


                _mm256_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void avx2_rb8(double* x, double* w, double* xout, int array_size){
    __m256d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_w, v_xout; // 6 (12 * 2 registers used)
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));
    
    for(int i = 0; i < array_size; ++i){
        int k = 0;
        for(; k + 7 < array_size; k += 8){
            v_x = _mm256_set1_pd(x[i * array_size + k]);
            v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);
            v_x5 = _mm256_set1_pd(x[i * array_size + (k + 4)]);
            v_x6 = _mm256_set1_pd(x[i * array_size + (k + 5)]);
            v_x7 = _mm256_set1_pd(x[i * array_size + (k + 6)]);
            v_x8 = _mm256_set1_pd(x[i * array_size + (k + 7)]);

            int j = 0;
            for(; j + 3 < array_size; j += 4){
                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm256_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm256_fmadd_pd(v_x, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x2, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x3, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x4, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x5, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x6, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x7, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x8, v_w, v_xout);

                _mm256_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void avx2_rb12(double* x, double* w, double* xout, int array_size){
    __m256d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_x9, v_x10, v_x11, v_x12, v_w, v_xout; // 14 (28 * 2 registers used)
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));
    
    for(int i = 0; i < array_size; ++i){
        int k = 0;
        for(; k + 11 < array_size; k += 12){
            v_x = _mm256_set1_pd(x[i * array_size + k]);
            v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);
            v_x5 = _mm256_set1_pd(x[i * array_size + (k + 4)]);
            v_x6 = _mm256_set1_pd(x[i * array_size + (k + 5)]);
            v_x7 = _mm256_set1_pd(x[i * array_size + (k + 6)]);
            v_x8 = _mm256_set1_pd(x[i * array_size + (k + 7)]);
            v_x9 = _mm256_set1_pd(x[i * array_size + (k + 8)]);
            v_x10 = _mm256_set1_pd(x[i * array_size + (k + 9)]);
            v_x11 = _mm256_set1_pd(x[i * array_size + (k + 10)]);
            v_x12 = _mm256_set1_pd(x[i * array_size + (k + 11)]);


            int j = 0;
            for(; j + 3 < array_size; j += 4){
                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm256_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm256_fmadd_pd(v_x, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x2, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x3, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x4, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x5, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x6, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x7, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x8, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x9, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x10, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x11, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x12, v_w, v_xout);

                _mm256_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void avx2_rb16(double* x, double* w, double* xout, int array_size){
    __m256d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_x9, v_x10, v_x11, v_x12, v_x13, v_x14, v_x15, v_x16, v_w, v_xout; // 18 (36 * 2 registers used)
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));
    
    for(int i = 0; i < array_size; ++i){
        int k = 0;
        for(; k + 15 < array_size; k += 16){
            v_x = _mm256_set1_pd(x[i * array_size + k]);
            v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);
            v_x5 = _mm256_set1_pd(x[i * array_size + (k + 4)]);
            v_x6 = _mm256_set1_pd(x[i * array_size + (k + 5)]);
            v_x7 = _mm256_set1_pd(x[i * array_size + (k + 6)]);
            v_x8 = _mm256_set1_pd(x[i * array_size + (k + 7)]);
            v_x9 = _mm256_set1_pd(x[i * array_size + (k + 8)]);
            v_x10 = _mm256_set1_pd(x[i * array_size + (k + 9)]);
            v_x11 = _mm256_set1_pd(x[i * array_size + (k + 10)]);
            v_x12 = _mm256_set1_pd(x[i * array_size + (k + 11)]);
            v_x13 = _mm256_set1_pd(x[i * array_size + (k + 12)]);
            v_x14 = _mm256_set1_pd(x[i * array_size + (k + 13)]);
            v_x15 = _mm256_set1_pd(x[i * array_size + (k + 14)]);
            v_x16 = _mm256_set1_pd(x[i * array_size + (k + 15)]);

            int j = 0;
            for(; j + 3 < array_size; j += 4){
                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm256_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm256_fmadd_pd(v_x, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x2, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x3, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x4, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x5, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x6, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x7, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x8, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x9, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x10, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x11, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x12, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x13, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x14, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x15, v_w, v_xout);
                v_xout = _mm256_fmadd_pd(v_x16, v_w, v_xout);

                _mm256_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}



void matmul_avx2(double* x, double* w, double* xout, int array_size, int registers, int cache_line[4], int reruns){
    double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        //avx2_simd(x, w, xout, array_size);

        //avx2_rb4(x, w, xout, array_size);
        //avx2_rb6(x, w, xout, array_size);
        //avx2_rb8(x, w, xout, array_size);
        avx2_rb12(x, w, xout, array_size); // most performant one
        //avx2_rb16(x, w, xout, array_size);

    }
    end = omp_get_wtime();

    double average_time = (end - start) / reruns;
    double flops = (2.0 * array_size * array_size * array_size * reruns) / ((end - start)*1e9);
    std::cout << "===============================\n";
    std::cout << "\e[1mAVX2 Intrinsics Implementation\e[0m\nTime Elapsed: " << (end - start) 
    << "s\nAverage time: " << average_time << "s\n"
    << "Array size: " << array_size 
    << "\nGFLOPS: " << flops 
    << std::endl;
    std::cout << "===============================\n";

}