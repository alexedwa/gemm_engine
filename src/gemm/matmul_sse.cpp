#include <emmintrin.h> // sse
#include "../main.hpp"

/*
    SSE INTRINSICS IMPLEMENTATIONS
*/
void sse_simd(double* x, double* w, double* xout, int array_size){
    __m128d v_x, v_w, v_xout; // 3 
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int i = 0; i < array_size; ++i){
        for(int k = 0; k < array_size; ++k){
            v_x = _mm_set1_pd(x[i * array_size + k]);
            int j = 0;
            for(; j + 1 < array_size; j += 2){
                v_w = _mm_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                _mm_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}



/*
    REGISTER BLOCKING IMPLEMENTATIONS
*/
void sse_rb2(double* x, double* w, double* xout, int array_size){
    __m128d v_x, v_x2, v_w, v_xout; // 6
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int i = 0; i < array_size; ++i){

        int k = 0;
        for(; k + 1 < array_size; k += 2){
            v_x = _mm_set1_pd(x[i * array_size + k]);
            v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);

            int j = 0;
            for(; j + 1 < array_size; j += 2){
                v_w = _mm_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x2, v_w));

                _mm_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            // cleanup
            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }

        // cleanup
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void sse_rb4(double* x, double* w, double* xout, int array_size){
    __m128d v_x, v_x2, v_x3, v_x4, v_w, v_xout; // 6
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int i = 0; i < array_size; ++i){

        int k = 0;
        for(; k + 3 < array_size; k += 4){
            v_x = _mm_set1_pd(x[i * array_size + k]);
            v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm_set1_pd(x[i * array_size + (k + 3)]);
            int j = 0;
            for(; j + 1 < array_size; j += 2){
                v_w = _mm_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x2, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x3, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x4, v_w));

                _mm_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            // cleanup
            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }

        // cleanup
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void sse_rb8(double* x, double* w, double* xout, int array_size){
    __m128d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_w, v_xout; // 10
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int i = 0; i < array_size; ++i){

        int k = 0;
        for(; k + 7 < array_size; k += 8){
            v_x = _mm_set1_pd(x[i * array_size + k]);
            v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm_set1_pd(x[i * array_size + (k + 3)]);
            v_x5 = _mm_set1_pd(x[i * array_size + (k + 4)]);
            v_x6 = _mm_set1_pd(x[i * array_size + (k + 5)]);
            v_x7 = _mm_set1_pd(x[i * array_size + (k + 6)]);
            v_x8 = _mm_set1_pd(x[i * array_size + (k + 7)]);

            int j = 0;
            for(; j + 1 < array_size; j += 2){
                v_w = _mm_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x2, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x3, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x4, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x5, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x6, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x7, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x8, v_w));

                _mm_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            // cleanup
            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }

        // cleanup
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void sse_rb10(double* x, double* w, double* xout, int array_size){
    __m128d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_x9, v_x10, v_w, v_xout; // 12
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int i = 0; i < array_size; ++i){

        int k = 0;
        for(; k + 9 < array_size; k += 10){
            v_x = _mm_set1_pd(x[i * array_size + k]);
            v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm_set1_pd(x[i * array_size + (k + 3)]);
            v_x5 = _mm_set1_pd(x[i * array_size + (k + 4)]);
            v_x6 = _mm_set1_pd(x[i * array_size + (k + 5)]);
            v_x7 = _mm_set1_pd(x[i * array_size + (k + 6)]);
            v_x8 = _mm_set1_pd(x[i * array_size + (k + 7)]);
            v_x9 = _mm_set1_pd(x[i * array_size + (k + 8)]);
            v_x10 = _mm_set1_pd(x[i * array_size + (k + 9)]);


            int j = 0;
            for(; j + 1 < array_size; j += 2){
                v_w = _mm_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x2, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x3, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x4, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x5, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x6, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x7, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x8, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x9, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x10, v_w));


                _mm_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            // cleanup
            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }

        // cleanup
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}

void sse_rb12(double* x, double* w, double* xout, int array_size){
    __m128d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_x9, v_x10, v_x11, v_x12, v_w, v_xout; // 14
    
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int i = 0; i < array_size; ++i){
        int k = 0;
        for(; k + 11 < array_size; k += 12){
            v_x = _mm_set1_pd(x[i * array_size + k]);
            v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
            v_x3 = _mm_set1_pd(x[i * array_size + (k + 2)]);
            v_x4 = _mm_set1_pd(x[i * array_size + (k + 3)]);
            v_x5 = _mm_set1_pd(x[i * array_size + (k + 4)]);
            v_x6 = _mm_set1_pd(x[i * array_size + (k + 5)]);
            v_x7 = _mm_set1_pd(x[i * array_size + (k + 6)]);
            v_x8 = _mm_set1_pd(x[i * array_size + (k + 7)]);
            v_x9 = _mm_set1_pd(x[i * array_size + (k + 8)]);
            v_x10 = _mm_set1_pd(x[i * array_size + (k + 9)]);
            v_x11 = _mm_set1_pd(x[i * array_size + (k + 10)]);
            v_x12 = _mm_set1_pd(x[i * array_size + (k + 11)]);

            int j = 0;
            for(; j + 1 < array_size; j += 2){
                v_w = _mm_loadu_pd(&w[k * array_size + j]);
                v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x2, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x3, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x4, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x5, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x6, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x7, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x8, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x9, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x10, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x11, v_w));
                v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x12, v_w));

                _mm_storeu_pd(&xout[i * array_size + j], v_xout);
            }

            // cleanup
            for (; j < array_size; ++j) {
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }

        // cleanuip
        for(; k < array_size; ++k){
            for(int j = 0; j < array_size; ++j){
                xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
            }
        }
    }
}



/*
    LOOP TILING IMPLEMENTATION
*/
void sse_rb8_lt_l1d(double* x, double* w, double* xout, int array_size, int cache_line[4]){
    __m128d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_w, v_xout; // 10
    int l1d_tile = cache_line[1] / (sizeof(double) * 7);
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    for(int kk = 0; kk < array_size; kk += l1d_tile){
        for(int jj = 0; jj < array_size; jj += l1d_tile){
            for(int ii = 0; ii < array_size; ii += l1d_tile){

                for(int i = ii; i < std::min(ii + l1d_tile, array_size); ++i){
                    int k = kk;
                    for(; k + 7 < std::min(kk + l1d_tile, array_size); k += 8){
                        v_x = _mm_set1_pd(x[i * array_size + k]);
                        v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
                        v_x3 = _mm_set1_pd(x[i * array_size + (k + 2)]);
                        v_x4 = _mm_set1_pd(x[i * array_size + (k + 3)]);
                        v_x5 = _mm_set1_pd(x[i * array_size + (k + 4)]);
                        v_x6 = _mm_set1_pd(x[i * array_size + (k + 5)]);
                        v_x7 = _mm_set1_pd(x[i * array_size + (k + 6)]);
                        v_x8 = _mm_set1_pd(x[i * array_size + (k + 7)]);

                        int j = jj;
                        for(; j + 1 < std::min(jj + l1d_tile, array_size); j += 2){
                            v_w = _mm_loadu_pd(&w[k * array_size + j]);
                            v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x2, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x3, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x4, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x5, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x6, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x7, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x8, v_w));

                            _mm_storeu_pd(&xout[i * array_size + j], v_xout);
                        }

                        //cleanup
                        for (; j <  std::min(jj + l1d_tile, array_size); ++j) {
                            xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
                        }
                    }

                    // cleanup
                    for(; k <  std::min(kk + l1d_tile, array_size); ++k){
                        for(int j = 0; j < array_size; ++j){
                            xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
                        }
                    }
                }
            }
        }
    }
}



/*
    OMP IMPLEMENTATION
*/
void sse_rb8_lt_l1d_omp(double* x, double* w, double* xout, int array_size, int cache_line[4]){
    __m128d v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_w, v_xout; // 10
    int l1d_tile = cache_line[1] / (sizeof(double));
    memset(xout, 0, (size_t)array_size * array_size * sizeof(double));

    #pragma omp for private(v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_w, v_xout) schedule(static)
    for(int kk = 0; kk < array_size; kk += l1d_tile){
        for(int jj = 0; jj < array_size; jj += l1d_tile){
            for(int ii = 0; ii < array_size; ii += l1d_tile){

                for(int i = ii; i < std::min(ii + l1d_tile, array_size); ++i){
                    int k = kk;
                    for(; k + 7 < std::min(kk + l1d_tile, array_size); k += 8){
                        v_x = _mm_set1_pd(x[i * array_size + k]);
                        v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
                        v_x3 = _mm_set1_pd(x[i * array_size + (k + 2)]);
                        v_x4 = _mm_set1_pd(x[i * array_size + (k + 3)]);
                        v_x5 = _mm_set1_pd(x[i * array_size + (k + 4)]);
                        v_x6 = _mm_set1_pd(x[i * array_size + (k + 5)]);
                        v_x7 = _mm_set1_pd(x[i * array_size + (k + 6)]);
                        v_x8 = _mm_set1_pd(x[i * array_size + (k + 7)]);

                        int j = jj;
                        for(; j + 1 < std::min(jj + l1d_tile, array_size); j += 2){
                            v_w = _mm_loadu_pd(&w[k * array_size + j]);
                            v_xout = _mm_loadu_pd(&xout[i * array_size + j]);

                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x2, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x3, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x4, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x5, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x6, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x7, v_w));
                            v_xout = _mm_add_pd(v_xout, _mm_mul_pd(v_x8, v_w));

                            _mm_storeu_pd(&xout[i * array_size + j], v_xout);
                        }

                        //cleanup
                        for (; j <  std::min(jj + l1d_tile, array_size); ++j) {
                            xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
                        }
                    }

                    // cleanup
                    for(; k <  std::min(kk + l1d_tile, array_size); ++k){
                        for(int j = 0; j < array_size; ++j){
                            xout[i * array_size + j] += x[i * array_size + k] * w[k * array_size + j];
                        }
                    }
                }
            }
        }
    }
}




void matmul_sse(double* x, double* w, double* xout, int array_size, int cache_line[4], int reruns){
    double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        //sse_simd(x, w, xout, array_size);

        //sse_rb2(x, w, xout, array_size);
        //sse_rb4(x, w, xout, array_size);
        //sse_rb8(x, w, xout, array_size); // most optimal for 8 registers
        //sse_rb10(x, w, xout, array_size);
        //sse_rb12(x, w, xout, array_size);

        //sse_rb8_lt_l1d(x, w, xout, array_size, cache_line);

        sse_rb8_lt_l1d_omp(x, w, xout, array_size, cache_line);
    }
    end = omp_get_wtime();

    double average_time = (end - start) / reruns;
    double flops = (2.0 * array_size * array_size * array_size * reruns) / ((end - start)*1e9);
    std::cout << "===============================\n";
    std::cout << "\e[1mSSE Intrinsics Implementation\e[0m\nTime Elapsed: " << (end - start) 
    << "s\nAverage time: " << average_time << "s\n"
    << "Array size: " << array_size 
    << "\nGFLOPS: " << flops 
    << std::endl;
    std::cout << "===============================\n";

}
