#include <immintrin.h> // avx2
#include "../main.hpp"

void avx2_rb3(double* x, double* w, double* xout, int array_size){
    __m256d v_sum, v_x, v_x2, v_x3, v_w, v_w2, v_w3; // 7 (14 registers used)

    for(int i = 0; i < array_size; ++i){
        int j = 0;

        for(; j + 3 < array_size; j += 4){ // avx2 incrementation
            v_sum = _mm256_setzero_pd();

            int k = 0;
            for(; k + 2 < array_size; k += 3){ // rb incrementation
                v_x = _mm256_set1_pd(x[i * array_size + k]);
                v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
                v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);

                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_w2 = _mm256_loadu_pd(&w[(k + 1) * array_size + j]);
                v_w3 = _mm256_loadu_pd(&w[(k + 2) * array_size + j]);

                v_sum = _mm256_fmadd_pd(v_x, v_w, v_sum);
                v_sum = _mm256_fmadd_pd(v_x2, v_w2, v_sum);
                v_sum = _mm256_fmadd_pd(v_x3, v_w3, v_sum);
            }

            // cleanup
            double tail0 = 0.0, tail1 = 0.0, tail2 = 0.0, tail3 = 0.0;
            for(; k < array_size; ++k){
                tail0 += x[i * array_size + k] * w[k * array_size + j];
                tail1 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                tail2 += x[i * array_size + k] * w[k * array_size + (j + 2)];
                tail3 += x[i * array_size + k] * w[k * array_size + (j + 3)];
            }

            double packed[4];
            _mm256_storeu_pd(packed, v_sum);
            xout[i * array_size + j] = packed[0] + tail0;
            xout[i * array_size + (j + 1)] = packed[1] + tail1;
            xout[i * array_size + (j + 2)] = packed[2] + tail2;
            xout[i * array_size + (j + 3)] = packed[3] + tail3;
        }

        // cleanup
        for(; j < array_size; ++j){
            double res = 0.0;
            for(int k = 0; k < array_size; ++k){
                res += x[i * array_size + k] * w[k * array_size + j];
            }
            xout[i * array_size + j] = res;
        }
    }
}

void avx2_rb4(double* x, double* w, double* xout, int array_size){
    __m256d v_sum, v_x, v_x2, v_x3, v_x4, v_w, v_w2, v_w3, v_w4; // 7 (14 registers used)

    for(int i = 0; i < array_size; ++i){
        int j = 0;

        for(; j + 3 < array_size; j += 4){ // avx2 incrementation
            v_sum = _mm256_setzero_pd();

            int k = 0;
            for(; k + 2 < array_size; k += 4){ // rb incrementation
                v_x = _mm256_set1_pd(x[i * array_size + k]);
                v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
                v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
                v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);

                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_w2 = _mm256_loadu_pd(&w[(k + 1) * array_size + j]);
                v_w3 = _mm256_loadu_pd(&w[(k + 2) * array_size + j]);
                v_w4 = _mm256_loadu_pd(&w[(k + 3) * array_size + j]);

                v_sum = _mm256_fmadd_pd(v_x, v_w, v_sum);
                v_sum = _mm256_fmadd_pd(v_x2, v_w2, v_sum);
                v_sum = _mm256_fmadd_pd(v_x3, v_w3, v_sum);
                v_sum = _mm256_fmadd_pd(v_x4, v_w4, v_sum);
            }

            // cleanup
            double tail0 = 0.0, tail1 = 0.0, tail2 = 0.0, tail3 = 0.0;
            for(; k < array_size; ++k){
                tail0 += x[i * array_size + k] * w[k * array_size + j];
                tail1 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                tail2 += x[i * array_size + k] * w[k * array_size + (j + 2)];
                tail3 += x[i * array_size + k] * w[k * array_size + (j + 3)];
            }

            double packed[4];
            _mm256_storeu_pd(packed, v_sum);
            xout[i * array_size + j] = packed[0] + tail0;
            xout[i * array_size + (j + 1)] = packed[1] + tail1;
            xout[i * array_size + (j + 2)] = packed[2] + tail2;
            xout[i * array_size + (j + 3)] = packed[3] + tail3;
        }

        // cleanup
        for(; j < array_size; ++j){
            double res = 0.0;
            for(int k = 0; k < array_size; ++k){
                res += x[i * array_size + k] * w[k * array_size + j];
            }
            xout[i * array_size + j] = res;
        }
    }
}

void avx2_rb7(double* x, double* w, double* xout, int array_size){
    __m256d v_sum, v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_w, v_w2, v_w3, v_w4, v_w5, v_w6, v_w7; // 7 (14 registers used)

    for(int i = 0; i < array_size; ++i){
        int j = 0;

        for(; j + 3 < array_size; j += 4){ // avx2 incrementation
            v_sum = _mm256_setzero_pd();

            int k = 0;
            for(; k + 2 < array_size; k += 7){ // rb incrementation
                v_x = _mm256_set1_pd(x[i * array_size + k]);
                v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
                v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
                v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);
                v_x5 = _mm256_set1_pd(x[i * array_size + (k + 4)]);
                v_x6 = _mm256_set1_pd(x[i * array_size + (k + 5)]);
                v_x7 = _mm256_set1_pd(x[i * array_size + (k + 6)]);

                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_w2 = _mm256_loadu_pd(&w[(k + 1) * array_size + j]);
                v_w3 = _mm256_loadu_pd(&w[(k + 2) * array_size + j]);
                v_w4 = _mm256_loadu_pd(&w[(k + 3) * array_size + j]);
                v_w5 = _mm256_loadu_pd(&w[(k + 4) * array_size + j]);
                v_w6 = _mm256_loadu_pd(&w[(k + 5) * array_size + j]);
                v_w7 = _mm256_loadu_pd(&w[(k + 6) * array_size + j]);

                v_sum = _mm256_fmadd_pd(v_x, v_w, v_sum);
                v_sum = _mm256_fmadd_pd(v_x2, v_w2, v_sum);
                v_sum = _mm256_fmadd_pd(v_x3, v_w3, v_sum);
                v_sum = _mm256_fmadd_pd(v_x4, v_w4, v_sum);
                v_sum = _mm256_fmadd_pd(v_x5, v_w5, v_sum);
                v_sum = _mm256_fmadd_pd(v_x6, v_w6, v_sum);
                v_sum = _mm256_fmadd_pd(v_x7, v_w7, v_sum);
            }

            // cleanup
            double tail0 = 0.0, tail1 = 0.0, tail2 = 0.0, tail3 = 0.0;
            for(; k < array_size; ++k){
                tail0 += x[i * array_size + k] * w[k * array_size + j];
                tail1 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                tail2 += x[i * array_size + k] * w[k * array_size + (j + 2)];
                tail3 += x[i * array_size + k] * w[k * array_size + (j + 3)];
            }

            double packed[4];
            _mm256_storeu_pd(packed, v_sum);
            xout[i * array_size + j] = packed[0] + tail0;
            xout[i * array_size + (j + 1)] = packed[1] + tail1;
            xout[i * array_size + (j + 2)] = packed[2] + tail2;
            xout[i * array_size + (j + 3)] = packed[3] + tail3;
        }

        // cleanup
        for(; j < array_size; ++j){
            double res = 0.0;
            for(int k = 0; k < array_size; ++k){
                res += x[i * array_size + k] * w[k * array_size + j];
            }
            xout[i * array_size + j] = res;
        }
    }
}

void avx2_rb8(double* x, double* w, double* xout, int array_size){
    __m256d v_sum, v_x, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7, v_x8, v_w, v_w2, v_w3, v_w4, v_w5, v_w6, v_w7, v_w8; // 7 (14 registers used)

    for(int i = 0; i < array_size; ++i){
        int j = 0;

        for(; j + 3 < array_size; j += 4){ // avx2 incrementation
            v_sum = _mm256_setzero_pd();

            int k = 0;
            for(; k + 2 < array_size; k += 8){ // rb incrementation
                v_x = _mm256_set1_pd(x[i * array_size + k]);
                v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
                v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);
                v_x4 = _mm256_set1_pd(x[i * array_size + (k + 3)]);
                v_x5 = _mm256_set1_pd(x[i * array_size + (k + 4)]);
                v_x6 = _mm256_set1_pd(x[i * array_size + (k + 5)]);
                v_x7 = _mm256_set1_pd(x[i * array_size + (k + 6)]);
                v_x8 = _mm256_set1_pd(x[i * array_size + (k + 7)]);

                v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                v_w2 = _mm256_loadu_pd(&w[(k + 1) * array_size + j]);
                v_w3 = _mm256_loadu_pd(&w[(k + 2) * array_size + j]);
                v_w4 = _mm256_loadu_pd(&w[(k + 3) * array_size + j]);
                v_w5 = _mm256_loadu_pd(&w[(k + 4) * array_size + j]);
                v_w6 = _mm256_loadu_pd(&w[(k + 5) * array_size + j]);
                v_w7 = _mm256_loadu_pd(&w[(k + 6) * array_size + j]);
                v_w8 = _mm256_loadu_pd(&w[(k + 7) * array_size + j]);

                v_sum = _mm256_fmadd_pd(v_x, v_w, v_sum);
                v_sum = _mm256_fmadd_pd(v_x2, v_w2, v_sum);
                v_sum = _mm256_fmadd_pd(v_x3, v_w3, v_sum);
                v_sum = _mm256_fmadd_pd(v_x4, v_w4, v_sum);
                v_sum = _mm256_fmadd_pd(v_x5, v_w5, v_sum);
                v_sum = _mm256_fmadd_pd(v_x6, v_w6, v_sum);
                v_sum = _mm256_fmadd_pd(v_x7, v_w7, v_sum);
                v_sum = _mm256_fmadd_pd(v_x8, v_w8, v_sum);
            }

            // cleanup
            double tail0 = 0.0, tail1 = 0.0, tail2 = 0.0, tail3 = 0.0;
            for(; k < array_size; ++k){
                tail0 += x[i * array_size + k] * w[k * array_size + j];
                tail1 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                tail2 += x[i * array_size + k] * w[k * array_size + (j + 2)];
                tail3 += x[i * array_size + k] * w[k * array_size + (j + 3)];
            }

            double packed[4];
            _mm256_storeu_pd(packed, v_sum);
            xout[i * array_size + j] = packed[0] + tail0;
            xout[i * array_size + (j + 1)] = packed[1] + tail1;
            xout[i * array_size + (j + 2)] = packed[2] + tail2;
            xout[i * array_size + (j + 3)] = packed[3] + tail3;
        }

        // cleanup
        for(; j < array_size; ++j){
            double res = 0.0;
            for(int k = 0; k < array_size; ++k){
                res += x[i * array_size + k] * w[k * array_size + j];
            }
            xout[i * array_size + j] = res;
        }
    }
}

void avx2_rb3_lt(double* x, double* w, double* xout, int array_size, int cache_line[4]){
    __m256d v_sum, v_x, v_x2, v_x3, v_w, v_w2, v_w3; // 7 (14 registers used)
    int l1d_tile = cache_line[1] / (sizeof(double));


    for(int kk = 0; kk < array_size; kk += l1d_tile){
        for(int jj = 0; jj < array_size; jj += l1d_tile){
            for(int ii = 0; ii < array_size; ii += l1d_tile){
                for(int i = ii; i < std::min(ii + l1d_tile, array_size); ++i){
                    int j = jj;

                    for(; j + 3 < std::min(jj + l1d_tile, array_size); j += 4){ // avx2 incrementation
                        v_sum = _mm256_setzero_pd();

                        int k = kk;
                        for(; k + 2 < std::min(kk + l1d_tile, array_size); k += 3){ // rb incrementation
                            v_x = _mm256_set1_pd(x[i * array_size + k]);
                            v_x2 = _mm256_set1_pd(x[i * array_size + (k + 1)]);
                            v_x3 = _mm256_set1_pd(x[i * array_size + (k + 2)]);

                            v_w = _mm256_loadu_pd(&w[k * array_size + j]);
                            v_w2 = _mm256_loadu_pd(&w[(k + 1) * array_size + j]);
                            v_w3 = _mm256_loadu_pd(&w[(k + 2) * array_size + j]);

                            v_sum = _mm256_fmadd_pd(v_x, v_w, v_sum);
                            v_sum = _mm256_fmadd_pd(v_x2, v_w2, v_sum);
                            v_sum = _mm256_fmadd_pd(v_x3, v_w3, v_sum);
                        }

                        // cleanup
                        double tail0 = 0.0, tail1 = 0.0, tail2 = 0.0, tail3 = 0.0;
                        for(; k < array_size; ++k){
                            tail0 += x[i * array_size + k] * w[k * array_size + j];
                            tail1 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                            tail2 += x[i * array_size + k] * w[k * array_size + (j + 2)];
                            tail3 += x[i * array_size + k] * w[k * array_size + (j + 3)];
                        }

                        double packed[4];
                        _mm256_storeu_pd(packed, v_sum);
                        xout[i * array_size + j] = packed[0] + tail0;
                        xout[i * array_size + (j + 1)] = packed[1] + tail1;
                        xout[i * array_size + (j + 2)] = packed[2] + tail2;
                        xout[i * array_size + (j + 3)] = packed[3] + tail3;
                    }

                    // cleanup
                    for(; j < array_size; ++j){
                        double res = 0.0;
                        for(int k = 0; k < array_size; ++k){
                            res += x[i * array_size + k] * w[k * array_size + j];
                        }
                        xout[i * array_size + j] = res;
                    }
                }
            }
        }
    }
}

void matmul_avx2(double* x, double* w, double* xout, int array_size, int registers, int cache_line[4], int reruns){
    double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        //avx2_rb3(x, w, xout, array_size);
        avx2_rb3_lt(x, w, xout, array_size, cache_line);
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