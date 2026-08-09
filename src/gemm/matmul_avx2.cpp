#include "../main.hpp"

void avx2(double* x, double* w, double* xout, int array_size){
    __m256d v_sum, v_x, v_x2, v_x3, v_x4, v_w, v_w2, v_w3, v_w4;

    for(int i = 0; i < array_size; ++i){
        int j = 0;

        for(; j < array_size; j += 4){ // avx2 incrementation
            v_sum = _mm256_setzero_pd();

            int k = 0;
            for(; k < array_size; k += 4){ // rb incrementation
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

void matmul_avx2(double* x, double* w, double* xout, int array_size, int registers, int reruns){
    double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        avx2(x, w, xout, array_size);
    }
    end = omp_get_wtime();

    double flops = (2.0 * array_size * array_size * array_size * reruns) / ((end - start)*1e9);
    std::cout << "AVX2 Intrinsics Implementation\nTime Elapsed: " << (end - start) << "s\nArray size: " << array_size << "\nGFLOPS: " << flops << std::endl;
}