#include <emmintrin.h> // sse
#include "../main.hpp"

void sse_rb2(double* x, double* w, double* xout, int array_size){
    __m128d v_sum, v_x, v_x2, v_w, v_w2;

    for(int i = 0; i < array_size; ++i){

        int j = 0;
        for(; j < array_size; j += 2){ // sse incrementation
            v_sum = _mm_setzero_pd();

            int k = 0;
            for(; k < array_size; k += 2){ // rb incrementation
                v_x  = _mm_set1_pd(x[i * array_size + k]);
                v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
                
                v_w  = _mm_loadu_pd(&w[k * array_size + j]);
                v_w2 = _mm_loadu_pd(&w[(k + 1) * array_size + j]);

                v_sum = _mm_add_pd(v_sum, _mm_mul_pd(v_x,  v_w));
                v_sum = _mm_add_pd(v_sum, _mm_mul_pd(v_x2, v_w2));
            }

            // cleanup
            double tail0 = 0.0, tail1 = 0.0;
            for(; k < array_size; ++k){
                tail0 += x[i * array_size + k] * w[k * array_size + j];
                tail1 += x[i * array_size + k] * w[k * array_size + (j + 1)];
            }

            double packed[2];
            _mm_storeu_pd(packed, v_sum);
            xout[i * array_size + j] = packed[0] + tail0;
            xout[i * array_size + (j + 1)] = packed[1] + tail1;
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

void sse_rb2_lt(double* x, double* w, double* xout, int array_size, int cache_size[4]){
    __m128d v_sum, v_x, v_x2, v_w, v_w2;
    int l1d_tile = cache_size[1] / (sizeof(double));

    for(int kk = 0; kk < array_size; kk += l1d_tile){
        for(int jj = 0; jj < array_size; jj += l1d_tile){
            for(int ii = 0; ii < array_size; ii += l1d_tile){
                for(int i = 0; i < array_size; ++i){
                    int j = jj;
                    for(; j + 1< array_size; j += 2){ // sse incrementation
                        v_sum = _mm_setzero_pd();

                        int k = kk;
                        for(; k + 1< array_size; k += 2){ // rb incrementation
                            v_x  = _mm_set1_pd(x[i * array_size + k]);
                            v_x2 = _mm_set1_pd(x[i * array_size + (k + 1)]);
                            
                            v_w  = _mm_loadu_pd(&w[k * array_size + j]);
                            v_w2 = _mm_loadu_pd(&w[(k + 1) * array_size + j]);

                            v_sum = _mm_add_pd(v_sum, _mm_mul_pd(v_x,  v_w));
                            v_sum = _mm_add_pd(v_sum, _mm_mul_pd(v_x2, v_w2));
                        }

                        // cleanup
                        double tail0 = 0.0, tail1 = 0.0;
                        for(; k < array_size; ++k){
                            tail0 += x[i * array_size + k] * w[k * array_size + j];
                            tail1 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                        }

                        double packed[2];
                        _mm_storeu_pd(packed, v_sum);
                        xout[i * array_size + j] = packed[0] + tail0;
                        xout[i * array_size + (j + 1)] = packed[1] + tail1;
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

void matmul_sse(double* x, double* w, double* xout, int array_size, int registers, int cache_line[4], int reruns){
    double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        //sse_rb2(x, w, xout, array_size);
        sse_rb2_lt(x, w, xout, array_size, cache_line);
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
