#include "../main.hpp"

void baseline(double* x, double* w, double* xout, int array_size){
    for(int i = 0; i < array_size; ++i){
        for(int j = 0; j < array_size; ++j){
            double sum = 0.0f;
            for(int k = 0; k < array_size; ++k){
                sum += x[i * array_size + k] * w[k * array_size + j];
            }
            xout[i * array_size + j] = sum;
        }
    }
}

void matmul_base(double* x, double* w, double* xout, int array_size, int registers, int reruns){
    double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        baseline(x, w, xout, array_size);
    }
    end = omp_get_wtime();

    double flops = (2.0 * array_size * array_size * array_size * reruns) / ((end - start)*1e9);
    std::cout << "Baseline Implementation\nTime Elapsed: " << (end - start) << "s\nArray size: " << array_size << "\nGFLOPS: " << flops << std::endl;
}