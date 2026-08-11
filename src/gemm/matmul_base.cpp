#include "../main.hpp"

void matmul_scalar(double* x, double* w, double* xout, int array_size){
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

void matmul_scalar_rb4(double* x, double* w, double* xout, int array_size){
    for(int i = 0; i < array_size; ++i){
        int j = 0;
        for(; j < array_size; j+=4){ // rb incrementation
            double sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;

            for(int k = 0; k < array_size; ++k){
                sum1 += x[i * array_size + k] * w[k * array_size + j];
                sum2 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                sum3 += x[i * array_size + k] * w[k * array_size + (j + 2)];
                sum4 += x[i * array_size + k] * w[k * array_size + (j + 3)];

            }
            xout[i * array_size + j] = sum1;
            xout[i * array_size + (j + 1)] = sum2;
            xout[i * array_size + (j + 2)] = sum3;
            xout[i * array_size + (j + 3)] = sum4;
        }

        // cleanup
        for(; j < array_size; ++j){
            double sum = 0.0f;
            for(int k = 0; k < array_size; ++k){
                sum += x[i * array_size + k] * w[k * array_size + j];
            }
            xout[i * array_size + j] = sum;
        }
    }
}

void matmul_scalar_rb8(double* x, double* w, double* xout, int array_size){
    for(int i = 0; i < array_size; ++i){

        int j = 0;
        for(; j < array_size; j+=8){ // rb incrementation
            double sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f, sum8 = 0.0f;

            for(int k = 0; k < array_size; ++k){
                sum1 += x[i * array_size + k] * w[k * array_size + j];
                sum2 += x[i * array_size + k] * w[k * array_size + (j + 1)];
                sum3 += x[i * array_size + k] * w[k * array_size + (j + 2)];
                sum4 += x[i * array_size + k] * w[k * array_size + (j + 3)];
                sum5 += x[i * array_size + k] * w[k * array_size + (j + 4)];
                sum6 += x[i * array_size + k] * w[k * array_size + (j + 5)];
                sum7 += x[i * array_size + k] * w[k * array_size + (j + 6)];
                sum8 += x[i * array_size + k] * w[k * array_size + (j + 7)];

            }
            xout[i * array_size + j] = sum1;
            xout[i * array_size + (j + 1)] = sum2;
            xout[i * array_size + (j + 2)] = sum3;
            xout[i * array_size + (j + 3)] = sum4;
            xout[i * array_size + (j + 4)] = sum5;
            xout[i * array_size + (j + 5)] = sum6;
            xout[i * array_size + (j + 6)] = sum7;
            xout[i * array_size + (j + 7)] = sum8;
        }

        // cleanup
        for(; j < array_size; ++j){
            double sum = 0.0f;
            for(int k = 0; k < array_size; ++k){
                sum += x[i * array_size + k] * w[k * array_size + j];
            }
            xout[i * array_size + j] = sum;
        }
    }
}

void matmul_base_scalar(double* x, double* w, double* xout, int array_size, int registers, int reruns){
    double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        matmul_scalar_rb8(x, w, xout, array_size);
    }
    end = omp_get_wtime();

    double average_time = (end - start) / reruns;
    double flops = (2.0 * array_size * array_size * array_size) / (average_time * 1e9);
    std::cout << "===============================\n";
    std::cout << "\e[1mBaseline Implementation\e[0m\nTotal Time Elapsed: " << (end - start) 
    << "s\nAverage time: " << average_time << "s\n"
    << "Array size: " << array_size 
    << "\nGFLOPS: " << flops 
    << std::endl;
    std::cout << "===============================\n";
}

void matmul_unoptimised(double* x, double* w, double* xout, int array_size, int registers, int reruns){
        double start, end;

    start = omp_get_wtime();
    for(int i = 0; i < reruns; ++i){
        matmul_scalar(x, w, xout, array_size);
    }
    end = omp_get_wtime();

    double average_time = (end - start) / reruns;
    double flops = (2.0 * array_size * array_size * array_size) / (average_time * 1e9);
    std::cout << "===============================\n";
    std::cout << "\e[1mUnoptimised Implementation\e[0m\nTotal Time Elapsed: " << (end - start) 
    << "s\nAverage time: " << average_time << "s\n"
    << "Array size: " << array_size 
    << "\nGFLOPS: " << flops 
    << std::endl;
    std::cout << "===============================\n";
}
