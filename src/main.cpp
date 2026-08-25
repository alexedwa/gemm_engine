#include "main.hpp"

void gemm(int array_size, int reruns) {
    // 2d matrices
    alignas(64) double* x = (double*)malloc(sizeof(double) * array_size * array_size);
    alignas(64) double* w = (double*)malloc(sizeof(double) * array_size * array_size);
    alignas(64) double* xout = (double*)malloc(sizeof(double) * array_size * array_size);
    double* xout_check = (double*)malloc(sizeof(double) * array_size * array_size);

    //L1 Instruction, L1 Data, L2, L3
    int cache_size[4]; 
    get_cache_size(cache_size);

    omp_set_num_threads(omp_get_max_threads());


    initialise(x, w, array_size);
    matmul_unoptimised(x, w, xout_check, array_size, reruns);

    //matmul_base_scalar(x, w, xout, array_size, registers, reruns);
    //matmul_sse(x, w, xout, array_size, registers, cache_size, reruns);
    matmul_avx2(x, w, xout, array_size, cache_size, reruns);

    correctness_check(xout, xout_check, array_size);

    free(x);
    free(w);
    free(xout);
    free(xout_check);
}


int main(int argc, char* argv[]) {
    int reruns = 5; // baseline number of reruns
    int array_size = 1024; // baseline array size

    uint8_t i = 1;
    while(i < argc){
        if(strcmp(argv[i], "-s") == 0 && i + 1 < argc){
            array_size = std::atoi(argv[i + 1]);
        }      
        if(strcmp(argv[i], "-rr") == 0 && i + 1 < argc){
            reruns = std::atoi(argv[i + 1]);
        }
        
        i++;
    }

    gemm(array_size, reruns);

    return 0;
}