#include "main.hpp"

void initialise(double* x, double* w, int size) {
    std::cout << "\nInitialising arrays..." << std::endl;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            x[i * size + j] = (double)(i%7+0.01);
            w[i * size + j] = (double)(j%7+0.01);
        }
    }

    std::cout << "Initialisation complete.\n" << std::endl;

}

void correctness_check(double* xout, double* xout_check, int size) {
    std::cout << "\nPerforming correctness check..." << std::endl;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (std::abs(xout[i * size + j] - xout_check[i * size + j]) > 1e-6) {
                std::cerr << "Error: Incorrect value at (" << i << ", " << j << "): "<< xout[i * size + j] << " != " << xout_check[i * size + j] << std::endl;
                return;
            }
        }
    }

    std::cout << "Correctness check passed.\n" << std::endl;
}

void gemm(int registers, int* cache_line[], int array_size, int reruns) {
    // 2d matrices
    double* x = (double*)malloc(sizeof(double) * array_size * array_size);
    double* w = (double*)malloc(sizeof(double) * array_size * array_size);
    double* xout = (double*)malloc(sizeof(double) * array_size * array_size);
    double* xout_check = (double*)malloc(sizeof(double) * array_size * array_size);

    initialise(x, w, array_size);
    matmul_base(x, w, xout_check, array_size, registers, reruns);

    //matmul_sse(x, w, xout, array_size, registers, reruns);
    matmul_avx2(x, w, xout, array_size, registers, reruns);

    correctness_check(xout, xout_check, array_size);

    free(x);
    free(w);
    free(xout);
    free(xout_check);
}


int main(int argc, char* argv[]) {
    int registers = 8; // baseline number of registers
    int reruns = 5; // baseline number of reruns
    int array_size = 1024; // baseline array size

    uint8_t i = 1;
    while(i < argc){
        if(strcmp(argv[i], "-r") == 0 && i + 1 < argc){
            registers = std::atoi(argv[i + 1]);
        }
        if(strcmp(argv[i], "-s") == 0 && i + 1 < argc){
            array_size = std::atoi(argv[i + 1]);
        }      
        if(strcmp(argv[i], "-rr") == 0 && i + 1 < argc){
            reruns = std::atoi(argv[i + 1]);
        }
        
        i++;
    }

    gemm(registers, nullptr, array_size, reruns);

    return 0;
}