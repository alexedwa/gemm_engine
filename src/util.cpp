//#include <intrin.h>
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

/* GETS CACHE SIZE
* Index 0: L1 cache size
* Index 1: L2 cache size
* Index 2: L3 cache size
*/
void get_cache_size(int cache_size[4]) {
    /*
    for(int i = 0; i < 4; ++i){
        int cpuInfo[4] = {0};
        __cpuidex(cpuInfo, 4, i);

        int level = (cpuInfo[0] >> 5) & 0x7;
        int line_size = (cpuInfo[1] & 0xFFF) + 1;
        int partitions = ((cpuInfo[1] >> 12) & 0x3FF) + 1;
        int ways = ((cpuInfo[1] >> 22) & 0x3FF) + 1;
        int sets = cpuInfo[2] + 1;

        int total_bytes = ways * partitions * line_size * sets;

        cache_size[level] = total_bytes;
    }
    */
   cache_size[1] = 262144; // L1 cache size in bytes
}
