/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <cuda_runtime.h>

__global__ void helloCudaKernel()
{
    printf("Hello from CUDA!\n");
}

int main(int argc, char *argv[])
{
    std::cout << "Hello CUDA?" << std::endl;
    helloCudaKernel<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

