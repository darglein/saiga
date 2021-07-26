/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>

#include <cuda_runtime.h>


#include "saiga/core/util/Align.h"

__global__ void helloCudaKernel()
{
    printf("Hello from thread %d on block %d!\n", threadIdx.x, blockIdx.x);
}


int main(int argc, char* argv[])
{
    std::cout << "Hello CUDA?" << std::endl;
    helloCudaKernel<<<2, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}

// Possible output:
//
// Hello CUDA?
// Hello from thread 0 on block 1!
// Hello from thread 1 on block 1!
// Hello from thread 2 on block 1!
// Hello from thread 3 on block 1!
// Hello from thread 0 on block 0!
// Hello from thread 1 on block 0!
// Hello from thread 2 on block 0!
// Hello from thread 3 on block 0!



