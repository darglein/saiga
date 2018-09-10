/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include <cuda_runtime.h>

template<unsigned int BLOCK_SIZE>
__global__ static
void oob(int* data, int size)
{
    auto id = threadIdx.x + BLOCK_SIZE * blockIdx.x;
    if(id >= size) return;

    data[id + 1] = 0;
}

static void intraBlockTest(int N)
{



    // Compute launch arguments
    const unsigned int BLOCK_SIZE = 128;
    const unsigned int BLOCKS = N / BLOCK_SIZE;

    thrust::device_vector<int> d_data(N);

    oob<BLOCK_SIZE><<<BLOCKS,BLOCK_SIZE>>>(d_data.data().get(),N);

    cudaDeviceSynchronize();
}

// nvcc -arch=sm_61 -g main.cu -o segfault

int main(int argc, char *argv[])
{
    intraBlockTest(1024 * 2);

    std::cout << "Done." << std::endl;
}

