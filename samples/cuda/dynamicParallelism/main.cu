/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"

#include <iostream>
#include <vector>

// Ressources:
// https://devblogs.nvidia.com/introduction-cuda-dynamic-parallelism/
// https://devblogs.nvidia.com/cuda-dynamic-parallelism-api-principles/

// Example output:
//
// sum before cudaDeviceSynchronize 32 0
// sum after cudaDeviceSynchronize 32 64

__device__ int sum = 0;
__device__ int sum2 = 0;

static void __global__ addOne()
{
    atomicAdd(&sum,1);
}

static void __global__ addOne2()
{
    // Test if the first kernel has finished
    if(sum == 32)
        atomicAdd(&sum2,1);
}

static void __global__ parent()
{
    // launch a kernel which adds to 'sum'
    addOne<<<1,32>>>();

    // this kernel is implicitly synchronized to 'addOne'.
    // -> This kernel adds 64 to 'sum2' because 'sum' is always 32
    addOne2<<<1,64>>>();

    // Write-Read race condiion.
    // Undefined output!
    printf("sum before cudaDeviceSynchronize %d %d\n",sum,sum2);

    // Wait for the two previous kerneles to finish
    cudaDeviceSynchronize();

    // Always (!) outputs 32,64
    printf("sum after cudaDeviceSynchronize %d %d\n",sum,sum2);
}

int main()
{
    parent<<<1,1>>>();
    cudaDeviceSynchronize();
    cout << "Done." << endl;
	return 0;
}
