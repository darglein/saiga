////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>

#include <cuda_runtime.h>

namespace CUDA{

__global__ void
testKernel(float *g_idata, float *g_odata)
{
    int tid = threadIdx.x;

    // write data to global memory
    g_odata[tid] = g_idata[tid]+5;
}


int test()
{

    unsigned int num_threads = 32;
    unsigned int mem_size = sizeof(float) * num_threads;

    // allocate host memory
    float *h_idata = (float *) malloc(mem_size);

    // initalize the memory
    for (unsigned int i = 0; i < num_threads; ++i)
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float *d_idata;
    cudaMalloc((void **) &d_idata, mem_size);
    // copy host memory to device
    cudaMemcpy(d_idata, h_idata, mem_size,cudaMemcpyHostToDevice);

    // allocate device memory for result
    float *d_odata;
    cudaMalloc((void **) &d_odata, mem_size);

    // setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>(d_idata, d_odata);


    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(mem_size);
    // copy result from device to host
    cudaMemcpy(h_odata, d_odata, sizeof(float) * num_threads,cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < num_threads; ++i)
    {
        std::cout<<h_odata[i]<<std::endl;
    }


    // cleanup memory
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

}

}
