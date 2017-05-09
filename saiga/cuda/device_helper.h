#pragma once

#include "cudaHelper.h"



#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//atomicAdd is already defined for compute capability 6.x and higher.
#else
#if 0
__device__ inline
double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ inline
double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
#endif


namespace CUDA{



//inline __device__ int get_grid_size()
//{
//    return blockDim.x * gridDim.x;
//}


//inline __device__ int get_global_index()
//{
//    unsigned int globalThreadNum = blockIdx.x * blockDim.x + threadIdx.x;
//    return globalThreadNum;
//}
//inline __device__ int get_global_index_2D()
//{
//    int threadid = threadIdx.y * blockDim.x + threadIdx.x;
//    int blockNumInGrid = blockIdx.x + gridDim.x  * blockIdx.y;
//    int block_width = blockDim.x * blockDim.y;
//    int globalThreadNum = blockNumInGrid * block_width + threadid;
//    return globalThreadNum;
//}




}
