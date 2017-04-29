#pragma once

#include "cudaHelper.h"


namespace CUDA{

#define WARP_SIZE 32


inline __device__ int get_grid_size()
{
    return blockDim.x * gridDim.x;
}


inline __device__ int get_global_index()
{
    unsigned int globalThreadNum = blockIdx.x * blockDim.x + threadIdx.x;
    return globalThreadNum;
}
inline __device__ int get_global_index_2D()
{
    int threadid = threadIdx.y * blockDim.x + threadIdx.x;
    int blockNumInGrid = blockIdx.x + gridDim.x  * blockIdx.y;
    int block_width = blockDim.x * blockDim.y;
    int globalThreadNum = blockNumInGrid * block_width + threadid;
    return globalThreadNum;
}



static __inline__ __device__ double fetch_double(uint2 p){
    return __hiloint2double(p.y, p.x);
}



__device__ inline
double __shfl_downD(double var, unsigned int srcLane, int width=32) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__inline__ __device__
double warpReduceSumD(double val, int width=warpSize) {
    for (int offset = width/2; offset > 0; offset /= 2)
        val += __shfl_downD(val, offset,width);
    return val;
}

//broadcasts
__inline__ __device__
double warpBroadcast(double val, int srcLane, int width=warpSize) {
    int2 a = *reinterpret_cast<int2*>(&val);
    a.x = __shfl(a.x, srcLane, width);
    a.y = __shfl(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__inline__ __device__
float warpReduceSum(float val, int width=warpSize) {
    for (int offset = width/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset,width);
    return val;
}

}
