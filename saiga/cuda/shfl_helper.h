#pragma once

#include "saiga/cuda/common.h"


namespace CUDA{

__device__ inline
double fetch_double(uint2 p){
    return __hiloint2double(p.y, p.x);
}


__device__ inline
double __shfl_downD(double var, unsigned int srcLane, int width=32) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__device__ inline
double warpReduceSumD(double val, int width=warpSize) {
    for (int offset = width/2; offset > 0; offset /= 2)
        val += __shfl_downD(val, offset,width);
    return val;
}

//broadcasts
__device__ inline
double warpBroadcast(double val, int srcLane, int width=warpSize) {
    int2 a = *reinterpret_cast<int2*>(&val);
    a.x = __shfl(a.x, srcLane, width);
    a.y = __shfl(a.y, srcLane, width);
    return *reinterpret_cast<double*>(&a);
}

__device__ inline
float warpReduceSum(float val, int width=warpSize) {
    for (int offset = width/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset,width);
    return val;
}

}
