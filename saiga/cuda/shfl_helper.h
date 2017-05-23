#pragma once


#include "saiga/cuda/device_helper.h"

__device__ inline
double fetch_double(uint2 p){
    return __hiloint2double(p.y, p.x);
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300
//shfl for 64 bit datatypes is already defined in sm_30_intrinsics.h
#else
__device__ inline
double __shfl_down(double var, unsigned int srcLane, int width=32) {
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return 0;
    return *reinterpret_cast<double*>(&a);
}
#endif


namespace CUDA{



template<typename T, typename ShuffleType = int>
__device__ inline
T shfl(T var, unsigned int srcLane, int width=WARP_SIZE) {
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for(int i = 0 ; i < sizeof(T) / sizeof(ShuffleType) ; ++i){
        a[i] = __shfl(a[i], srcLane, width);
    }
    return var;
}

}

