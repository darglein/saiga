/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/shfl_helper.h"


namespace Saiga {
namespace CUDA{

template<typename T, unsigned int LOCAL_WARP_SIZE=32, bool RESULT_FOR_ALL_THREADS=false, typename ShuffleType = T>
__device__ inline
T warpReduceSum(T val) {
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE/2; offset > 0; offset /= 2){
//        auto v = RESULT_FOR_ALL_THREADS ? __shfl_xor(val, offset) : __shfl_down(val, offset);
        auto v = RESULT_FOR_ALL_THREADS ? shfl_xor<T,ShuffleType>(val, offset) : shfl_down<T,ShuffleType>(val, offset);
        val = val + v;
    }
    return val;
}

template<typename T, unsigned int LOCAL_WARP_SIZE=32, bool RESULT_FOR_ALL_THREADS=false>
__device__ inline
T warpReduceMax(T val) {
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE/2; offset > 0; offset /= 2){
        auto v = RESULT_FOR_ALL_THREADS ? __shfl_xor(val, offset) : __shfl_down(val, offset);
        val = max(val , v);
    }
    return val;
}




template<typename T, unsigned int BLOCK_SIZE>
__device__ inline
T blockReduceSum(T val, T* shared) {
    int lane = threadIdx.x & (WARP_SIZE-1);
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane==0) shared[wid]=val;

    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum<T,BLOCK_SIZE/WARP_SIZE>(val); //Final reduce within first warp
    return val;
}

template<typename T, unsigned int BLOCK_SIZE>
__device__ inline
T blockReduceAtomicSum(T val, T* shared) {
    int lane = threadIdx.x & (WARP_SIZE-1);

    if(threadIdx.x == 0)
        shared[0] = T(0);

    __syncthreads();

    val = warpReduceSum(val);

    if (lane==0){
        atomicAdd(&shared[0],val);
    }

    __syncthreads();


    if(threadIdx.x == 0)
        val = shared[0];

    return val;
}

}
}
