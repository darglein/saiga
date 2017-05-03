#pragma once

#include "saiga/cuda/device_helper.h"


namespace CUDA{


template<typename T, int LOCAL_WARP_SIZE=32>
__inline__ __device__
T warpReduceSum(T val) {
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}


template<typename T, int BLOCK_SIZE>
__inline__ __device__
T blockReduceSum(T val) {
    static __shared__ T shared[BLOCK_SIZE/WARP_SIZE];
    int lane = threadIdx.x & (WARP_SIZE-1);
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);

    if (lane==0) shared[wid]=val;

    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / WARP_SIZE) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum<T,BLOCK_SIZE/WARP_SIZE>(val); //Final reduce within first warp
    return val;
}

template<typename T, int BLOCK_SIZE>
__inline__ __device__
T blockReduceAtomicSum(T val) {

    static __shared__ T shared;
    int lane = threadIdx.x & (WARP_SIZE-1);

    if(threadIdx.x == 0)
        shared = T(0);

    __syncthreads();

    val = warpReduceSum(val);

    if (lane==0){
        atomicAdd(&shared,val);
    }

    __syncthreads();


    if(threadIdx.x == 0)
        val = shared;

    return val;
}

template<typename T, int BLOCK_SIZE>
__global__
void reduceBlockShared(array_view<T> in, T* out) {
    T sum = T(0);
    for(int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < in.size(); i += BLOCK_SIZE * gridDim.x){
        sum += in[i];
    }
    sum = blockReduceSum<T,BLOCK_SIZE>(sum);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

template<typename T, int BLOCK_SIZE>
__global__
void reduceBlockSharedAtomic(array_view<T> in, T* out) {
    T sum = T(0);
    for(int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < in.size(); i += BLOCK_SIZE * gridDim.x){
        sum += in[i];
    }
    sum = blockReduceAtomicSum<T,BLOCK_SIZE>(sum);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

template<typename T, int BLOCK_SIZE>
__global__
void reduceAtomic(array_view<T> in, T* out) {
    T sum = T(0);
    for(int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < in.size(); i += BLOCK_SIZE * gridDim.x){
        sum += in[i];
    }
    sum = warpReduceSum(sum);


    int lane = threadIdx.x & (WARP_SIZE-1);
    if(lane == 0)
        atomicAdd(out, sum);
}


//defined in reduce_test.cu
SAIGA_GLOBAL extern void reduceTest();

}
