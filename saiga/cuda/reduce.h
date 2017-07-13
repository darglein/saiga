#pragma once

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/shfl_helper.h"
#include "saiga/cuda/memory.h"

namespace CUDA{



template<typename T, unsigned int LOCAL_WARP_SIZE=32, bool RESULT_FOR_ALL_THREADS=false>
__device__ inline
T warpReduceSum(T val) {
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE/2; offset > 0; offset /= 2){
        auto v = RESULT_FOR_ALL_THREADS ? __shfl_xor(val, offset) : __shfl_down(val, offset);
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





template<typename T, unsigned int BLOCK_SIZE>
__device__ inline
T reduceLocalVector(array_view<T> in){
    T sum = T(0);
    unsigned int N = in.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;

    if(sizeof(T) == 4){
        //use vectorized loads for 4 byte types like int/float
        //we could also use the 16 byte load by using int4 below, but
        //this isn't faster than the 8 byte load
        using vector_type = int2;
        const unsigned int elements_per_vector = sizeof(vector_type) / sizeof(T);
//        int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        for(auto i = ti.thread_id; i < N / elements_per_vector; i += ti.grid_size){
            T locals[elements_per_vector];
            vectorArrayCopy<T,vector_type>( in.data() + (i*elements_per_vector) , locals);
#pragma unroll
            for(auto i = 0 ; i < elements_per_vector; ++i)
                sum += locals[i];
        }
        //process remaining elements
        for(auto i = ti.thread_id + N/elements_per_vector * elements_per_vector; i<N; i += ti.grid_size){
            sum += in[i];
        }
    }else{
        for(auto i = ti.thread_id; i < in.size(); i += ti.grid_size){
            sum += in[i];
        }
    }
    return sum;
}



template<typename T, unsigned int BLOCK_SIZE>
__global__
void reduceBlockShared(array_view<T> in, T* out) {
    __shared__ T shared[BLOCK_SIZE/WARP_SIZE];

    T sum = reduceLocalVector<T,BLOCK_SIZE>(in);
    sum = blockReduceSum<T,BLOCK_SIZE>(sum,shared);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}

template<typename T, unsigned int BLOCK_SIZE>
__global__
void reduceBlockSharedAtomic(array_view<T> in, T* out) {
    __shared__ T shared;
    T sum = reduceLocalVector<T,BLOCK_SIZE>(in);
    sum = blockReduceAtomicSum<T,BLOCK_SIZE>(sum,&shared);
    if (threadIdx.x == 0)
        atomicAdd(out, sum);
}


template<typename T, unsigned int BLOCK_SIZE>
__global__
void reduceAtomic(array_view<T> in, T* out) {
    T sum = reduceLocalVector<T,BLOCK_SIZE>(in);
    sum = warpReduceSum(sum);
    int lane = threadIdx.x & (WARP_SIZE-1);
    if(lane == 0)
        atomicAdd(out, sum);
}




}
