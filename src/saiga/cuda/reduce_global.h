/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/memory.h"
#include "saiga/cuda/reduce.h"
#include "saiga/cuda/shfl_helper.h"
#include "saiga/cuda/thread_info.h"

namespace Saiga
{
namespace CUDA
{
template <typename T, unsigned int BLOCK_SIZE>
__device__ inline T reduceLocalVector(ArrayView<T> in)
{
    T sum          = T(0);
    unsigned int N = in.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;

    if (sizeof(T) == 4)
    {
        // use vectorized loads for 4 byte types like int/float
        // we could also use the 16 byte load by using int4 below, but
        // this isn't faster than the 8 byte load
        using vector_type                      = int2;
        const unsigned int elements_per_vector = sizeof(vector_type) / sizeof(T);
        //        int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
        for (auto i = ti.thread_id; i < N / elements_per_vector; i += ti.grid_size)
        {
            T locals[elements_per_vector];
            vectorArrayCopy<T, vector_type>(in.data() + (i * elements_per_vector), locals);
#pragma unroll
            for (auto i = 0; i < elements_per_vector; ++i) sum += locals[i];
        }
        // process remaining elements
        for (auto i = ti.thread_id + N / elements_per_vector * elements_per_vector; i < N; i += ti.grid_size)
        {
            sum += in[i];
        }
    }
    else
    {
        for (auto i = ti.thread_id; i < in.size(); i += ti.grid_size)
        {
            sum += in[i];
        }
    }
    return sum;
}



template <typename T, unsigned int BLOCK_SIZE>
__global__ void reduceBlockShared(ArrayView<T> in, T* out)
{
    __shared__ T shared[BLOCK_SIZE / SAIGA_WARP_SIZE];

    T sum = reduceLocalVector<T, BLOCK_SIZE>(in);
    sum   = blockReduceSum<T, BLOCK_SIZE>(sum, shared);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

template <typename T, unsigned int BLOCK_SIZE>
__global__ void reduceBlockSharedAtomic(ArrayView<T> in, T* out)
{
    __shared__ T shared;
    T sum = reduceLocalVector<T, BLOCK_SIZE>(in);
    sum   = blockReduceAtomicSum<T, BLOCK_SIZE>(sum, &shared);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}


template <typename T, unsigned int BLOCK_SIZE>
__global__ void reduceAtomic(ArrayView<T> in, T* out)
{
    T sum    = reduceLocalVector<T, BLOCK_SIZE>(in);
    sum      = warpReduceSum(sum);
    int lane = threadIdx.x & (SAIGA_WARP_SIZE - 1);
    if (lane == 0) atomicAdd(out, sum);
}

}  // namespace CUDA
}  // namespace Saiga
