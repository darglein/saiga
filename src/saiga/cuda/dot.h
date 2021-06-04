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

/**
 * Simple Dot Product.
 * Tested for float and double types.
 *
 * Same performance as cublasSdot and cublasDdot
 *
 * The performance is limited by loading all elements from global memory.
 * It is not possible to go faster on current hardware.
 *
 */

namespace Saiga
{
namespace CUDA
{
template <typename T, unsigned int BLOCK_SIZE>
__device__ inline T dotLocalVector(ArrayView<T> v1, ArrayView<T> v2)
{
    T sum          = T(0);
    unsigned int N = v1.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;

    if (sizeof(T) == 4)
    {
        // use vectorized loads for 4 byte types like int/float
        // we could also use the 16 byte load by using int4 below, but
        // this isn't faster than the 8 byte load
        using vector_type                      = int2;
        const unsigned int elements_per_vector = sizeof(vector_type) / sizeof(T);
        for (auto i = ti.thread_id; i < N / elements_per_vector; i += ti.grid_size)
        {
            T locals1[elements_per_vector];
            T locals2[elements_per_vector];
            vectorArrayCopy<T, vector_type>(v1.data() + (i * elements_per_vector), locals1);
            vectorArrayCopy<T, vector_type>(v2.data() + (i * elements_per_vector), locals2);
#pragma unroll
            for (auto i = 0; i < elements_per_vector; ++i) sum += locals1[i] * locals2[i];
        }
        // process remaining elements
        for (auto i = ti.thread_id + N / elements_per_vector * elements_per_vector; i < N; i += ti.grid_size)
        {
            sum += v1[i] * v2[i];
        }
    }
    else
    {
        for (auto i = ti.thread_id; i < N; i += ti.grid_size)
        {
            sum += v1[i] * v2[i];
        }
    }
    return sum;
}



template <typename T, unsigned int BLOCK_SIZE>
__global__ void dot(ArrayView<T> v1, ArrayView<T> v2, T* out)
{
    __shared__ T shared[BLOCK_SIZE / SAIGA_WARP_SIZE];


    T sum = dotLocalVector<T, BLOCK_SIZE>(v1, v2);

    sum = blockReduceSum<T, BLOCK_SIZE>(sum, shared);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

}  // namespace CUDA
}  // namespace Saiga
