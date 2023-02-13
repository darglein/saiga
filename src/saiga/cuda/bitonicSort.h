/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/shfl_helper.h"
#include "saiga/cuda/thread_info.h"

namespace Saiga
{
namespace CUDA
{
// extract k-th bit from i
inline HD int bfe(int i, int k)
{
    return (i >> k) & 1;
}


template <typename T, unsigned int SIZE = 32>
inline __device__ T shuffleSwapCompare(T x, int mask, int direction)
{
    auto y = Saiga::CUDA::shfl_xor(x, mask, SIZE);
    return x < y == direction ? y : x;
}


template <unsigned int SIZE, typename T>
inline __device__ T bitonicSortStage(T v, unsigned int stage, unsigned int l)
{
#pragma unroll
    for (int i = stage; i >= 0; --i)
    {
        auto distance = 1 << i;
        unsigned int direction;


        if (1 << (stage + 1) < SIZE)
            direction = bfe(l, i) ^ bfe(l, stage + 1);
        else
            // Small optimization because bfe(l,stage+1) is always 0 here
            direction = bfe(l, i);

        v = shuffleSwapCompare(v, distance, direction);
    }
    return v;
}

template <typename T, unsigned int SIZE = 32>
inline __device__ T bitonicWarpSort(T v, unsigned int l)
{
    if (SIZE < 32) l = l & (SIZE - 1);

    int stage = 0;
#pragma unroll
    for (int i = 1; i < SIZE; i *= 2)
    {
        v = bitonicSortStage<SIZE>(v, stage, l);
        ++stage;
    }
    return v;
}


}  // namespace CUDA
}  // namespace Saiga
