/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/shfl_helper.h"


namespace Saiga
{
namespace CUDA
{
/**
 * Usage:
 *
 * float value = ...;
 * float sum = warpReduceSum(value);
 *
 * if(ti.lane_id == 0)
 *     output[ti.warp_id] = sum;
 *
 */
template <typename T, unsigned int LOCAL_WARP_SIZE = 32>
__device__ inline T warpReduceSum(T val)
{
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = shfl_xor<T>(val, offset);
        val    = val + v;
    }
    return val;
}

template <typename T, typename OP, unsigned int LOCAL_WARP_SIZE = 32>
__device__ inline T warpReduce(T val, OP op)
{
#pragma unroll
    for (int offset = LOCAL_WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        auto v = shfl_xor(val, offset);
        val    = op(val, v);
    }
    return val;
}


template <typename T, unsigned int BLOCK_SIZE>
__device__ inline T blockReduceSum(T val, T* shared, T default_value)
{
    int lane = threadIdx.x & (SAIGA_WARP_SIZE - 1);
    int wid  = threadIdx.x / SAIGA_WARP_SIZE;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / SAIGA_WARP_SIZE) ? shared[lane] : default_value;

    if (wid == 0) val = warpReduceSum<T, BLOCK_SIZE / SAIGA_WARP_SIZE>(val);  // Final reduce within first warp
    return val;
}

template <typename T, unsigned int BLOCK_SIZE>
__device__ inline T blockReduceAtomicSum(T val, T* shared)
{
    int lane = threadIdx.x & (SAIGA_WARP_SIZE - 1);

    // Each warp reduces with registers
    val = warpReduceSum(val);

    // Init shared memory
    if (threadIdx.x == 0) shared[0] = T(0);

    __syncthreads();


    // The first thread in each warp writes to smem
    if (lane == 0)
    {
        atomicAdd(&shared[0], val);
    }

    __syncthreads();

    // The first thread in this block has the result
    // Optional: remove if so that every thread has the result
    if (threadIdx.x == 0) val = shared[0];

    return val;
}

// ===============
// More general reductions with a custom OP



template <int BLOCK_SIZE, typename T, typename OP>
__device__ inline T blockReduce(T val, OP op, T default_val)
{
    __shared__ T shared[BLOCK_SIZE / 32];

    int lane   = threadIdx.x % 32;
    int warpid = threadIdx.x / 32;

    // Each warp reduces with registers
    val = warpReduce(val, op);

    // The first thread in each warp writes to smem
    if (lane == 0)
    {
        shared[warpid] = val;
    }

    __syncthreads();

    if (warpid == 0)
    {
        if (threadIdx.x < BLOCK_SIZE / 32)
        {
            val = shared[threadIdx.x];
        }
        else
        {
            val = default_val;
        }


        val = warpReduce(val, op);
    }


    return val;
}

}  // namespace CUDA
}  // namespace Saiga
