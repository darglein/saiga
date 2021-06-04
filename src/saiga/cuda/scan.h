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
template <typename T, unsigned int LOCAL_WARP_SIZE = 32>
__device__ inline T warpInclusiveScan(T val, unsigned int lane)
{
#pragma unroll
    for (int d = 1; d < LOCAL_WARP_SIZE; d *= 2)
    {
        T tmp = shfl_up(val, d, LOCAL_WARP_SIZE);
        if (lane >= d) val += tmp;
    }
    return val;
}


template <unsigned int BLOCK_SIZE, typename T>
__device__ inline T blockInclusiveScan(const CUDA::ThreadInfo<BLOCK_SIZE>& ti, T val, T* shared, T* sharedBlockPrefix)
{
    T prefix = *sharedBlockPrefix;
    if (ti.local_thread_id == 0) val += prefix;

    val = warpInclusiveScan(val, ti.lane_id);

    // the last thread in the warp writes its value to shared memory
    if (ti.lane_id == 31) shared[ti.warp_lane] = val;
    __syncthreads();


    if (ti.local_thread_id < BLOCK_SIZE / SAIGA_WARP_SIZE)
    {
        // Scan in first warp
        T valWarp          = shared[ti.lane_id];
        valWarp            = warpInclusiveScan<T, BLOCK_SIZE / SAIGA_WARP_SIZE>(valWarp, ti.lane_id);
        shared[ti.lane_id] = valWarp;
        if (ti.lane_id == BLOCK_SIZE / SAIGA_WARP_SIZE - 1)
        {
            *sharedBlockPrefix = valWarp;
        }
    }

    __syncthreads();

    // add value of previous warp
    if (ti.warp_lane > 0)
    {
        val += shared[ti.warp_lane - 1];
    }

    // I'm not sure why i need this __syncthreads. On the GTX760M it works without it, but on
    // GTX970 it doesn't.
    __syncthreads();

    return val;
}


template <unsigned int BLOCK_SIZE, typename T>
__device__ inline T blockInclusiveScan(T val, T* shared, T* sharedBlockPrefix)
{
    auto local_thread_id = threadIdx.x;
    auto lane_id         = local_thread_id & (SAIGA_WARP_SIZE - 1);
    auto warp_lane       = local_thread_id / SAIGA_WARP_SIZE;

    T prefix = *sharedBlockPrefix;
    if (local_thread_id == 0) val += prefix;

    val = warpInclusiveScan(val, lane_id);

    // the last thread in the warp writes its value to shared memory
    if (lane_id == 31) shared[warp_lane] = val;
    __syncthreads();


    if (local_thread_id < BLOCK_SIZE / SAIGA_WARP_SIZE)
    {
        // Scan in first warp
        T valWarp       = shared[lane_id];
        valWarp         = warpInclusiveScan<T, BLOCK_SIZE / SAIGA_WARP_SIZE>(valWarp, lane_id);
        shared[lane_id] = valWarp;
        if (lane_id == BLOCK_SIZE / SAIGA_WARP_SIZE - 1)
        {
            *sharedBlockPrefix = valWarp;
        }
    }

    __syncthreads();

    // add value of previous warp
    if (warp_lane > 0)
    {
        val += shared[warp_lane - 1];
    }

    // I'm not sure why i need this __syncthreads. On the GTX760M it works without it, but on
    // GTX970 it doesn't.
    __syncthreads();

    return val;
}


#define SCAN_BLOCK_PREFIX 0x00
#define SCAN_BLOCK_AGGREGATE 0x01

template <bool EXCLUSIVE_SCAN = true, unsigned int BLOCK_SIZE = 256, unsigned int TILES_PER_BLOCK = 8,
          typename vector_type = int4, bool CHECK_BOUNDS = true>
__global__ __launch_bounds__(BLOCK_SIZE) void tiledSinglePassScan(ArrayView<unsigned int> in,
                                                                  ArrayView<unsigned int> out,
                                                                  ArrayView<unsigned int> aggregate)
{
    const unsigned int ELEMENTS_PER_VECTOR = sizeof(vector_type) / sizeof(unsigned int);
    const unsigned int ELEMENTS_PER_TILE   = BLOCK_SIZE * ELEMENTS_PER_VECTOR;
    const unsigned int ELEMENTS_PER_BLOCK  = TILES_PER_BLOCK * ELEMENTS_PER_TILE;

    const unsigned int N         = in.size();
    const unsigned int numBlocks = getBlockCount(N, ELEMENTS_PER_BLOCK);
    const unsigned int numTiles  = getBlockCount(N, ELEMENTS_PER_TILE);

    __shared__ int orderedBlockId;
    __shared__ unsigned int blockExclusive;
    __shared__ unsigned int currentTilePrefix;
    __shared__ unsigned int shared[BLOCK_SIZE / SAIGA_WARP_SIZE];

    // this alone requires 8 * 5 * 256 = 10240 registers per block
    unsigned int elementsLocal[TILES_PER_BLOCK][ELEMENTS_PER_VECTOR + 1];

    CUDA::ThreadInfo<BLOCK_SIZE> ti;


    if (ti.local_thread_id == 0)
    {
        // we use the last element of the aggregate array as a global counter starting from -1
        orderedBlockId    = atomicAdd(&aggregate[numBlocks], 1) + 1;
        currentTilePrefix = 0;
        blockExclusive    = 0;
    }

    __syncthreads();

    auto block = orderedBlockId;

    auto tileIdStart = block * TILES_PER_BLOCK;


    // each block processes multiple consecutive tiles
#pragma unroll
    for (int i = 0; i < TILES_PER_BLOCK; ++i)
    {
        auto tileId = tileIdStart + i;

        if (CHECK_BOUNDS && tileId >= numTiles) break;

        auto localIndex = ELEMENTS_PER_TILE * tileId + ti.local_thread_id * ELEMENTS_PER_VECTOR;

        auto& val = elementsLocal[i][ELEMENTS_PER_VECTOR];
        val       = 0;

        if (CHECK_BOUNDS)
        {
            if (localIndex >= N)
            {
                // set vector to 0
#pragma unroll
                for (int z = 0; z < ELEMENTS_PER_VECTOR; ++z)
                {
                    elementsLocal[i][z] = 0;
                }
            }
            else
            {
                if (localIndex + ELEMENTS_PER_VECTOR >= N)
                {
#pragma unroll
                    for (int z = 0; z < ELEMENTS_PER_VECTOR; ++z)
                    {
                        auto idx            = localIndex + z;
                        elementsLocal[i][z] = (idx < N) ? in[idx] : 0;
                    }
                }
                else
                {
                    vectorArrayCopy<unsigned int, vector_type>(in.data() + (localIndex), &elementsLocal[i][0]);
                }
            }
        }
        else
        {
            vectorArrayCopy<unsigned int, vector_type>(in.data() + (localIndex), &elementsLocal[i][0]);
        }


        // compute the sum of all elements in this vector
#pragma unroll
        for (int k = 0; k < ELEMENTS_PER_VECTOR; ++k)
        {
            val += elementsLocal[i][k];
        }

        // scan the local sums of this tile
        val = blockInclusiveScan<BLOCK_SIZE>(ti, val, shared, &currentTilePrefix);
        //        val = blockInclusiveScan<BLOCK_SIZE>(val,shared,&currentTilePrefix);
    }

    unsigned int global_exclusive_prefix(0);

    if (ti.local_thread_id == 0)
    {
        // the inclusive prefix of that block is still stored in shared memory
        unsigned int blockSum = shared[BLOCK_SIZE / SAIGA_WARP_SIZE - 1];


        aggregate[block] = (((blockSum) & ((1 << 30) - 1)) | (1 << 30));
        if (block == 0)
        {
            aggregate[0]            = blockSum;
            global_exclusive_prefix = 0;
        }
        else
        {
            int current_pred        = block - 1;
            global_exclusive_prefix = 0;
            while (current_pred >= 0)
            {
#if 0
                //In fact L1 cache is disabled by default for global memory reads,
                //if this changes use this special function
                unsigned int blockData = loadNoL1Cache(aggregate.data()+current_pred);
                //                unsigned int blockData = atomicCAS(aggregate.data()+current_pred,0,0);
#else
                unsigned int blockData = aggregate[current_pred];
#endif

                unsigned int offsetData = (blockData) & ((1 << 30) - 1);
                bool prefixAvailable    = blockData >> 30 == SCAN_BLOCK_PREFIX;
                bool aggregateAvailable = blockData >> 30 == SCAN_BLOCK_AGGREGATE;

                if (prefixAvailable)
                {
                    global_exclusive_prefix += offsetData;
                    break;
                }

                if (aggregateAvailable)
                {
                    global_exclusive_prefix += offsetData;
                    if (current_pred-- == 0) break;
                }
            }

            aggregate[block] = ((blockSum + global_exclusive_prefix) & ((1 << 30) - 1));
            blockExclusive   = global_exclusive_prefix;
        }
    }


    __syncthreads();


    global_exclusive_prefix = blockExclusive;

#pragma unroll
    for (int i = 0; i < TILES_PER_BLOCK; ++i)
    {
        int tileId = tileIdStart + i;

        if (CHECK_BOUNDS && tileId >= numTiles) break;

        auto localIndex = ELEMENTS_PER_TILE * tileId + ti.local_thread_id * ELEMENTS_PER_VECTOR;

        // add the global offset to the local vector sum
        elementsLocal[i][ELEMENTS_PER_VECTOR] += global_exclusive_prefix;

        // finally compute the prefix sum of each vector from right to left
#pragma unroll
        for (int k = ELEMENTS_PER_VECTOR - 1; k >= 0; --k)
        {
            if (EXCLUSIVE_SCAN)
            {
                elementsLocal[i][ELEMENTS_PER_VECTOR] -= elementsLocal[i][k];
                elementsLocal[i][k] = elementsLocal[i][ELEMENTS_PER_VECTOR];
            }
            else
            {
                auto tmp            = elementsLocal[i][k];
                elementsLocal[i][k] = elementsLocal[i][ELEMENTS_PER_VECTOR];
                elementsLocal[i][ELEMENTS_PER_VECTOR] -= tmp;
            }
        }


        if (CHECK_BOUNDS)
        {
            if (localIndex >= N)
            {
                // the complete vector is out of bounds
            }
            else
            {
                if (localIndex + ELEMENTS_PER_VECTOR >= N)
                {
                    // parts of the vector are out of bounds
                    // copy elements one by one
#pragma unroll
                    for (int z = 0; z < ELEMENTS_PER_VECTOR; ++z)
                    {
                        auto idx = localIndex + z;
                        if (idx < N)
                        {
                            out[idx] = elementsLocal[i][z];
                        }
                    }
                }
                else
                {
                    // complete vector is inside
                    vectorArrayCopy<unsigned int, vector_type>(&elementsLocal[i][0], out.data() + (localIndex));
                }
            }
        }
        else
        {
            // no bounds check
            vectorArrayCopy<unsigned int, vector_type>(&elementsLocal[i][0], out.data() + (localIndex));
        }
    }
}

}  // namespace CUDA
}  // namespace Saiga
