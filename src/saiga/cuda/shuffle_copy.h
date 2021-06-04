/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "shfl_helper.h"

namespace Saiga
{
namespace CUDA
{
/**
 *
 * Example Ussage:
 *
 * //Global array of particles with size = 8 * float
 * Saiga::ArrayView<Particle> particles
 *
 * Saiga::CUDA::ThreadInfo<0,2> ti;
 *
 * //Do not return with thread id because we need the threads for loading
 * if (ti.warp_id*2 >= particles.size())
 *       return;
 *
 * //Load to registers
 * Particle particle;
 * loadShuffleStruct<2,Particle,float4>(particles.data(),&particle,ti.lane_id,ti.warp_id,particles.size());
 *
 * //do something with particle
 *
 * //Store in global memory
 * storeShuffleStruct<2,Particle,float4>(particles.data(),&particle,ti.lane_id,ti.warp_id,particles.size());
 *
 */

template <int G, int SIZE, typename VectorType = int4>
__device__ inline void loadShuffle(VectorType* globalStart, VectorType* localStart, int lane, int globalOffset,
                                   int Nvectors)
{
    static_assert(SIZE % sizeof(VectorType) == 0, "Type must be loadable by vector type.");

    int bytesPerCycle = G * sizeof(VectorType);
    //    int cycles = SIZE / bytesPerCycle;
    int cycles = getBlockCount(SIZE, bytesPerCycle);
    //    int vectorsPerCycle = bytesPerCycle / sizeof(VectorType);
    int vectorsPerElement = SIZE / sizeof(VectorType);

    //    printf("bytesPerCycle %d, cycles %d, vectorsPerElement %d\n",bytesPerCycle,cycles,vectorsPerElement);


    VectorType* global = reinterpret_cast<VectorType*>(globalStart);
    VectorType* local  = reinterpret_cast<VectorType*>(localStart);

    VectorType l[G];
    VectorType tmp;

    for (int g = 0; g < G; ++g)
    {
        for (int c = 0; c < cycles; ++c)
        {
            auto localIdx  = lane + c * G;
            auto globalIdx = globalOffset + localIdx + g * vectorsPerElement;

            if (globalIdx < Nvectors && localIdx < vectorsPerElement)
            {
                //                printf("read %d, %d \n",globalIdx,lane);
                tmp = global[globalIdx];
            }

            // broadcast loaded value to all threads in this warp
            for (int s = 0; s < G; ++s)
            {
                l[s] = shfl(tmp, s, G);
            }

            // this thread now has the correct value
            if (lane == g)
            {
                for (int s = 0; s < G; ++s)
                {
                    local[s + c * G] = l[s];
                }
            }
        }
    }
}


template <int G, int SIZE, typename VectorType = int4>
__device__ inline void storeShuffle(VectorType* globalStart, VectorType* localStart, int lane, int globalOffset,
                                    int Nvectors)
{
    static_assert(SIZE % sizeof(VectorType) == 0, "Type must be loadable by vector type.");

    int bytesPerCycle = G * sizeof(VectorType);
    //    int cycles = SIZE / bytesPerCycle;
    int cycles = getBlockCount(SIZE, bytesPerCycle);
    //    int vectorsPerCycle = bytesPerCycle / sizeof(VectorType);
    int vectorsPerElement = SIZE / sizeof(VectorType);

    VectorType* global = reinterpret_cast<VectorType*>(globalStart);
    VectorType* local  = reinterpret_cast<VectorType*>(localStart);


    VectorType l[G];
    VectorType tmp;

    for (int g = 0; g < G; ++g)
    {
        for (int c = 0; c < cycles; ++c)
        {
            // this thread now has the correct value
            if (lane == g)
            {
                for (int s = 0; s < G; ++s)
                {
                    l[s] = local[s + c * G];
                }
            }

            // broadcast loaded value to all threads in this warp
            for (int s = 0; s < G; ++s)
            {
                l[s] = shfl(l[s], g, G);
            }

            // this loop does the same as "tmp = l[lane]", but with static indexing
            // so all the locals can be held in registers
            // https://stackoverflow.com/questions/44117704/why-is-local-memory-used-in-this-simple-loop
            for (int i = 0; i < G; ++i)
            {
                if (i <= lane) tmp = l[i];
            }

            //            auto globalIdx = globalOffset + lane + c * G + g * vectorsPerElement;

            auto localIdx  = lane + c * G;
            auto globalIdx = globalOffset + localIdx + g * vectorsPerElement;

            if (globalIdx < Nvectors && localIdx < vectorsPerElement) global[globalIdx] = tmp;
        }
    }
}



template <int LOCAL_WARP_SIZE, typename T, typename VectorType = int4>
__device__ inline void loadShuffleStruct(T* globalStart, T* localStart, int laneId, int warpId, int count)
{
    static_assert(sizeof(T) % sizeof(VectorType) == 0, "Type must be loadable by vector type.");
    const int vectors_per_element = sizeof(T) / sizeof(VectorType);
    loadShuffle<LOCAL_WARP_SIZE, sizeof(T), VectorType>(
        reinterpret_cast<VectorType*>(globalStart), reinterpret_cast<VectorType*>(localStart), laneId,
        warpId * LOCAL_WARP_SIZE * vectors_per_element, count * vectors_per_element);
}

template <int LOCAL_WARP_SIZE, typename T, typename VectorType = int4>
__device__ inline void storeShuffleStruct(T* globalStart, T* localStart, int laneId, int warpId, int count)
{
    static_assert(sizeof(T) % sizeof(VectorType) == 0, "Type must be loadable by vector type.");
    const int vectors_per_element = sizeof(T) / sizeof(VectorType);
    storeShuffle<LOCAL_WARP_SIZE, sizeof(T), VectorType>(
        reinterpret_cast<VectorType*>(globalStart), reinterpret_cast<VectorType*>(localStart), laneId,
        warpId * LOCAL_WARP_SIZE * vectors_per_element, count * vectors_per_element);
}

}  // namespace CUDA
}  // namespace Saiga
