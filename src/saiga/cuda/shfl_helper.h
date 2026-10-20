/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/device_helper.h"

namespace Saiga
{
namespace CUDA
{
__device__ inline double fetch_double(uint2 p)
{
    return __hiloint2double(p.y, p.x);
}



template <typename T, typename ShuffleType = int>
__device__ inline T shfl(T var, unsigned int srcLane, int width = SAIGA_WARP_SIZE, unsigned int mask = 0xFFFFFFFF)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_sync(mask, a[i], srcLane, width);
    }
    return var;
}

template <typename T, typename ShuffleType = int>
__device__ inline T shfl_down(T var, unsigned int srcLane, int width = SAIGA_WARP_SIZE)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_down_sync(0xFFFFFFFF, a[i], srcLane, width);
    }
    return var;
}

template <typename T, typename ShuffleType = int>
__device__ inline T shfl_up(T var, unsigned int srcLane, int width = SAIGA_WARP_SIZE)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_up_sync(0xFFFFFFFF, a[i], srcLane, width);
    }
    return var;
}

template <typename T, typename ShuffleType = int>
__device__ inline T shfl_xor(T var, unsigned int srcLane, int width = SAIGA_WARP_SIZE, unsigned int mask = 0xFFFFFFFF)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
        a[i] = __shfl_xor_sync(mask, a[i], srcLane, width);
    }
    return var;
}

}  // namespace CUDA
}  // namespace Saiga
