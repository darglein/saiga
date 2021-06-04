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
//#if CUDA_VERSION >= 9000
#if CUDART_VERSION >= 9000
#    define SAIGA_CUDA_USE_SHFL_SYNC
#endif

__device__ inline double fetch_double(uint2 p)
{
    return __hiloint2double(p.y, p.x);
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 300
// shfl for 64 bit datatypes is already defined in sm_30_intrinsics.h
#else
__device__ inline double __shfl_down(double var, unsigned int srcLane, int width = 32)
{
    int2 a = *reinterpret_cast<int2*>(&var);
    a.x    = __shfl_down(a.x, srcLane, width);
    a.y    = __shfl_down(a.y, srcLane, width);
    return 0;
    return *reinterpret_cast<double*>(&a);
}
#endif

template <typename T, typename ShuffleType = int>
__device__ inline T shfl(T var, unsigned int srcLane, int width = SAIGA_WARP_SIZE)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
#ifdef SAIGA_CUDA_USE_SHFL_SYNC
        a[i] = __shfl_sync(0xFFFFFFFF, a[i], srcLane, width);
#else
        a[i] = __shfl(a[i], srcLane, width);
#endif
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
#ifdef SAIGA_CUDA_USE_SHFL_SYNC
        a[i] = __shfl_down_sync(0xFFFFFFFF, a[i], srcLane, width);
#else
        a[i] = __shfl_down(a[i], srcLane, width);
#endif
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
#ifdef SAIGA_CUDA_USE_SHFL_SYNC
        a[i] = __shfl_up_sync(0xFFFFFFFF, a[i], srcLane, width);
#else
        a[i] = __shfl_up(a[i], srcLane, width);
#endif
    }
    return var;
}

template <typename T, typename ShuffleType = int>
__device__ inline T shfl_xor(T var, unsigned int srcLane, int width = SAIGA_WARP_SIZE)
{
    static_assert(sizeof(T) % sizeof(ShuffleType) == 0, "Cannot shuffle this type.");
    ShuffleType* a = reinterpret_cast<ShuffleType*>(&var);
    for (int i = 0; i < sizeof(T) / sizeof(ShuffleType); ++i)
    {
#ifdef SAIGA_CUDA_USE_SHFL_SYNC
        a[i] = __shfl_xor_sync(0xFFFFFFFF, a[i], srcLane, width);
#else
        a[i] = __shfl_xor(a[i], srcLane, width);
#endif
    }
    return var;
}

}  // namespace CUDA
}  // namespace Saiga
