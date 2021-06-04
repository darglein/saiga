/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "cudaHelper.h"

namespace Saiga
{
namespace CUDA
{
/**
 *  Copy a large datatype T with vector instructions.
 * Example: if you have a 32 byte datatype and use int4 as a vector type
 * the compiler generates two 16 byte load instructions, instead of potentially
 * eight 4 byte loads.
 */
template <typename T, typename VECTOR_TYPE = int4>
__device__ inline void vectorCopy(const T* source, T* dest)
{
    static_assert(sizeof(T) % sizeof(VECTOR_TYPE) == 0, "Incompatible types.");
    const VECTOR_TYPE* source4 = reinterpret_cast<const VECTOR_TYPE*>(source);
    VECTOR_TYPE* dest4         = reinterpret_cast<VECTOR_TYPE*>(dest);
#pragma unroll
    for (int i = 0; i < sizeof(T) / sizeof(VECTOR_TYPE); ++i)
    {
        dest4[i] = source4[i];
    }
}


/**
 * Copy an array of small T's with vector instructions.
 */
template <typename T, typename VECTOR_TYPE = int4>
__device__ inline void vectorArrayCopy(const T* source, T* dest)
{
    static_assert(sizeof(VECTOR_TYPE) % sizeof(T) == 0, "Wrong use of this function.");
    static_assert(sizeof(VECTOR_TYPE) >= sizeof(T), "Wrong use of this function.");
    reinterpret_cast<VECTOR_TYPE*>(dest)[0] = reinterpret_cast<const VECTOR_TYPE*>(source)[0];
}

// enum CacheLoadModifier
//{
//    LOAD_DEFAULT,       ///< Default (no modifier)
//    LOAD_CA,            ///< Cache at all levels
//    LOAD_CG,            ///< Cache at global level
//    LOAD_CS,            ///< Cache streaming (likely to be accessed once)
//    LOAD_CV,            ///< Cache as volatile (including cached system lines)
//    LOAD_LDG,           ///< Cache as texture
//    LOAD_VOLATILE,      ///< Volatile (any memory space)
//};

//_CUB_LOAD_ALL(LOAD_CA, ca)
//_CUB_LOAD_ALL(LOAD_CG, cg)
//_CUB_LOAD_ALL(LOAD_CS, cs)
//_CUB_LOAD_ALL(LOAD_CV, cv)
//_CUB_LOAD_ALL(LOAD_LDG, global.nc)

//#define _CUB_LOAD_4(cub_modifier, ptx_modifier)

/**
 * Cache at global level (cache in L2 and below, not L1).
 * Use ld.cg to cache loads only globally, bypassing the L1 cache, and cache only in the L2 cache.
 *
 * Read more at: http://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators
 * Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
 */
__device__ __forceinline__ unsigned int loadNoL1Cache4(unsigned int const* ptr)
{
#if !defined(WIN32)
    unsigned int retval;
    asm volatile("ld.cg.u32 %0, [%1];" : "=r"(retval) : "l"(ptr));
    return retval;
#else
    return *ptr;
#endif
}


__device__ __forceinline__ uint2 loadNoL1Cache8(uint2 const* ptr)
{
#if !defined(WIN32)
    uint2 retval;
    asm volatile("ld.cg.v2.u32 {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(ptr));
    return retval;
#else
    return *ptr;
#endif
}

__device__ __forceinline__ uint4 loadNoL1Cache16(uint4 const* ptr)
{
#if !defined(WIN32)
    uint4 retval;
    asm volatile("ld.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w)
                 : "l"(ptr));
    return retval;
#else
    return *ptr;
#endif
}

template <typename T>
__device__ inline T loadNoL1Cache(T const* ptr)
{
    T t;
    if (sizeof(T) == 4)
        reinterpret_cast<unsigned int*>(&t)[0] = loadNoL1Cache4(reinterpret_cast<unsigned int const*>(ptr));
    if (sizeof(T) == 8) reinterpret_cast<uint2*>(&t)[0] = loadNoL1Cache8(reinterpret_cast<uint2 const*>(ptr));
    if (sizeof(T) == 16) reinterpret_cast<uint4*>(&t)[0] = loadNoL1Cache16(reinterpret_cast<uint4 const*>(ptr));
    return t;
}



// B.10. Read-Only Data Cache Load Function
template <typename T>
__device__ inline T ldg(const T* address)
{
#if __CUDA_ARCH__ >= 350
    // The read-only data cache load function is only supported by devices of compute capability 3.5 and higher.
    return __ldg(address);
#else
    return *address;
#endif
}

}  // namespace CUDA
}  // namespace Saiga
