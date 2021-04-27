/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/imath.h"
#include "saiga/core/util/assert.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/thread_info.h"

#if !defined(IS_CUDA)
#    error device_helper.h must only be included by nvcc
// A few defintions to trick IDE's that do not activley support CUDA
dim3 threadIdx;
dim3 blockIdx;
dim3 blockDim;
dim3 gridDim;
int warpSize;
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// atomicAdd is already defined for compute capability 6.x and higher.
#else
#    if 0
__device__ inline
double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
        return __longlong_as_double(old);
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#    else
__device__ inline double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old             = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old     = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#    endif
#endif


// CUDA_ASSERT

#if defined(CUDA_DEBUG)


namespace Saiga
{
namespace CUDA
{
__device__ inline void cuda_assert_fail(const char* __assertion, const char* __file, unsigned int __line,
                                        const char* __function)
{
    printf(
        "Assertion '%s' failed!\n"
        "  File: %s:%d\n"
        "  Function: %s\n"
        "  Thread: %d,%d,%d\n"
        "  Block: %d, %d, %d\n",
        __assertion, __file, __line, __function, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y,
        blockIdx.z);
    // provoke a segfault
    *(int*)0 = 0;
}
}  // namespace CUDA
}  // namespace Saiga

#    define CUDA_ASSERT(expr)          \
        ((expr) ? static_cast<void>(0) \
                : Saiga::CUDA::cuda_assert_fail(#expr, __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION))

#else

#    define CUDA_ASSERT(expr) (static_cast<void>(0))

#endif


#define WARP_FOR_NO_IF(_variableName, _initExpr, _length, _step)                                  \
    for (unsigned int _variableName = _initExpr, _k = 0; _k < Saiga::iDivUp<int>(_length, _step); \
         _k++, _variableName += _step)

#define WARP_FOR(_variableName, _initExpr, _length, _step)   \
    WARP_FOR_NO_IF(_variableName, _initExpr, _length, _step) \
    if (_variableName < _length)
