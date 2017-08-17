/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/config.h"

//remove all CUDA_SYNC_CHECK_ERROR and CUDA_ASSERTS
//for gcc add cppflag: -DCUDA_NDEBUG
#ifndef CUDA_NDEBUG
#define CUDA_DEBUG
#else
#undef CUDA_DEBUG
#endif

#include "saiga/cuda/array_view.h"
#include "saiga/cuda/cudaTimer.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "saiga/util/assert.h"

#include "thrust_helper.h"


# define CHECK_CUDA_ERROR(cudaFunction) {							\
  cudaError_t  cudaErrorCode = cudaFunction;                                                       \
  ((cudaErrorCode == cudaSuccess)								\
   ? static_cast<void>(0)						\
   : Saiga::saiga_assert_fail (#cudaFunction " == cudaSuccess", __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION,cudaGetErrorString(cudaErrorCode))); \
}

#if defined(CUDA_DEBUG)
# define CUDA_SYNC_CHECK_ERROR() { CHECK_CUDA_ERROR(cudaDeviceSynchronize()); }
#else
# define CUDA_SYNC_CHECK_ERROR()		( static_cast<void>(0))
#endif



namespace Saiga {
namespace CUDA {

template<typename T1, typename T2>
__host__ __device__ constexpr
T1 getBlockCount(T1 problemSize, T2 threadCount){
    return ( problemSize + (threadCount - T2(1)) ) / (threadCount);
}


HD constexpr inline
//int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivUp(int a, int b) { return (a + b - 1) / b; }

HD constexpr inline
int iDivDown(int a, int b) { return a / b; }

HD constexpr inline
int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }

HD constexpr inline
int iAlignDown(int a, int b) {return a - a % b; }

HD inline
int iFloor(float value){
    int i = (int)value;
    return i - (i > value);
}

HD inline
int iCeil(float value){
    int i = (int)value;
    return i + (i < value);
}

HD inline
int iRound(float value){
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}



SAIGA_GLOBAL extern void initCUDA();
SAIGA_GLOBAL extern void destroyCUDA();

}
}


#define THREAD_BLOCK(_problemSize, _threadCount) Saiga::CUDA::getBlockCount(_problemSize,_threadCount) , _threadCount
