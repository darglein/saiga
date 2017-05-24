#pragma once
#include "saiga/config.h"

//uncomment to remove all CUDA_SYNC_CHECK_ERROR
#define CUDA_DEBUG



#include "saiga/cuda/array_view.h"
#include "saiga/cuda/cudaTimer.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "saiga/util/assert.h"

#include "thrust_helper.h"





namespace CUDA {


# define CHECK_CUDA_ERROR(cudaFunction) {							\
  cudaError_t  cudaErrorCode = cudaFunction;                                                       \
  ((cudaErrorCode == cudaSuccess)								\
   ? static_cast<void>(0)						\
   : saiga_assert_fail (#cudaFunction " == cudaSuccess", __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION,cudaGetErrorString(cudaErrorCode))); \
}

#if defined(CUDA_DEBUG)
# define CUDA_SYNC_CHECK_ERROR() { CHECK_CUDA_ERROR(cudaDeviceSynchronize()); }
#else
# define CUDA_SYNC_CHECK_ERROR()		( static_cast<void>(0))
#endif


template<typename T1, typename T2>
__host__ __device__ constexpr
T1 getBlockCount(T1 problemSize, T2 threadCount){
    return ( problemSize + (threadCount - T2(1)) ) / (threadCount);
}




#define THREAD_BLOCK(_problemSize, _threadCount) getBlockCount(_problemSize,_threadCount) , _threadCount






SAIGA_GLOBAL extern void initCUDA();
SAIGA_GLOBAL extern void destroyCUDA();
SAIGA_GLOBAL extern void runTests();

//defined in reduce_test.cu
SAIGA_GLOBAL extern void reduceTest();

}
