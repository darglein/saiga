#pragma once

#include "cudaHelper.h"
#include "saiga/util/assert.h"


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
//atomicAdd is already defined for compute capability 6.x and higher.
#else
#if 0
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
#else
__device__ inline
double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif
#endif


//CUDA_ASSERT

#if defined(CUDA_DEBUG)


__device__ inline
void cuda_assert_fail (const char *__assertion, const char *__file,
               unsigned int __line, const char *__function){
    printf("Assertion '%s' failed!\n"
           "  File: %s:%d\n"
           "  Function: %s\n"
           "  Thread: %d,%d,%d\n"
           "  Block: %d, %d, %d\n",
           __assertion, __file, __line, __function,
           threadIdx.x,threadIdx.y,threadIdx.z,
           blockIdx.x,blockIdx.y,blockIdx.z);
    //provoke a segfault
     *(int*)0 = 0;
}

# define CUDA_ASSERT(expr)							\
  ((expr)								\
   ? static_cast<void>(0)						\
   : cuda_assert_fail (#expr, __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION))

#else

# define CUDA_ASSERT(expr)		( static_cast<void>(0))

#endif


