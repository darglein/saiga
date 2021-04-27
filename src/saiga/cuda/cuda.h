/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

//#include <cuda.h>
#include "saiga/core/util/assert.h"

#include <cuda_runtime.h>
#define SAIGA_CUDA_INCLUDED


#if !defined(__host__)
#    define __host__
#endif
#if !defined(__device__)
#    define __device__
#endif
#if !defined(__launch_bounds__)
#    define __launch_bounds__
#endif



#define CHECK_CUDA_ERROR(cudaFunction)                                                                              \
    {                                                                                                               \
        cudaError_t cudaErrorCode = cudaFunction;                                                                   \
        ((cudaErrorCode == cudaSuccess)                                                                             \
             ? static_cast<void>(0)                                                                                 \
             : Saiga::saiga_assert_fail(#cudaFunction " == cudaSuccess", __FILE__, __LINE__, SAIGA_ASSERT_FUNCTION, \
                                        cudaGetErrorString(cudaErrorCode)));                                        \
    }

#if defined(CUDA_DEBUG)
#    define CUDA_SYNC_CHECK_ERROR()                    \
        {                                              \
            CHECK_CUDA_ERROR(cudaDeviceSynchronize()); \
        }
#else
#    define CUDA_SYNC_CHECK_ERROR() (static_cast<void>(0))
#endif
