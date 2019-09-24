/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#define SAIGA_ARRAY_VIEW_THRUST

#include "saiga/config.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/assert.h"
#include "saiga/cuda/cuda.h"
#include "saiga/cuda/cudaTimer.h"
#include "saiga/cuda/thrust_helper.h"



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



namespace Saiga
{
namespace CUDA
{
template <typename T1, typename T2>
HD SAIGA_CONSTEXPR T1 getBlockCount(T1 problemSize, T2 threadCount)
{
    return (problemSize + (threadCount - T2(1))) / (threadCount);
}


inline void initCUDA() {}


inline void destroyCUDA()
{
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    CHECK_CUDA_ERROR(cudaDeviceReset());
}

inline void printCUDAInfo()
{
    int devID;
    cudaGetDevice(&devID);

    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, devID));
    /* Statistics about the GPU device */
    std::cout << "Device Properties for CUDA device: " << devID << std::endl;
    std::cout << "  Device name: " << deviceProp.name << std::endl;
    std::cout << "  Compute capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "  Global Memory: " << deviceProp.totalGlobalMem << std::endl;
    std::cout << "  Shared Memory Per Block: " << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "  Shared Memory Per SM: " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    std::cout << "  Constant Memory: " << deviceProp.totalConstMem << std::endl;
    std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads per Multi Processor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  32-Bit Registers per Block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "  32-Bit Registers per SM: " << deviceProp.regsPerMultiprocessor << std::endl;
    std::cout << "  L2 cache size: " << deviceProp.l2CacheSize << std::endl;
    std::cout << "  Multi-Processors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
    std::cout << "  Memory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;

    // In this calculation, we convert the memory clock rate to Hz,
    // multiply it by the interface width (divided by 8, to convert bits to bytes)
    // and multiply by 2 due to the double data rate. Finally, we divide by 109 to convert the result to GB/s.
    double clockRateHz = deviceProp.memoryClockRate * 1000.0;
    std::cout << "  Theoretical Memory Bandwidth (GB/s): "
              << 2.0 * clockRateHz * (deviceProp.memoryBusWidth / 8) / 1.0e9 << std::endl;


    std::cout << "  32-Bit Registers per Thread (100% Occ): "
              << deviceProp.regsPerBlock / deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Shared Memory per Thread (100% Occ): "
              << deviceProp.sharedMemPerBlock / deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  32-Bit Shared Memory elements per Thread (100% Occ): "
              << deviceProp.sharedMemPerBlock / deviceProp.maxThreadsPerMultiProcessor / 4 << std::endl;

    std::cout << std::endl;
}

}  // namespace CUDA
}  // namespace Saiga


#define THREAD_BLOCK(_problemSize, _threadCount) Saiga::CUDA::getBlockCount(_problemSize, _threadCount), _threadCount
