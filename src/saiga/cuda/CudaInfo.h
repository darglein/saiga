/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/cuda/cudaHelper.h"

#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/table.h"


namespace Saiga
{
namespace CUDA
{

inline void PrintMemoryInfo()
{
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);
    double mem_free_gb  = mem_free / (1000.0 * 1000.0 * 1000.0);
    double mem_total_gb = mem_total / (1000.0 * 1000.0 * 1000.0);
    double mem_alloc_gb = mem_total_gb - mem_free_gb;
    int device;
    cudaGetDevice(&device);
    std::cout << "CUDA Used Memory (Device " << device << "): " << mem_alloc_gb << " / " << mem_total_gb << " GB"
              << std::endl;
}

inline double UsedMemoryGB()
{
    size_t mem_free, mem_total;
    cudaMemGetInfo(&mem_free, &mem_total);
    double mem_free_gb  = mem_free / (1000.0 * 1000.0 * 1000.0);
    double mem_total_gb = mem_total / (1000.0 * 1000.0 * 1000.0);
    double mem_alloc_gb = mem_total_gb - mem_free_gb;
    return mem_alloc_gb;
}

inline void initCUDA(int device_id = 0)
{
    int runtimeVersion;
    {
        cudaError_t cudaErrorCode = cudaRuntimeGetVersion(&runtimeVersion);
        if (cudaErrorCode != cudaSuccess)
        {
            std::cout << "Invalid CUDA Runtime!" << std::endl;
            std::cout << "Please install a CUDA cabable Graphics Driver and restart the computer!" << std::endl;
            throw std::runtime_error(cudaGetErrorString(cudaErrorCode));
            //exit(1);
        }
    }

    int driverVersion;
    {
        cudaError_t cudaErrorCode = cudaDriverGetVersion(&driverVersion);
        if (cudaErrorCode != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(cudaErrorCode));
        }
    }

#ifdef CUDA_DEBUG
    bool cudadebug = true;
#else
    bool cudadebug = false;
#endif


    cudaDeviceProp deviceProp;
    {
        cudaError_t cudaErrorCode = cudaGetDeviceProperties(&deviceProp, device_id);
        if (cudaErrorCode != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(cudaErrorCode));
        }
    }

    std::cout << ConsoleColor::GREEN;
    Table table({2, 24, 32, 1});
    std::cout << "======================= CUDA Init =======================" << std::endl;
    table <<"|" << "Runtime Version" << runtimeVersion << "|";
    table <<"|" << "Driver Version" << driverVersion <<  "|";
    table <<"|" << "CUDA_DEBUG" << cudadebug <<  "|";
    table <<"|" << "Device name" << deviceProp.name <<  "|";
    table <<"|" << "Compute capabilities" << std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor) <<  "|";
    table <<"|" << "Global Memory" << deviceProp.totalGlobalMem <<  "|";
    std::cout << "=========================================================" << std::endl;

    std::cout.unsetf(std::ios_base::floatfield);
    std::cout << ConsoleColor::RESET;
}


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
