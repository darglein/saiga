/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/core/time/timer.h"

namespace Saiga
{
namespace CUDA
{
template <unsigned int THREADS_PER_BLOCK, typename vector_type = int2>
__global__ static void copy(void* src, void* dest, unsigned int size)
{
    vector_type* src2  = reinterpret_cast<vector_type*>(src);
    vector_type* dest2 = reinterpret_cast<vector_type*>(dest);

    // This copy kernel is only correct when size%sizeof(vector_type)==0
    auto numElements = size / sizeof(vector_type);

    for (auto id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; id < numElements; id += gridDim.x * THREADS_PER_BLOCK)
    {
        dest2[id] = src2[id];
    }
}


template <unsigned int ELEMENTS_PER_THREAD, unsigned int THREADS_PER_BLOCK>
__global__ static void copyFixed(void* src, void* dest, unsigned int size)
{
    using vector_type  = int2;
    vector_type* src2  = reinterpret_cast<vector_type*>(src);
    vector_type* dest2 = reinterpret_cast<vector_type*>(dest);

    // This copy kernel is only correct when size%(sizeof(vector_type)*ELEMENTS_PER_THREAD)==0
    auto numElements = size / sizeof(vector_type);

    auto tid = ELEMENTS_PER_THREAD * THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;

    if (tid >= numElements) return;

#pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        auto id   = tid + THREADS_PER_BLOCK * i;
        dest2[id] = src2[id];
    }
}



// nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr
// integrate_test.cu


template <typename vectorT>
void occupancyTest2()
{
    size_t N          = 100 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int);

    CUDA::PerformanceTestHelper pth(
        "Occupancy for bandwidth limited kernels Vector size: " + std::to_string(sizeof(vectorT)), readWrites);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);



    const size_t THREADS_PER_BLOCK = 128;
    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, 0));
    //    auto sharedMemMP = deviceProp.sharedMemPerMultiprocessor;
    auto sharedMem = deviceProp.sharedMemPerBlock;

    auto threads                     = deviceProp.maxThreadsPerMultiProcessor;
    auto sharedMemAllocationUnitSize = 256;
    int maxBlocks                    = threads / THREADS_PER_BLOCK;

    {
        for (int i = 1; i <= maxBlocks; ++i)
        {
            auto blocksPerSM     = i;
            auto targetSharedMem = sharedMem / blocksPerSM;
            // floor to next 256 byte boundary
            auto res        = targetSharedMem % sharedMemAllocationUnitSize;
            targetSharedMem = targetSharedMem - res;

            //            std::cout << blocksPerSM << " " << targetSharedMem << " = " << sharedMem << " / " <<
            //            blocksPerSM << std::endl;
            float time;
            const size_t NUM_BLOCKS = CUDA::getBlockCount(N * sizeof(int) / sizeof(vectorT), THREADS_PER_BLOCK);
            {
                CUDA::ScopedTimer t(time);
                copy<THREADS_PER_BLOCK, vectorT><<<NUM_BLOCKS, THREADS_PER_BLOCK, targetSharedMem>>>(
                    thrust::raw_pointer_cast(src.data()), thrust::raw_pointer_cast(dest.data()), N * sizeof(int));
            }
            float occupancy = (float(blocksPerSM * THREADS_PER_BLOCK) / threads) * 100;
            pth.addMeassurement("Memcpy Occupancy: " + std::to_string(occupancy) + "%)", time);
        }
    }



    {
        float time;
        {
            CUDA::ScopedTimer t(time);
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()), thrust::raw_pointer_cast(src.data()), N * sizeof(int),
                       cudaMemcpyDeviceToDevice);
        }
        pth.addMeassurement("cudaMemcpy", time);
    }
}

void occupancyTest()
{
    CUDA_SYNC_CHECK_ERROR();
    occupancyTest2<int>();
    occupancyTest2<int2>();
    occupancyTest2<int4>();
    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
