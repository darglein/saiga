/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/reduce.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/core/math/math.h"

#include <fstream>
#include <random>

using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;

//#define LECTURE

std::ofstream outstrm;


HD inline uint32_t simpleRand(uint32_t state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}


template <typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static void randomAccessSimple(ArrayView<T> data, ArrayView<T> result)
{
    ThreadInfo<BLOCK_SIZE> ti;
    if (ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for (int i = 0; i < K; ++i)
    {
        r          = simpleRand(r);
        auto index = r % data.size();
        //        sum += data[index];
        sum += Saiga::CUDA::ldg(data.data() + index);
    }

    // Reduce the cache impact of the output array
    sum = Saiga::CUDA::warpReduceSum<T>(sum);
    if (ti.lane_id == 0) result[ti.warp_id] = sum;
}


#ifndef LECTURE

template <typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static void randomAccessConstRestricted(ArrayView<T> vdata, const T* __restrict__ data, ArrayView<T> result)
{
    ThreadInfo<BLOCK_SIZE> ti;
    if (ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for (int i = 0; i < K; ++i)
    {
        r          = simpleRand(r);
        auto index = r % vdata.size();
        sum += data[index];
    }

    // Reduce the cache impact of the output array
    sum = Saiga::CUDA::warpReduceSum<T>(sum);
    if (ti.lane_id == 0) result[ti.warp_id] = sum;
}


template <typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static void randomAccessLdg(ArrayView<T> data, ArrayView<T> result)
{
    ThreadInfo<BLOCK_SIZE> ti;
    if (ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for (int i = 0; i < K; ++i)
    {
        r          = simpleRand(r);
        auto index = r % data.size();
        sum += Saiga::CUDA::ldg(data.data() + index);
    }

    // Reduce the cache impact of the output array
    sum = Saiga::CUDA::warpReduceSum<T>(sum);
    if (ti.lane_id == 0) result[ti.warp_id] = sum;
}


static texture<int, 1, cudaReadModeElementType> dataTexture;

template <typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static void randomAccessTexture(ArrayView<T> data, ArrayView<T> result)
{
    ThreadInfo<BLOCK_SIZE> ti;
    if (ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for (int i = 0; i < K; ++i)
    {
        r          = simpleRand(r);
        auto index = r % data.size();
        sum += tex1Dfetch(dataTexture, index);
    }

    // Reduce the cache impact of the output array
    sum = Saiga::CUDA::warpReduceSum<T>(sum);
    if (ti.lane_id == 0) result[ti.warp_id] = sum;
}

#endif

template <typename ElementType>
void randomAccessTest2(int numIndices, int numElements)
{
    const int K = 16;

    outstrm << numIndices * sizeof(int) / 1024 << ",";

    size_t readWrites = numElements * sizeof(ElementType) / 32 + numElements * sizeof(int) * K;

    Saiga::CUDA::PerformanceTestHelper test("Coalesced processing test. numIndices: " + std::to_string(numIndices) +
                                                " numElements: " + std::to_string(numElements),
                                            readWrites);

    thrust::host_vector<ElementType> data(numIndices);
    thrust::host_vector<ElementType> result(numElements, 0);
    thrust::host_vector<ElementType> ref(numElements);



    thrust::device_vector<ElementType> d_data(data);
    thrust::device_vector<ElementType> d_result(result);

    int its = 50;

    const int BLOCK_SIZE = 128;
    const int BLOCKS     = Saiga::CUDA::getBlockCount(numElements, BLOCK_SIZE);

    {
        d_result = result;

        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { randomAccessSimple<ElementType, BLOCK_SIZE, K><<<BLOCKS, BLOCK_SIZE>>>(d_data, d_result); });
        test.addMeassurement("randomAccessSimple", st.median);
        outstrm << test.bandwidth(st.median) << ",";
        CUDA_SYNC_CHECK_ERROR();
    }

    //    SAIGA_ASSERT(ref == d_result);

#ifndef LECTURE
    {
        d_result = result;

        //        cudaFuncSetCacheConfig(randomAccessConstRestricted<ElementType,BLOCK_SIZE,K>,cudaFuncCachePreferShared);
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            randomAccessConstRestricted<ElementType, BLOCK_SIZE, K>
                <<<BLOCKS, BLOCK_SIZE>>>(d_data, d_data.data().get(), d_result);
        });
        test.addMeassurement("randomAccessConstRestricted", st.median);
        outstrm << test.bandwidth(st.median) << ",";

        CUDA_SYNC_CHECK_ERROR();
    }

    //    SAIGA_ASSERT(ref == d_result);

    {
        d_result = result;

        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            randomAccessLdg<ElementType, BLOCK_SIZE, K><<<BLOCKS, BLOCK_SIZE>>>(d_data, d_result);
        });
        test.addMeassurement("randomAccessLdg", st.median);
        outstrm << test.bandwidth(st.median) << ",";

        CUDA_SYNC_CHECK_ERROR();
    }

    //    SAIGA_ASSERT(ref == d_result);



#endif
    {
        cudaBindTexture(0, dataTexture, d_data.data().get(), d_data.size() * sizeof(ElementType));
        d_result = result;

        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { randomAccessTexture<ElementType, BLOCK_SIZE, K><<<BLOCKS, BLOCK_SIZE>>>(d_data, d_result); });
        test.addMeassurement("randomAccessTexture", st.median);
        outstrm << test.bandwidth(st.median);

        cudaUnbindTexture(dataTexture);
        CUDA_SYNC_CHECK_ERROR();
    }


    outstrm << std::endl;
    return;
}


int main(int argc, char* argv[])
{
    //    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    //    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    outstrm.open("out.csv");
    outstrm << "size,simple,cr,ldg,texture" << std::endl;

#ifdef LECTURE
    int start = 8;
    int end   = 9;
    randomAccessTest2<int>(1 << 12, 1 * 1024 * 1024);
#else
    int start = 8;
    int end   = 24;


    for (int i = start; i < end; ++i)
    {
        randomAccessTest2<int>(1 << i, 1 * 1024 * 1024);
        if (i > 0) randomAccessTest2<int>((1 << i) + (1 << (i - 1)), 1 * 1024 * 1024);
    }
#endif
    CUDA_SYNC_CHECK_ERROR();
    return 0;
}
