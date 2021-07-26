/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/core/math/math.h"


//#define LECTURE

using Saiga::ArrayView;

#define WEIGHT_SIZE 4096
#define WEIGHT_ADD_COUNT 128

#ifdef LECTURE

__global__ static void addWeightBase(ArrayView<int> src, ArrayView<int> dst, ArrayView<int> weight)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= dst.size()) return;

    int sum = 0;

    for (int i = 0; i < WEIGHT_ADD_COUNT; ++i)
    {
        int loadIndex = (i) % WEIGHT_SIZE;
        //        auto we = weight[loadIndex];
        auto we = Saiga::CUDA::ldg(&weight[loadIndex]);
        sum += we;
    }

    dst[ti.thread_id] = src[ti.thread_id] * sum;
}

#else

static __constant__ int cweights[WEIGHT_SIZE];


template <bool USE_CONSTANT, bool BROADCAST, bool LDG>
__global__ static void addWeight(ArrayView<int> src, ArrayView<int> dst, ArrayView<int> weight)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= dst.size()) return;
    int* w = USE_CONSTANT ? cweights : weight.data();

    int sum = 0;

    for (int i = 0; i < WEIGHT_ADD_COUNT; ++i)
    {
        int loadIndex = BROADCAST ? (i) % WEIGHT_SIZE : (i * 32 + ti.local_thread_id) % WEIGHT_SIZE;
        auto we       = LDG ? Saiga::CUDA::ldg(w + loadIndex) : w[loadIndex];
        sum += we;
    }

    dst[ti.thread_id] = src[ti.thread_id] * sum;
}

template <bool BROADCAST>
__global__ static void addWeightShared(ArrayView<int> src, ArrayView<int> dst, ArrayView<int> weight)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= dst.size()) return;

    __shared__ int dataShared[WEIGHT_SIZE];

    for (int i = ti.local_thread_id; i < WEIGHT_SIZE; i += ti.threads_per_block)
    {
        dataShared[i] = weight[i];
    }

    __syncthreads();

    int sum = 0;

    for (int i = 0; i < WEIGHT_ADD_COUNT; ++i)
    {
        int loadIndex = BROADCAST ? (i) % WEIGHT_SIZE : (i * 32 + ti.local_thread_id) % WEIGHT_SIZE;
        auto we       = dataShared[loadIndex];
        sum += we;
    }

    dst[ti.thread_id] = src[ti.thread_id] * sum;
}

#endif


static void constantTest()
{
    size_t N          = 10 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int) + N * WEIGHT_ADD_COUNT * sizeof(int);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);
    thrust::device_vector<int> weight(WEIGHT_SIZE);



    const int BLOCK_SIZE = 128;
    const int BLOCKS     = Saiga::CUDA::getBlockCount(N, BLOCK_SIZE);


    Saiga::CUDA::PerformanceTestHelper pth("Memcpy", readWrites);
    int its = 10;

#ifdef LECTURE
    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
            its, [&]() { addWeightBase<<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeightBase", st.median);
    }
#else

    cudaMemcpyToSymbol(cweights, weight.data().get(), weight.size() * sizeof(int), cudaMemcpyDeviceToDevice);
    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeight<false, true, false><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight global broadcast", st.median);
    }
    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeight<false, true, true><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight global broadcast ldg", st.median);
    }
    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeight<false, false, false><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight global coalesced", st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeight<false, false, true><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight global coalesced ldg", st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeightShared<false><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight shared coalesced", st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeightShared<true><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight shared broadcast", st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeight<true, true, false><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight constant broadcast", st.median);
    }
    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { addWeight<true, false, false><<<BLOCKS, BLOCK_SIZE>>>(src, dest, weight); });
        pth.addMeassurement("addWeight constant coalesced", st.median);
    }
#endif


    CUDA_SYNC_CHECK_ERROR();
}

int main(int argc, char* argv[])
{
    constantTest();
    return 0;
}
