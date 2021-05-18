/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/core/time/timer.h"

#include <algorithm>

namespace Saiga
{
namespace CUDA
{
// nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr
// random.cu

template <int BLOCK_SIZE>
__global__ static void curand_setup_kernel(ArrayView<curandState> state, unsigned long long seed)
{
    ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto tid = ti.thread_id; tid < state.size(); tid += ti.grid_size)
    {
        /* Each thread gets same seed, a different sequence
       number, no offset */
        curandState localState = state[tid];
        curand_init(seed, tid % 997, tid / 997 * 10, &localState);
        //        curand_init(seed, tid, 0, &localState);
        state[tid] = localState;
    }
}

template <int BLOCK_SIZE, int N>
__global__ static void random_test(ArrayView<curandState> state, ArrayView<float> out)
{
    ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto tid = ti.thread_id; tid < state.size(); tid += ti.grid_size)
    {
        curandState localState = state[tid];
        float f                = 0;
        for (int i = 0; i < N; ++i)
        {
            f += linearRand(0, 1, localState);
        }
        out[tid] = f / N;
        //        state[tid] = localState;
    }
}


void initRandom(ArrayView<curandState> states, unsigned long long seed)
{
    const size_t THREADS_PER_BLOCK = 256;
    const size_t NUM_BLOCKS        = getBlockCount(states.size(), THREADS_PER_BLOCK);
    curand_setup_kernel<THREADS_PER_BLOCK><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(states, seed);
    CUDA_SYNC_CHECK_ERROR();
}

void randomTest()
{
    CUDA_SYNC_CHECK_ERROR();

    size_t N          = 1000 * 1000;
    size_t readWrites = N * sizeof(curandState);

    CUDA::PerformanceTestHelper pth("curand", readWrites);
    std::cout << "sizeof(curandState)=" << sizeof(curandState) << std::endl;

    thrust::device_vector<curandState> states(N);


    {
        float time;
        {
            CUDA::ScopedTimer t(time);
            initRandom(states, 495673603);
        }
        pth.addMeassurement("initRandom", time);
    }

    {
        thrust::device_vector<float> outF(states.size(), 0);
        float time;
        {
            CUDA::ScopedTimer t(time);
            const size_t THREADS_PER_BLOCK = 256;
            const size_t NUM_BLOCKS        = getBlockCount(states.size(), THREADS_PER_BLOCK);
            random_test<THREADS_PER_BLOCK, 1><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(states, outF);
        }
        pth.addMeassurement("1 random number per thread", time);
    }

    {
        thrust::device_vector<float> outF(states.size(), 0);
        float time;
        {
            CUDA::ScopedTimer t(time);
            const size_t THREADS_PER_BLOCK = 256;
            const size_t NUM_BLOCKS        = getBlockCount(states.size(), THREADS_PER_BLOCK);
            random_test<THREADS_PER_BLOCK, 10><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(states, outF);
        }
        pth.addMeassurement("10 random numbers per thread", time);
    }

    {
        thrust::device_vector<float> outF(states.size(), 0);
        float time;
        {
            CUDA::ScopedTimer t(time);
            const size_t THREADS_PER_BLOCK = 256;
            const size_t NUM_BLOCKS        = getBlockCount(states.size(), THREADS_PER_BLOCK);
            random_test<THREADS_PER_BLOCK, 100><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(states, outF);
        }
        pth.addMeassurement("100 random numbers per thread", time);
    }

    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
