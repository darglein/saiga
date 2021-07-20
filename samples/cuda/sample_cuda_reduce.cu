/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/math.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/reduce.h"

#include <iostream>
#include <vector>
using namespace Saiga;
using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;

//#define LECTURE

#ifdef LECTURE

template <typename T>
__global__ static void warpReduceSimple(ArrayView<T> data, ArrayView<T> output)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;
}

static void reduceTest()
{
    int N   = 823674;
    using T = int;
    Saiga::pinned_vector<T> h_data(N);

    for (auto& f : h_data)
    {
        f = rand() % 10;
    }

    thrust::device_vector<T> d_data = h_data;
    thrust::device_vector<T> output(1);

    {
        int n = 32;

        // Reduce only the first n elements
        thrust::device_vector<T> data(d_data.begin(), d_data.begin() + n);
        warpReduceSimple<T><<<1, n>>>(data, output);

        // Validate output with thrust::reduce
        T res  = output[0];
        T tres = thrust::reduce(data.begin(), data.end());
        std::cout << "warpReduceSimple=" << res << ", thrust::reduce=" << tres << std::endl;
        SAIGA_ASSERT(res == tres);
    }
}

int main(int argc, char* argv[])
{
    reduceTest();
    std::cout << "Done." << std::endl;
}

#else

template <typename T>
__device__ inline T warpReduceSum(T val)
{
#    pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        auto v = Saiga::CUDA::shfl_down(val, offset);
        val    = val + v;
    }
    return val;
}


template <typename T>
__global__ static void warpReduceSimple(ArrayView<T> data, ArrayView<T> output)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;

    auto v = data[ti.thread_id];

    v = warpReduceSum(v);

    if (ti.thread_id == 0) output[0] = v;
}

template <typename T>
__device__ inline T blockReduceSum(T val, T& blockSum)
{
    int lane = threadIdx.x & (SAIGA_WARP_SIZE - 1);

    // Each warp reduces with registers
    val = warpReduceSum(val);

    // Init shared memory
    if (threadIdx.x == 0) blockSum = T(0);

    __syncthreads();


    // The first thread in each warp writes to smem
    if (lane == 0)
    {
        atomicAdd(&blockSum, val);
    }

    __syncthreads();

    // The first thread in this block has the result
    // Optional: remove if so that every thread has the result
    if (threadIdx.x == 0) val = blockSum;

    return val;
}


template <typename T>
__global__ static void blockReduceSimple(ArrayView<T> data, ArrayView<T> output)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;

    __shared__ T blockSum;

    auto v = data[ti.thread_id];

    v = blockReduceSum(v, blockSum);

    if (ti.local_thread_id == 0) output[0] = v;
}

template <typename T>
__global__ static void globalReduceSimple(ArrayView<T> data, ArrayView<T> output)
{
    ThreadInfo<> ti;

    __shared__ T blockSum;

    // All threads needs to participate
    // -> reduce a 0 for out-of-bounds threads
    auto v = ti.thread_id >= data.size() ? 0 : data[ti.thread_id];

    v = blockReduceSum(v, blockSum);

    if (ti.local_thread_id == 0) atomicAdd(output.data(), v);
}

struct Particle
{
    vec3 position;
    float radius = 0;
};

struct MaxRadius
{
    HD Particle operator()(const Particle& p1, const Particle& p2) { return p1.radius < p2.radius ? p2 : p1; }
};


static void reduceTest()
{
    int N   = 823674;
    using T = int;
    Saiga::pinned_vector<T> h_data(N);

    for (auto& f : h_data)
    {
        f = rand() % 10;
    }

    thrust::device_vector<T> d_data = h_data;
    thrust::device_vector<T> output(1);

    {
        int n = 32;

        // Reduce only the first n elements
        thrust::device_vector<T> data(d_data.begin(), d_data.begin() + n);
        warpReduceSimple<T><<<1, n>>>(data, output);

        // Validate output with thrust::reduce
        T res  = output[0];
        T tres = thrust::reduce(data.begin(), data.end());
        std::cout << "warpReduceSimple=" << res << ", thrust::reduce=" << tres << std::endl;
        SAIGA_ASSERT(res == tres);
    }

    {
        int n = 256;

        // Reduce only the first n elements
        thrust::device_vector<T> data(d_data.begin(), d_data.begin() + n);
        blockReduceSimple<T><<<1, n>>>(data, output);

        // Validate output with thrust::reduce
        T res  = output[0];
        T tres = thrust::reduce(data.begin(), data.end());
        std::cout << "blockReduceSimple=" << res << ", thrust::reduce=" << tres << std::endl;
        SAIGA_ASSERT(res == tres);
    }

    {
        // Reduce everything
        output[0] = 0;
        globalReduceSimple<T><<<THREAD_BLOCK(N, 128)>>>(d_data, output);

        // Validate output with thrust::reduce
        T res  = output[0];
        T tres = thrust::reduce(d_data.begin(), d_data.end());
        std::cout << "globalReduceSimple=" << res << ", thrust::reduce=" << tres << std::endl;
        SAIGA_ASSERT(res == tres);
    }


    {
        // thrust::reduce with a custom reduce operator
        // Here: Finding the particle with the largest radius
        thrust::device_vector<Particle> particles(100000);

        Particle test;
        test.radius    = 12314;
        particles[100] = test;

        Particle p = thrust::reduce(particles.begin(), particles.end(), Particle(), MaxRadius());
        std::cout << "Max radius = " << p.radius << std::endl;

        SAIGA_ASSERT(test.radius == p.radius);
    }
}

int main(int argc, char* argv[])
{
    reduceTest();
    std::cout << "Done." << std::endl;
}

#endif
