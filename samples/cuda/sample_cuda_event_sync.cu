/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/event.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/stream.h"
#include "saiga/core/math/math.h"

#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;



template <int K>
__global__ static void A(ArrayView<int> data)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;
    for (int i = 0; i < K; ++i) atomicAdd(&data[ti.thread_id], 1);
}

template <int K>
__global__ static void B(ArrayView<int> data)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;
    for (int i = 0; i < K; ++i) atomicAdd(&data[ti.thread_id], 1);
}

template <int K>
__global__ static void C(ArrayView<int> data)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;
    for (int i = 0; i < K; ++i) atomicAdd(&data[ti.thread_id], 1);
}

template <int K>
__global__ static void D(ArrayView<int> data)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;
    for (int i = 0; i < K; ++i) atomicAdd(&data[ti.thread_id], 1);
}

static void uploadProcessDownloadAsync(int N)
{
    thrust::device_vector<int> d_data(N);
    thrust::device_vector<int> d_data3(N);


    // Compute launch arguments
    const unsigned int BLOCK_SIZE = 128;
    const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(N, BLOCK_SIZE);

    Saiga::CUDA::CudaStream stream1, stream2;

    {
#ifdef LECTURE
        CUDA_SYNC_CHECK_ERROR();

        A<1024 * 2><<<BLOCKS, BLOCK_SIZE, 0, stream1>>>(d_data);
        B<1024><<<BLOCKS, BLOCK_SIZE, 0, stream1>>>(d_data);

        C<1024><<<BLOCKS, BLOCK_SIZE, 0, stream2>>>(d_data3);
        D<1024><<<BLOCKS, BLOCK_SIZE, 0, stream2>>>(d_data);

        CUDA_SYNC_CHECK_ERROR();
#else
        Saiga::CUDA::CudaEvent event;

        A<1024 * 2><<<BLOCKS, BLOCK_SIZE, 0, stream1>>>(d_data);
        event.record(stream1);
        B<1024><<<BLOCKS, BLOCK_SIZE, 0, stream1>>>(d_data);

        C<1024><<<BLOCKS, BLOCK_SIZE, 0, stream2>>>(d_data3);
        stream2.waitForEvent(event);
        D<1024><<<BLOCKS, BLOCK_SIZE, 0, stream2>>>(d_data);

        CUDA_SYNC_CHECK_ERROR();
#endif
    }
}

int main(int argc, char* argv[])
{
    uploadProcessDownloadAsync(1024 * 8);

    std::cout << "Done." << std::endl;
}
