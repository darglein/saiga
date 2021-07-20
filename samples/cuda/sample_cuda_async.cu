/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include <thrust/device_vector.h>
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/event.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/stream.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include <iostream>
#include <vector>

using namespace Saiga;

using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;

//#define LECTURE

template <int K>
class Element
{
   public:
    vec4 data;

    HD inline void operator()()
    {
        for (int k = 0; k < K * 512; ++k)
        {
            data = data * 3.1f + data;
        }
    }
};


template <typename T>
__global__ static void process(ArrayView<T> data)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;

    T e = data[ti.thread_id];
    e();
    data[ti.thread_id] = e;
}

#ifdef LECTURE

template <int K>
static void uploadProcessDownloadAsync(int N)
{
    using T = Element<K>;

    thrust::host_vector<T> h_data(N);
    thrust::device_vector<T> d_data(N);

    {
        Saiga::CUDA::CudaScopedTimerPrint timer("process");
        // Compute launch arguments
        const unsigned int BLOCK_SIZE = 128;
        const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(N, BLOCK_SIZE);

        cudaMemcpy(d_data.data().get(), h_data.data(), N * sizeof(T), cudaMemcpyHostToDevice);
        process<T><<<BLOCKS, BLOCK_SIZE, 0>>>(d_data);
        cudaMemcpy(h_data.data(), d_data.data().get(), N * sizeof(T), cudaMemcpyDeviceToHost);
    }
}

int main(int argc, char* argv[])
{
    uploadProcessDownloadAsync<8>(1024 * 1024);
    cout << "Done." << endl;
}
#else

template <int K>
static void uploadProcessDownloadAsync(int N, int slices, int streamCount)
{
    using T = Element<K>;

    Saiga::pinned_vector<T> h_data(N);
    //        thrust::host_vector<T> h_data(N);
    thrust::device_vector<T> d_data(N);
//    size_t size = N * sizeof(T);



    SAIGA_ASSERT(N % slices == 0);
    int sliceN       = N / slices;
    size_t slizeSize = sliceN * sizeof(T);

    // Create a separate stream for each slice for maximum parallelism
    std::vector<Saiga::CUDA::CudaStream> streams(streamCount);

    {
        // ArrayViews simplify slice creation
        ArrayView<T> vd(d_data);
        ArrayView<T> vh(h_data);



        Saiga::CUDA::ScopedTimerPrint tim("uploadProcessDownloadAsync " + std::to_string(slices));

        for (int i = 0; i < slices; ++i)
        {
            // Pick current stream and slice
            auto& stream = streams[i % streamCount];
            auto d_slice = vd.slice_n(i * sliceN, sliceN);
            auto h_slice = vh.slice_n(i * sliceN, sliceN);

            // Compute launch arguments
            const unsigned int BLOCK_SIZE = 128;
            const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(sliceN, BLOCK_SIZE);

            cudaMemcpyAsync(d_slice.data(), h_slice.data(), slizeSize, cudaMemcpyHostToDevice, stream);
            process<T><<<BLOCKS, BLOCK_SIZE, 0, stream>>>(d_slice);
            cudaMemcpyAsync(h_slice.data(), d_slice.data(), slizeSize, cudaMemcpyDeviceToHost, stream);
        }
    }
}

int main(int argc, char* argv[])
{
    Saiga::CUDA::initCUDA();
    uploadProcessDownloadAsync<8>(1024 * 1024, 1, 1);
    uploadProcessDownloadAsync<8>(1024 * 1024, 2, 2);
    uploadProcessDownloadAsync<8>(1024 * 1024, 4, 4);
    uploadProcessDownloadAsync<8>(1024 * 1024, 8, 8);
    uploadProcessDownloadAsync<8>(1024 * 1024, 16, 16);
    uploadProcessDownloadAsync<8>(1024 * 1024, 64, 8);
    uploadProcessDownloadAsync<8>(1024 * 1024, 64, 64);

    std::cout << "Done." << std::endl;
}

#endif
