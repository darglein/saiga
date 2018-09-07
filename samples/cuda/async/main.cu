/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <vector>
#include "saiga/util/glm.h"
#include <thrust/device_vector.h>
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/stream.h"
#include "saiga/cuda/event.h"

using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;



template<int K>
class GLM_ALIGN(16) Element
{
    public:
    vec4 data;

    HD inline
            void operator ()()
    {
        for(int k = 0; k < K * 512; ++k)
        {
            data = data * data + data;
        }
    }
};


template<typename T>
__global__ static
void process(ArrayView<T> data)
{
    ThreadInfo<> ti;
    if(ti.thread_id >= data.size()) return;

    T e = data[ti.thread_id];
    e();
    data[ti.thread_id] = e;
}


template<int K>
static void uploadProcessDownloadAsync(int N, int slices)
{
    using T = Element<K>;

    Saiga::thrust::pinned_vector<T> h_data(N);
//        thrust::host_vector<T> h_data(N);
    thrust::device_vector<T> d_data(N);
    size_t size = N * sizeof(T);



    SAIGA_ASSERT(N % slices == 0);
    int sliceN = N / slices;
    size_t slizeSize = sliceN * sizeof(T);

    // Create a separate stream for each slice for maximum parallelism
    std::vector<Saiga::CUDA::CudaStream> streams(slices);

    {
        // ArrayViews simplify slice creation
        ArrayView<T> vd(d_data);
        ArrayView<T> vh(h_data);



        Saiga::CUDA::CudaScopedTimerPrint tim("uploadProcessDownloadAsync " + std::to_string(slices));

        for(int i = 0; i < slices; ++i)
        {
            // Pick current stream and slice
            auto& stream = streams[i];
            auto d_slice = vd.slice_n(i * sliceN,sliceN);
            auto h_slice = vh.slice_n(i * sliceN,sliceN);

            // Compute launch arguments
            const unsigned int BLOCK_SIZE = 128;
            const unsigned int BLOCKS = Saiga::CUDA::getBlockCount(sliceN,BLOCK_SIZE);

            cudaMemcpyAsync(d_slice.data(),h_slice.data(),slizeSize,cudaMemcpyHostToDevice,stream);

            process<T><<<BLOCKS,BLOCK_SIZE,0,stream>>>(d_slice);

            cudaMemcpyAsync(h_slice.data(),d_slice.data(),slizeSize,cudaMemcpyDeviceToHost,stream);
        }
    }
}

int main(int argc, char *argv[])
{
    uploadProcessDownloadAsync  <8>(1024 * 1024,1);
    uploadProcessDownloadAsync  <8>(1024 * 1024,2);
    uploadProcessDownloadAsync  <8>(1024 * 1024,4);
    uploadProcessDownloadAsync  <8>(1024 * 1024,8);
    uploadProcessDownloadAsync  <8>(1024 * 1024,16);

    cout << "Done." << endl;
}

