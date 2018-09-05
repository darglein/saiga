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
#include <thrust/system/cuda/experimental/pinned_allocator.h>

using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;

template<int K>
__global__ static
void randomAccessSimple(ArrayView<int> data)
{
    ThreadInfo<> ti;
    if(ti.thread_id >= data.size()) return;

    for(int i = 0; i < K; ++i)
        data[ (ti.thread_id + i) % data.size()]++;
}

template<typename T>
using pinnend_vector=thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T> >;

static void asyncTest()
{
    size_t N = 1024 * 1024 * 1;
    size_t size = N * sizeof(int);
    const int K = 512 * 20;


    pinnend_vector<int> h1(N,0);
    pinnend_vector<int> h2(N,5);

//    thrust::host_vector<int> h1(N,0);
//    thrust::host_vector<int> h2(N,5);

    thrust::host_vector<int> href1(N,0+K);
    thrust::host_vector<int> href2(N,5+K);

    thrust::device_vector<int> d1(N);
    thrust::device_vector<int> d2(N);

    cudaStream_t s1, s2;
       cudaStreamCreate(&s1); cudaStreamCreate(&s2);

//    s1 = cudaStreamLegacy;

    const int BLOCK_SIZE = 128;
    const int BLOCKS = Saiga::CUDA::getBlockCount(N/16,BLOCK_SIZE);

    cudaMemcpyAsync(d1.data().get(),h1.data(),size,cudaMemcpyHostToDevice,s1);
    cudaMemcpyAsync(d2.data().get(),h2.data(),size,cudaMemcpyHostToDevice,s2);

    randomAccessSimple<K><<<BLOCKS,BLOCK_SIZE,0,s1>>>(d1);
    randomAccessSimple<K><<<BLOCKS,BLOCK_SIZE,0,s2>>>(d2);

    cudaMemcpyAsync(h1.data(),d1.data().get(),size,cudaMemcpyDeviceToHost,s1);
    cudaMemcpyAsync(h2.data(),d2.data().get(),size,cudaMemcpyDeviceToHost,s2);

    CUDA_SYNC_CHECK_ERROR();

//    SAIGA_ASSERT(href1 == h1);
//    SAIGA_ASSERT(href2 == h2);
}



int main(int argc, char *argv[])
{
    asyncTest();
    cout << "Done." << endl;
}

