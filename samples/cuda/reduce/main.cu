/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <vector>
#include "saiga/util/glm.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/reduce.h"



using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;


template<typename T>
__device__ inline
T warpReduceSum(T val)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        auto v = Saiga::CUDA::shfl_down(val, offset);
        val = val + v;
    }
    return val;
}


template<typename T>
__global__ static
void warpReduceSimple(ArrayView<T> data, ArrayView<T> output)
{
    ThreadInfo<> ti;
    if(ti.thread_id >= data.size()) return;

    auto v = data[ti.thread_id];

    v = warpReduceSum(v);

    if(ti.thread_id == 0)
        output[0] = v;

}

template<typename T>
__device__ inline
T blockReduceAtomicSum(T val, T* shared)
{
    int lane = threadIdx.x & (WARP_SIZE-1);

    // Each warp reduces with registers
    val = warpReduceSum(val);

    // Init shared memory
    if(threadIdx.x == 0)
        shared[0] = T(0);

    __syncthreads();


    // The first thread in each warp writes to smem
    if (lane==0){
        atomicAdd(&shared[0],val);
    }

    __syncthreads();

    // The first thread in this block has the result
    // Optional: remove if so that every thread has the result
    if(threadIdx.x == 0)
        val = shared[0];

    return val;
}


template<typename T>
__global__ static
void blockReduceSimple(ArrayView<T> data, ArrayView<T> output)
{
    ThreadInfo<> ti;
    if(ti.thread_id >= data.size()) return;

    __shared__ T blockSum;

    auto v = data[ti.thread_id];

    v = blockReduceAtomicSum(v,&blockSum);

    if(ti.local_thread_id == 0)
        output[0] = v;

}

template<typename T>
__global__ static
void globalReduceSimple(ArrayView<T> data, ArrayView<T> output)
{
    ThreadInfo<> ti;

    __shared__ T blockSum;

    // All threads needs to participate
    // -> reduce a 0 for out-of-bounds threads
    auto v = ti.thread_id >= data.size() ? 0 : data[ti.thread_id];

    v = blockReduceAtomicSum(v,&blockSum);

    if(ti.local_thread_id == 0)
        atomicAdd(output.data(),v);

}

static void reduceTest()
{
    int N = 823674;
    using T = int;
    Saiga::thrust::pinned_vector<T> h_data(N);

    for(auto& f : h_data)
    {
        f = rand() % 10;
    }

    thrust::device_vector<T> d_data = h_data;
    thrust::device_vector<T> output(1);

    {
        int n = 32;
        thrust::device_vector<T>  data(n);
        thrust::copy(d_data.begin(),d_data.begin()+n,data.begin());
        warpReduceSimple<T><<<1,n>>>(data,output);
        T res = output[0];
        T tres = thrust::reduce(data.begin(),data.end());
        cout << res << " " << tres << endl;
        SAIGA_ASSERT(res == tres);
    }

    {
        int n = 256;
        thrust::device_vector<T>  data(n);
        thrust::copy(d_data.begin(),d_data.begin()+n,data.begin());
        blockReduceSimple<T><<<1,n>>>(data,output);
        T res = output[0];
        T tres = thrust::reduce(data.begin(),data.end());
        cout << res << " " << tres << endl;
        SAIGA_ASSERT(res == tres);
    }

    {
        output[0] = 0;
        globalReduceSimple<T><<<THREAD_BLOCK(N,128)>>>(d_data,output);
        T res = output[0];
        T tres = thrust::reduce(d_data.begin(),d_data.end());
        cout << res << " " << tres << endl;
        SAIGA_ASSERT(res == tres);
    }

}

int main(int argc, char *argv[])
{
    reduceTest();

    cout << "Done." << endl;
}

