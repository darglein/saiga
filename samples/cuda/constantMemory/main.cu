/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/glm.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/tests/test_helper.h"


using Saiga::ArrayView;

#define WEIGHT_SIZE 4096
#define WEIGHT_ADD_COUNT 128

static __constant__ int cweights[WEIGHT_SIZE];


template<bool USE_CONSTANT, bool BROADCAST, bool LDG>
__global__ static
void addWeight(ArrayView<int> src, ArrayView<int> dst, ArrayView<int> weight)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= dst.size()) return;
    int* w = USE_CONSTANT ? cweights : weight.data();

    int sum = 0;

    for(int i = 0; i < WEIGHT_ADD_COUNT; ++i)
    {
        int loadIndex = BROADCAST ? (i) % WEIGHT_SIZE  : (i * 32 + ti.local_thread_id) % WEIGHT_SIZE;
        auto we = LDG ?  Saiga::CUDA::ldg(w + loadIndex) : w[loadIndex];
        sum += we;
    }

    dst[ti.thread_id] = src[ti.thread_id] * sum;
}



static
void constantTest()
{
    size_t N = 100 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int) + N * WEIGHT_ADD_COUNT * sizeof(int);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);
    thrust::device_vector<int> weight(WEIGHT_SIZE);


    cudaMemcpyToSymbol(cweights,weight.data().get(),weight.size() * sizeof(int),cudaMemcpyDeviceToDevice);

    const int BLOCK_SIZE = 128;
    const int BLOCKS = Saiga::CUDA::getBlockCount(N,BLOCK_SIZE);


    Saiga::CUDA::PerformanceTestHelper pth("Memcpy", readWrites);
    int its = 10;

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
           addWeight<false,true,false><<<BLOCKS,BLOCK_SIZE>>>(src,dest,weight);
        });
        pth.addMeassurement("addWeight global broadcast",st.median);
    }
    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
           addWeight<false,true,true><<<BLOCKS,BLOCK_SIZE>>>(src,dest,weight);
        });
        pth.addMeassurement("addWeight global broadcast ldg",st.median);
    }
    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
           addWeight<false,false,false><<<BLOCKS,BLOCK_SIZE>>>(src,dest,weight);
        });
        pth.addMeassurement("addWeight global coalesced",st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
           addWeight<false,false,true><<<BLOCKS,BLOCK_SIZE>>>(src,dest,weight);
        });
        pth.addMeassurement("addWeight global coalesced ldg",st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
           addWeight<true,true,false><<<BLOCKS,BLOCK_SIZE>>>(src,dest,weight);
        });
        pth.addMeassurement("addWeight constant broadcast",st.median);
    }
    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
           addWeight<true,false,false><<<BLOCKS,BLOCK_SIZE>>>(src,dest,weight);
        });
        pth.addMeassurement("addWeight constant coalesced",st.median);
    }


    CUDA_SYNC_CHECK_ERROR();
}

int main(int argc, char *argv[])
{

    constantTest();
    return 0;
}

