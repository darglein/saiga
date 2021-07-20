/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/event.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/stream.h"
#include "saiga/core/math/math.h"

#include <iostream>
#include <vector>

#include <thrust/device_vector.h>

using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;


__device__ inline void computeBeforeSync() {}

__device__ inline void computeAfterSync(int previousBlockValue) {}


template <unsigned int BLOCK_SIZE>
__global__ static void ibs(ArrayView<int> data, ArrayView<int> blockTmp, ArrayView<unsigned int> atomics)
{
    ThreadInfo<BLOCK_SIZE> ti;
    if (ti.thread_id >= data.size()) return;

    __shared__ unsigned int blockId;
    __shared__ int previousBlockValue;

    if (ti.local_thread_id == 0)
    {
        blockId = atomicInc(&atomics[0], 0xFFFFFFFF);
    }

    __syncthreads();


    computeBeforeSync();

    if (ti.local_thread_id == 0)
    {
        // wait until previous block is finished

        // does not work
        //        while(atomics[1] != realBlockId)
        //        while(Saiga::CUDA::loadNoL1Cache(&atomics[1]) != realBlockId)
        while (atomicAdd(&atomics[1], 0) != blockId)
        {
        }
        previousBlockValue = blockId == 0 ? 0 : blockTmp[blockId - 1];
    }



    __syncthreads();


    // do something
    computeAfterSync(previousBlockValue);

    int computedBlockValue = previousBlockValue + blockId;

    __syncthreads();

    if (ti.local_thread_id == 0)
    {
        blockTmp[blockId] = computedBlockValue;
        // Increment wait counter so the next block can ru
        //        atomicInc(&atomics[1],0xFFFFFFFF);
        atomics[1] = blockId + 1;
    }
}

static void intraBlockTest(int N)
{
    // Compute launch arguments
    const unsigned int BLOCK_SIZE = 128;
    const unsigned int BLOCKS     = Saiga::CUDA::getBlockCount(N, BLOCK_SIZE);



    thrust::device_vector<int> d_data(N);
    thrust::device_vector<int> d_blockTmp(BLOCKS);
    thrust::device_vector<unsigned int> atomics(2, 0);


    ibs<BLOCK_SIZE><<<BLOCKS, BLOCK_SIZE>>>(d_data, d_blockTmp, atomics);


    CUDA_SYNC_CHECK_ERROR();

    thrust::host_vector<int> data = d_blockTmp;

    for (auto i : data) std::cout << i << std::endl;
}

int main(int argc, char* argv[])
{
    intraBlockTest(1024 * 2);

    std::cout << "Done." << std::endl;
}
