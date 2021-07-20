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
using namespace Saiga;


template<typename T>
__global__ static void copy(Saiga::ArrayView<T> src,
                                            Saiga::ArrayView<T> dst)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= src.size()) return;

    dst[ti.thread_id] = src[ti.thread_id];
}


void memcpyTest()
{
    size_t N          = 64 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);


    Saiga::CUDA::PerformanceTestHelper pth("Memcpy", readWrites);
    // Test 10 times and use the median time
    int its = 500;
    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()), thrust::raw_pointer_cast(src.data()), N * sizeof(int),
                       cudaMemcpyDeviceToDevice);
        });
        pth.addMeassurement("cudaMemcpy", st.median);
    }

        const unsigned int BLOCK_SIZE = 128;
    {
        using T = float;
        auto size =src.size() * sizeof(int) / sizeof(T);
        auto srcv = Saiga::ArrayView<T>( (T*)thrust::raw_pointer_cast(src.data()),size );
        auto dstv = Saiga::ArrayView<T>( (T*)thrust::raw_pointer_cast(dest.data()),size);
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            copy<T><<<THREAD_BLOCK(size, BLOCK_SIZE)>>>(srcv,dstv);
        });
        pth.addMeassurement("copy 4", st.median);
    }

    {
        using T = float2;
        auto size =src.size() * sizeof(int) / sizeof(T);
        auto srcv = Saiga::ArrayView<T>( (T*)thrust::raw_pointer_cast(src.data()),size );
        auto dstv = Saiga::ArrayView<T>( (T*)thrust::raw_pointer_cast(dest.data()),size);
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            copy<T><<<THREAD_BLOCK(size, BLOCK_SIZE)>>>(srcv,dstv);
        });
        pth.addMeassurement("copy 8", st.median);
    }
    {
        using T = float4;
        auto size =src.size() * sizeof(int) / sizeof(T);
        auto srcv = Saiga::ArrayView<T>( (T*)thrust::raw_pointer_cast(src.data()),size );
        auto dstv = Saiga::ArrayView<T>( (T*)thrust::raw_pointer_cast(dest.data()),size);
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            copy<T><<<THREAD_BLOCK(size, BLOCK_SIZE)>>>(srcv,dstv);
        });
        pth.addMeassurement("copy 16", st.median);
    }

    CUDA_SYNC_CHECK_ERROR();
}

int main(int argc, char* argv[])
{
    memcpyTest();
    return 0;
}
