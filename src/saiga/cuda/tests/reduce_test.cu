/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/reduce_global.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"

namespace Saiga
{
namespace CUDA
{
// nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr
// integrate_test.cu


void reduceTest()
{
    CUDA_SYNC_CHECK_ERROR();

    int N = 100 * 1000 * 1000;

    size_t readWrites = N * sizeof(int);
    CUDA::PerformanceTestHelper pth("Reduce Sum <int>", readWrites);

    thrust::device_vector<int> v(N, 1);
    thrust::host_vector<int> h(N, 1);

    //    std::cout << "Num elements: " << N << " Memory: " << double(N)*sizeof(int)/1000000.0 << "mb" << std::endl;



    int res = 0;
    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for (int i = 0; i < N; ++i)
            {
                res += h[i];
            }
        }
        SAIGA_ASSERT(res > 0);
        pth.addMeassurement("CPU reduce", time);
    }



    {
        float time;
        int sum;
        {
            CUDA::ScopedTimer t(time);
            sum = thrust::reduce(v.begin(), v.end());
        }
        SAIGA_ASSERT(sum == res);
        pth.addMeassurement("thrust::reduce", time);
    }

    {
        thrust::device_vector<int> res(1);
        float time;
        const int blockSize = 256;
        SAIGA_ASSERT(0);
        //        static auto numBlocks = CUDA::max_active_blocks(reduceBlockShared<int,blockSize>,blockSize,0);
        {
            CUDA::ScopedTimer t2(time);
            reduceBlockShared<int, blockSize><<<1, blockSize>>>(v, thrust::raw_pointer_cast(res.data()));
        }
        pth.addMeassurement("reduceBlockShared", time);
        //        std::cout << "reduceBlockShared - " << time << "ms." << std::endl;
        int sum = res[0];
        SAIGA_ASSERT(sum == N);
    }

    {
        const int blockSize = 256;
        //        auto numBlocks = CUDA::max_active_blocks(reduceBlockSharedAtomic<int,blockSize>,blockSize,0);
        thrust::device_vector<int> res(1);
        float time;
        {
            SAIGA_ASSERT(0);
            CUDA::ScopedTimer t2(time);
            reduceBlockSharedAtomic<int, blockSize><<<1, blockSize>>>(v, thrust::raw_pointer_cast(res.data()));
        }
        pth.addMeassurement("reduceBlockSharedAtomic", time);
        //        std::cout << "reduceBlockSharedAtomic - " << time << "ms." << std::endl;
        int sum = res[0];
        SAIGA_ASSERT(sum == N);
    }

    {
        const int blockSize = 256;
        //        auto numBlocks = CUDA::max_active_blocks(reduceAtomic<int,blockSize>,blockSize,0);
        thrust::device_vector<int> res(1);
        float time;
        {
            SAIGA_ASSERT(0);
            CUDA::ScopedTimer t2(time);
            reduceAtomic<int, blockSize><<<1, blockSize>>>(v, thrust::raw_pointer_cast(res.data()));
        }
        pth.addMeassurement("reduceAtomic", time);
        //        std::cout << "reduceAtomic - " << time << "ms." << std::endl;
        int sum = res[0];
        SAIGA_ASSERT(sum == N);
    }
    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
