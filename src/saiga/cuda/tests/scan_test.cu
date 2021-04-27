/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/timer.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/scan.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"

namespace Saiga
{
namespace CUDA
{
using uint = unsigned int;

void scanTest()
{
    CUDA_SYNC_CHECK_ERROR();

    const bool exclusive           = false;
    const size_t THREADS_PER_BLOCK = 256;
    const int TILES_PER_BLOCK      = 8;
    const int ELEMENTS_PER_VECTOR  = 4;

    const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * TILES_PER_BLOCK * ELEMENTS_PER_VECTOR;
    int N                        = 100 * 1000 * 1000;

    std::cout << "Elements: " << N << " Elements per block: " << ELEMENTS_PER_BLOCK << std::endl;

    size_t readWrites = N * 2 * sizeof(uint);

    CUDA::PerformanceTestHelper pth("Scan (exclusive)", readWrites);


    thrust::host_vector<uint> h(N, 1);

    for (int i = 0; i < N; ++i)
    {
        h[i] = rand() % 4;
    }

    thrust::device_vector<uint> v = h;

    thrust::device_vector<uint> d_res(N + ELEMENTS_PER_BLOCK, -1);
    thrust::host_vector<uint> h_res(N + ELEMENTS_PER_BLOCK, -1);

    thrust::device_vector<uint> aggregate(CUDA::getBlockCount(N, ELEMENTS_PER_BLOCK) + 1, -1);

    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            if (exclusive)
            {
                int sum = 0;
                for (int i = 0; i < N; ++i)
                {
                    h_res[i] = sum;
                    sum += h[i];
                }
            }
            else
            {
                int sum = 0;
                for (int i = 0; i < N; ++i)
                {
                    sum += h[i];
                    h_res[i] = sum;
                }
            }
        }
        pth.addMeassurement("CPU scan", time);
    }


    {
        float time;
        {
            CUDA::ScopedTimer t(time);
            if (exclusive)
            {
                thrust::exclusive_scan(v.begin(), v.end(), d_res.begin());
            }
            else
            {
                thrust::inclusive_scan(v.begin(), v.end(), d_res.begin());
            }
        }
        //        SAIGA_ASSERT(sum == res);
        pth.addMeassurement("thrust::scan", time);
    }

    SAIGA_ASSERT(d_res == h_res);



    {
        d_res = thrust::device_vector<uint>(N + ELEMENTS_PER_BLOCK, -1);

        float time;

        auto NUM_BLOCKS = CUDA::getBlockCount(N, ELEMENTS_PER_BLOCK);
        {
            CUDA::ScopedTimer t(time);
            CUDA::tiledSinglePassScan<exclusive, THREADS_PER_BLOCK, TILES_PER_BLOCK, int4, true>
                <<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(v, d_res, aggregate);
        }
        pth.addMeassurement("tiledSinglePassScan", time);
    }


    /*
    //check if the aggregate was computed correctly
    thrust::host_vector<unsigned int> h_a = aggregate;
    int i = ELEMENTS_PER_BLOCK;
    for(int ag : h_a){
        //        SAIGA_ASSERT(ag == i);
        i += ELEMENTS_PER_BLOCK;
    }

    thrust::host_vector<unsigned int> h_res2 = d_res;

    int maxPrint = ELEMENTS_PER_BLOCK * 2;
    for(int i = 0 ; i < int(h_res.size()) ; ++i){
        if(h_res2[i] != h_res[i]){
            std::cout << i << " " << h_res2[i] << "!=" << h_res[i] << std::endl;
            maxPrint--;
            if(maxPrint < 0)
                break;
        }
    }
    */

    SAIGA_ASSERT(d_res == h_res);


    {
        float time;
        {
            CUDA::ScopedTimer t(time);
            cudaMemcpy(thrust::raw_pointer_cast(d_res.data()), thrust::raw_pointer_cast(v.data()), N * sizeof(int),
                       cudaMemcpyDeviceToDevice);
        }
        pth.addMeassurement("cudaMemcpy", time);
    }
    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
