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
#include "saiga/util/math.h"
#include "saiga/cuda/tests/test_helper.h"
using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;

/**
  *
  * Test to check if small kernels can be launched faster with DP.
  * Answer: No, it is actually slower than the 'normal' CPU launch.
  *
  */

// Ressources:
// https://devblogs.nvidia.com/introduction-cuda-dynamic-parallelism/
// https://devblogs.nvidia.com/cuda-dynamic-parallelism-api-principles/


static __device__ int a = 0;

__global__ void add1()
{
    atomicAdd(&a,1);
}

__global__ void dp(int maxIts)
{
    for(int i = 0;i < maxIts; ++i)
    {
        add1<<<4,128>>>();
    }
}

int main()
{


    Saiga::CUDA::PerformanceTestHelper test("DP",4);

    float maxIts = 500;
    int testIts = 15;

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]() {
            for(int i = 0;i < maxIts; ++i)
            {
                add1<<<4,128>>>();
            }
        });
        test.addMeassurement("CPU Launch", st.median);
        float launchTime = st.median / maxIts;
        cout << "Kernel Launch Time " << launchTime*1000 << " micro seconds." << endl;
        CUDA_SYNC_CHECK_ERROR();
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]() {

            dp<<<1,1>>>(maxIts);

        });
        test.addMeassurement("DP Launch", st.median);
        float launchTime = st.median / maxIts;
        cout << "Kernel Launch Time " << launchTime*1000 << " micro seconds." << endl;
        CUDA_SYNC_CHECK_ERROR();
    }

    cudaDeviceSynchronize();
    cout << "Done." << endl;
    return 0;
}
