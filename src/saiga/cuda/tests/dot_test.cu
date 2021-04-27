/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/timer.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/dot.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"

#include <thrust/inner_product.h>

namespace Saiga
{
namespace CUDA
{
// nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr
// integrate_test.cu


void dotTest()
{
    CUDA_SYNC_CHECK_ERROR();

#ifdef SAIGA_USE_CUBLAS
    initBLASSPARSE();
#endif

    {
        using elementType = float;
        int N             = 100 * 1000 * 1000;
        size_t readWrites = N * 2 * sizeof(elementType);

        CUDA::PerformanceTestHelper pth("Dot Product <float>", readWrites);

        thrust::device_vector<elementType> v1(N, 1);
        thrust::device_vector<elementType> v2(N, 2);
        thrust::host_vector<elementType> h1 = v1;
        thrust::host_vector<elementType> h2 = v2;


        elementType ref = 0;
        {
            float time;
            {
                Saiga::ScopedTimer<float> t(&time);
                for (int i = 0; i < N; ++i)
                {
                    ref += h1[i] * h2[i];
                }
            }
            SAIGA_ASSERT(ref > 0);
            pth.addMeassurement("CPU dot", time);
        }


        {
            float time;
            elementType sum;
            {
                CUDA::ScopedTimer t(time);
                sum = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), 0);
            }
            pth.addMeassurement("thrust::inner_product", time);
            ref = sum;
            SAIGA_ASSERT(sum >= ref - 0.1f && sum <= ref + 0.1f);
        }

#ifdef SAIGA_USE_CUBLAS
        {
            thrust::device_vector<elementType> d_res(1, 0);
            // make sure no additional memcpy is issued
            cublasSetPointerMode(cublashandle, CUBLAS_POINTER_MODE_DEVICE);
            float time;
            elementType sum;
            {
                CUDA::CudaScopedTimer t(time);
                cublasSdot(cublashandle, N, thrust::raw_pointer_cast(v1.data()), 1, thrust::raw_pointer_cast(v2.data()),
                           1, thrust::raw_pointer_cast(d_res.data()));
            }
            sum = d_res[0];
            SAIGA_ASSERT(sum >= ref - 0.1f && sum <= ref + 0.1f);
            pth.addMeassurement("cublasSdot", time);
        }
#endif


        {
            thrust::device_vector<elementType> d_res(1, 0);
            float time;
            {
                const int blockSize = 256;
                SAIGA_ASSERT(0);
                //                static auto numBlocks = max_active_blocks(dot<elementType,blockSize>,blockSize,0);
                CUDA::ScopedTimer t2(time);
                dot<elementType, blockSize><<<1, blockSize>>>(v1, v2, thrust::raw_pointer_cast(d_res.data()));
            }
            elementType sum = d_res[0];
            SAIGA_ASSERT(sum >= ref - 0.1f && sum <= ref + 0.1f);
            time = time;
            pth.addMeassurement("my dot product", time);
        }
    }



    {
        using elementType = double;
        int N             = 50 * 1000 * 1000;
        size_t readWrites = N * 2 * sizeof(elementType);

        CUDA::PerformanceTestHelper pth("Dot Product <double>", readWrites);

        thrust::device_vector<elementType> v1(N, 1);
        thrust::device_vector<elementType> v2(N, 2);
        thrust::host_vector<elementType> h1 = v1;
        thrust::host_vector<elementType> h2 = v2;


        elementType ref = 0;
        {
            float time;
            {
                Saiga::ScopedTimer<float> t(&time);
                for (int i = 0; i < N; ++i)
                {
                    ref += h1[i] * h2[i];
                }
            }
            SAIGA_ASSERT(ref > 0);
            pth.addMeassurement("CPU dot", time);
        }


        {
            float time;
            elementType sum;
            {
                CUDA::ScopedTimer t(time);
                sum = thrust::inner_product(v1.begin(), v1.end(), v2.begin(), 0);
            }
            pth.addMeassurement("thrust::inner_product", time);
            ref = sum;
            SAIGA_ASSERT(sum >= ref - 0.1f && sum <= ref + 0.1f);
        }

#ifdef SAIGA_USE_CUBLAS
        {
            thrust::device_vector<elementType> d_res(1, 0);
            // make sure no additional memcpy is issued
            cublasSetPointerMode(cublashandle, CUBLAS_POINTER_MODE_DEVICE);
            float time;
            elementType sum;
            {
                CUDA::CudaScopedTimer t(time);
                cublasDdot(cublashandle, N, thrust::raw_pointer_cast(v1.data()), 1, thrust::raw_pointer_cast(v2.data()),
                           1, thrust::raw_pointer_cast(d_res.data()));
            }
            sum = d_res[0];
            SAIGA_ASSERT(sum >= ref - 0.1f && sum <= ref + 0.1f);
            pth.addMeassurement("cublasSdot", time);
        }
#endif


        {
            thrust::device_vector<elementType> d_res(1, 0);
            float time;
            {
                const int blockSize = 256;
                SAIGA_ASSERT(0);
                //                static auto numBlocks = max_active_blocks(dot<elementType,blockSize>,blockSize,0);
                CUDA::ScopedTimer t2(time);
                dot<elementType, blockSize><<<1, blockSize>>>(v1, v2, thrust::raw_pointer_cast(d_res.data()));
            }
            elementType sum = d_res[0];
            SAIGA_ASSERT(sum >= ref - 0.1f && sum <= ref + 0.1f);
            time = time;
            pth.addMeassurement("my dot product", time);
        }
    }
    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
