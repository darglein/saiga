/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/core/time/timer.h"

namespace Saiga
{
namespace CUDA
{
template <typename T>
HD inline T recFact(T a)
{
    if (a == T(0))
        return 1;
    else
        return a * recFact(a - 1);
}


template <typename T>
HD inline T recFib(T a)
{
    if (a == T(0))
        return 0;
    else if (a == T(1))
        return 1;
    else
        return recFib(a - 1) + recFib(a - 2);
}


template <typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ static void recurseFact(ArrayView<T> data)
{
    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id; id < data.size(); id += ti.grid_size)
    {
        data[id] = recFact(id);
    }
}


// This produces the following warning:
// ptxas warning : Stack size for entry function '_ZN4CUDA10recurseFibIiLj128EEEv10ArrayViewIT_E' cannot be statically
// determined
template <typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ static void recurseFib(ArrayView<T> data)
{
    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id; id < data.size(); id += ti.grid_size)
    {
//        data[id] = recFib(id);
    }
}

// nvcc $CPPFLAGS -I ~/Master/libs/data/include/eigen3/ -ptx -lineinfo -src-in-ptx
// -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr inverse_test.cu nvcc $CPPFLAGS -I
// ~/Master/libs/data/include/eigen3/ -ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11
// --expt-relaxed-constexpr recursion_test.cu


void recursionTest()
{
    CUDA_SYNC_CHECK_ERROR();

    using ElementType = int;


    int N = 30;

    thrust::host_vector<ElementType> data(N, 0);


    thrust::device_vector<ElementType> d_data(data);


    thrust::host_vector<ElementType> ref(N, 0);
    for (int i = 0; i < N; ++i)
    {
        ref[i] = recFact(i);
    }

    {
        const int BLOCK_SIZE = 128;
        d_data               = data;
        {
            CUDA::ScopedTimerPrint t("recurseFact");
            recurseFact<ElementType, BLOCK_SIZE><<<CUDA::getBlockCount(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_data);
        }
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_data);



    for (int i = 0; i < N; ++i)
    {
        ref[i] = recFib(i);
    }

    {
        const int BLOCK_SIZE = 128;
        d_data               = data;
        {
            CUDA::ScopedTimerPrint t("recurseFib");
            recurseFib<ElementType, BLOCK_SIZE><<<CUDA::getBlockCount(N, BLOCK_SIZE), BLOCK_SIZE>>>(d_data);
        }
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_data);


    std::cout << "Recursion test success!" << std::endl;
    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
