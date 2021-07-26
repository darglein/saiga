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
  * Solution of the following problem on the GPU:
  *
  * for(int i = 0; i < 100; ++i)
  * {
  *     if(compute(i) > threshold)
  *         break;
  * }
  *
  *
  * Problem:
  *     - The result of compute(i) is stored in device memory
  *     - The CPU has to see it to decide if we can break early from this loop
  *
  * 3 Different Solution are given here:
  *
  *     1. Synchronous memcpy after compute in every step
  *     2. Asynchronous memcpy and sync on earlier iteration
  *         -> The loop is terminated a few iterations too late, but no host-device sync is required
  *     3. The loop is moved to a kernel with dynamic parallelism
  *
  *
  * Output:
  *
  * Name                                    Time (ms)      Bandwidth (GB/s)
  * 1. Sync memcpy                          2.59715        0.0252338
  * 2. Async Streams                        1.72704        0.037947
  * 3. Dynamic Parallelism                  2.01776        0.0324796
  *
  *
  */

// Ressources:
// https://devblogs.nvidia.com/introduction-cuda-dynamic-parallelism/
// https://devblogs.nvidia.com/cuda-dynamic-parallelism-api-principles/

template <int K>
class SAIGA_ALIGN(16) Element
{
    public:
    vec4 data = vec4(1.1);

    HD inline void operator()()
    {
        for (int k = 0; k < K; ++k)
        {
            data = data * data + data;
        }
    }
};


template <typename T>
__global__ static void process(ArrayView<T> data, ArrayView<float> residual, int it)
{
    ThreadInfo<> ti;
    if (ti.thread_id >= data.size()) return;

    T e = data[ti.thread_id];
    e();
    data[ti.thread_id] = e;

    if(ti.thread_id == 0)
        residual[it] = it;
}




template<typename T>
static void __global__ earlyExitDP(ArrayView<T> data, ArrayView<float> residual, int maxIts, int earlyExitIterations)
{
    auto N = data.size();

    for(int i = 0;i < maxIts; ++i)
    {
        process<T><<<THREAD_BLOCK(N,128)>>>(data,residual,i);
        cudaDeviceSynchronize();
        auto res = residual[i];
        if(res > earlyExitIterations)
            break;
    }
}


int main()
{
    int N = 2 * 1024;
    int maxIts = 100;
    using T = Element<16*512>;
    Saiga::thrust::pinned_vector<T> h_data(N);
    thrust::device_vector<T> d_data(N);

    Saiga::thrust::pinned_vector<float> h_res(maxIts);
    thrust::device_vector<float> d_res(maxIts,0);


    Saiga::CUDA::PerformanceTestHelper test("DP",N*2*sizeof(T));

    float earlyExitIterations = 50;
    int testIts = 15;

#if 0
    {
        thrust::fill(d_res.begin(),d_res.end(),0);
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]() {
            process<T><<<THREAD_BLOCK(N,128)>>>(d_data,d_res,0);
        });
        test.addMeassurement("single iteration", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }

    {
        thrust::fill(d_res.begin(),d_res.end(),0);
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]() {
            for(int i = 0;i < maxIts; ++i)
            {
                process<T><<<THREAD_BLOCK(N,128)>>>(d_data,d_res,i);
            }
        });
        test.addMeassurement("100 iterations", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }

    {
        thrust::fill(d_res.begin(),d_res.end(),0);
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]() {
            for(int i = 0;i < earlyExitIterations; ++i)
            {
                process<T><<<THREAD_BLOCK(N,128)>>>(d_data,d_res,i);
            }
        });
        test.addMeassurement("early exit", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }
#endif

    {
        thrust::fill(d_res.begin(),d_res.end(),0);
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]() {
            for(int i = 0;i < maxIts; ++i)
            {
                process<T><<<THREAD_BLOCK(N,128)>>>(d_data,d_res,i);
                cudaMemcpy(h_res.data()+i,d_res.data().get()+i,4,cudaMemcpyDeviceToHost);
                auto res = h_res[i];
                if(res > earlyExitIterations)
                    break;
            }
        });
        test.addMeassurement("1. Sync memcpy", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }


    {
        thrust::fill(d_res.begin(),d_res.end(),0);

        Saiga::CUDA::CudaStream strm;
        Saiga::CUDA::CudaStream cpystrm;
        int tileSize = 4;
        int numTiles = maxIts / tileSize;


        std::vector<Saiga::CUDA::CudaEvent> events(numTiles);
        std::vector<Saiga::CUDA::CudaEvent> events2(numTiles);

        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]()
        {

            for(int t= 0; t < numTiles; ++t)
            {
                if(t > 1)
                {
                    // Wait on previous tile
                    // Check residual and break
                    events[t-2].synchronize();

                    int lastFromPreviousTile = (t-2) * tileSize + tileSize-1;
                    if(h_res[lastFromPreviousTile] > earlyExitIterations)
                    {
                          break;
                    }
                }


                // Queue next tile
                for(int i = 0; i < tileSize; ++i)
                {
                    int it = t * tileSize + i;
                    process<T><<<THREAD_BLOCK(N,128),0,strm>>>(d_data,d_res,it);
                }
                events2[t].record(strm);

                cpystrm.waitForEvent(events2[t]);
                int lastFromCurrentTile = t * tileSize + tileSize - 1;
                cudaMemcpyAsync(h_res.data()+lastFromCurrentTile,d_res.data().get()+lastFromCurrentTile,4,cudaMemcpyDeviceToHost,cpystrm);
                events[t].record(cpystrm);
            }
        });
        test.addMeassurement("2. Async Streams", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }

    {
        thrust::fill(d_res.begin(),d_res.end(),0);
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(
                    testIts, [&]() {
            earlyExitDP<T><<<1,1>>>(d_data,d_res,maxIts,earlyExitIterations);
        });
        test.addMeassurement("3. Dynamic Parallelism", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }





    cudaDeviceSynchronize();
    cout << "Done." << endl;
    return 0;
}
