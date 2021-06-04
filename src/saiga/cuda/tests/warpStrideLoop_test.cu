/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/timer.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/reduce.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"

namespace Saiga
{
namespace CUDA
{
// Note:
// batchreduce2 is always faster or same speed as batchreduce.
// the compiler can unroll the load loop in batchreduce2 but not in batchreduce
// for large Ns the compiler partially unrolls the loop (~8 iterations)


// nvcc $CPPFLAGS -ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr
// warpStrideLoop_test.cu


template <typename T, unsigned int BLOCK_SIZE, unsigned int LOCAL_WARP_SIZE, int N>
__global__ static void batchReduce(ArrayView<T> in, ArrayView<T> out)
{
    ThreadInfo<BLOCK_SIZE, LOCAL_WARP_SIZE> ti;

    if (ti.warp_id >= out.size()) return;

    int inoffset  = ti.warp_id * N;
    int outoffset = ti.warp_id;

    T sum = 0;
    for (int i = ti.lane_id; i < N; i += LOCAL_WARP_SIZE)
    {
        sum += in[i + inoffset];
    }

    sum = warpReduceSum<T, LOCAL_WARP_SIZE>(sum);

    if (ti.lane_id == 0)
    {
        out[outoffset] = sum;
    }
}


template <typename T, unsigned int BLOCK_SIZE, unsigned int LOCAL_WARP_SIZE, int N>
__global__ static void batchReduce2(ArrayView<T> in, ArrayView<T> out)
{
    ThreadInfo<BLOCK_SIZE, LOCAL_WARP_SIZE> ti;

    if (ti.warp_id >= out.size()) return;

    int inoffset  = ti.warp_id * N;
    int outoffset = ti.warp_id;

    T sum = 0;

    //    for(int k = 0, i = ti.lane_id; k < iDivUp(N,LOCAL_WARP_SIZE) ; ++k, i+=LOCAL_WARP_SIZE){
    //        if(i < N){
    //            sum += in[i+inoffset];
    //        }
    //    }

    WARP_FOR(i, ti.lane_id, N, LOCAL_WARP_SIZE) { sum += in[i + inoffset]; }

    sum = warpReduceSum<T, LOCAL_WARP_SIZE>(sum);

    if (ti.lane_id == 0)
    {
        out[outoffset] = sum;
    }
}

#if 0
__global__
static void shflTest(){
    int tid = threadIdx.x;
    float value = tid + 0.1f;
    int* ivalue = reinterpret_cast<int*>(&value);

    int ix = __shfl(ivalue[0],5,32);
    int iy = __shfl_sync(ivalue[0],5,32);

    float x = reinterpret_cast<float*>(&ix)[0];
    float y = reinterpret_cast<float*>(&iy)[0];

    if(tid == 0){
        printf("shfl tmp %d %d\n",ix,iy);
        printf("shfl final %f %f\n",x,y);
    }
}
#endif



template <int N>
void warpStrideLoopTest2()
{
#if 0
    {
        shflTest<<<1,32>>>();
        CUDA_SYNC_CHECK_ERROR();
        return;
    }
#endif

    using ReduceType = int;

    //    const int N = 32 * 100;

    int numEles = 1000 * 1000 * 10;
    //    int numEles = 1000000;
    const int K = iDivUp(numEles, N);

    size_t readWrites = K * N * sizeof(ReduceType) + K * sizeof(ReduceType);
    CUDA::PerformanceTestHelper pth("Batch Reduce Sum N=" + std::to_string(N), readWrites);

    thrust::device_vector<ReduceType> in(N * K, 1);
    thrust::device_vector<ReduceType> out(K, 0);


    thrust::host_vector<ReduceType> hin  = in;
    thrust::host_vector<ReduceType> hout = out;


    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for (int k = 0; k < K; ++k)
            {
                int res = 0;

                for (int i = 0; i < N; ++i)
                {
                    res += hin[i + k * N];
                }
                hout[k] = res;
            }
        }
        pth.addMeassurement("CPU reduce", time);
    }

    {
        const int blockSize       = 128;
        const int LOCAL_WARP_SIZE = 1;
        auto numBlocks            = iDivUp(K * LOCAL_WARP_SIZE, blockSize);

        float time;
        {
            CUDA::ScopedTimer t2(time);
            batchReduce<ReduceType, blockSize, LOCAL_WARP_SIZE, N><<<numBlocks, blockSize>>>(in, out);
        }
        pth.addMeassurement("batch reduce warpsize = 1", time);

        SAIGA_ASSERT(out == hout);
    }


    {
        const int blockSize       = 128;
        const int LOCAL_WARP_SIZE = 32;
        auto numBlocks            = iDivUp(K * LOCAL_WARP_SIZE, blockSize);

        float time;
        {
            CUDA::ScopedTimer t2(time);
            batchReduce<ReduceType, blockSize, LOCAL_WARP_SIZE, N><<<numBlocks, blockSize>>>(in, out);
        }
        pth.addMeassurement("batch reduce 1", time);

        SAIGA_ASSERT(out == hout);
    }

    {
        const int blockSize       = 128;
        const int LOCAL_WARP_SIZE = 32;
        auto numBlocks            = iDivUp(K * LOCAL_WARP_SIZE, blockSize);

        float time;
        {
            CUDA::ScopedTimer t2(time);
            batchReduce2<ReduceType, blockSize, LOCAL_WARP_SIZE, N><<<numBlocks, blockSize>>>(in, out);
        }
        pth.addMeassurement("batch reduce 2", time);

        SAIGA_ASSERT(out == hout);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void warpStrideLoopTest()
{
    warpStrideLoopTest2<57>();
    warpStrideLoopTest2<320>();
    warpStrideLoopTest2<1252>();
    warpStrideLoopTest2<19276>();
}

}  // namespace CUDA
}  // namespace Saiga
