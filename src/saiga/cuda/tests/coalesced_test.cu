/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/time/timer.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/shfl_helper.h"
#include "saiga/cuda/shuffle_copy.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"

namespace Saiga
{
namespace CUDA
{
template <typename T, int ElementSize, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ static void copyUnCoalesced(ArrayView<T> data, ArrayView<T> result)
{
    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id * ElementSize; id < data.size(); id += ti.grid_size * ElementSize)
    {
        T l[ElementSize];

        auto localStart = ti.thread_id * ElementSize;


        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] = data[localStart + i];
        }


        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] += 42;
        }

        for (int i = 0; i < ElementSize; ++i)
        {
            result[localStart + i] = l[i];
        }
    }
}


template <typename T, int ElementSize, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ static void copyFullCoalesced(ArrayView<T> data, ArrayView<T> result)
{
    const int elementsPerWarp = ElementSize * SAIGA_WARP_SIZE;

    CUDA::ThreadInfo<BLOCK_SIZE> ti;

    auto N             = data.size();
    auto Nelements     = N / ElementSize;
    auto requiredWarps = CUDA::getBlockCount(Nelements, SAIGA_WARP_SIZE);


    // grid stride loop
    for (auto wId = ti.warp_id; wId < requiredWarps; wId += ti.num_warps)
    {
        auto warpStart = wId * elementsPerWarp;

        for (auto e = ti.lane_id; e < elementsPerWarp; e += SAIGA_WARP_SIZE)
        {
            auto globalOffset = warpStart + e;
            if (globalOffset < N)
            {
                auto d = data[globalOffset];
                d += 42;
                result[globalOffset] = d;
            }
        }
    }
}

template <typename T, int ElementSize, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ static void sharedMemoryUnCoalesced(ArrayView<T> data, ArrayView<T> result)
{
    __shared__ T buffer[BLOCK_SIZE][ElementSize + 0];



    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id * ElementSize; id < data.size(); id += ti.grid_size * ElementSize)
    {
        T l[ElementSize];

        auto matrixId      = ti.thread_id;
        auto globalOffset  = matrixId * ElementSize;
        auto localMatrixId = ti.local_thread_id;  // id in shared buffer

        // linear copy
        for (int i = 0; i < ElementSize; ++i)
        {
            buffer[localMatrixId][i] = data[globalOffset + i];
        }

        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] = buffer[localMatrixId][i];
        }


        // add something so things don't get optimized away
        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] += 42;
        }

        for (int i = 0; i < ElementSize; ++i)
        {
            buffer[localMatrixId][i] = l[i];
        }

        // linear copy
        for (int i = 0; i < ElementSize; ++i)
        {
            result[globalOffset + i] = buffer[localMatrixId][i];
        }
    }
}


template <typename T, int ElementSize, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ static void sharedMemoryCoalesced(ArrayView<T> data, ArrayView<T> result)
{
    CUDA::ThreadInfo<BLOCK_SIZE> ti;

    const int elementsPerWarp = ElementSize * SAIGA_WARP_SIZE;

    auto N             = data.size();
    auto Nelements     = N / ElementSize;
    auto requiredWarps = CUDA::getBlockCount(Nelements, SAIGA_WARP_SIZE);


    //    __shared__ double buffer[elementsPerBlock];
    __shared__ T buffer[BLOCK_SIZE][ElementSize + 0];

    // grid stride loop
    for (auto wId = ti.warp_id; wId < requiredWarps; wId += ti.num_warps)
    {
        //    for(auto id = ti.thread_id * ElementSize; id < N; id += ti.num_warps){
        //    for(auto id = ti.thread_id; id < Nelements; id += ti.grid_size ){

        T l[ElementSize];

        auto localMatrixId = ti.local_thread_id;  // id in shared buffer
        auto warpStart     = ti.warp_id * elementsPerWarp;

        // strided copy
        for (auto e = ti.lane_id; e < elementsPerWarp; e += SAIGA_WARP_SIZE)
        {
            auto localMatrix  = ti.warp_lane * SAIGA_WARP_SIZE + e / ElementSize;
            auto localOffset  = e % ElementSize;
            auto globalOffset = warpStart + e;
            if (globalOffset < N)
            {
                buffer[localMatrix][localOffset] = data[globalOffset];
            }
        }

        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] = buffer[localMatrixId][i];
        }


        // add something so things don't get optimized away
        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] += 42;
        }

        for (int i = 0; i < ElementSize; ++i)
        {
            buffer[localMatrixId][i] = l[i];
        }

        // strided copy
        for (auto e = ti.lane_id; e < elementsPerWarp; e += SAIGA_WARP_SIZE)
        {
            auto localMatrix  = ti.warp_lane * SAIGA_WARP_SIZE + e / ElementSize;
            auto localOffset  = e % ElementSize;
            auto globalOffset = warpStart + e;
            if (globalOffset < N)
            {
                result[globalOffset] = buffer[localMatrix][localOffset];
            }
        }
    }
}


template <typename T, int ElementSize, unsigned int BLOCK_SIZE, typename VectorType = int2>
__launch_bounds__(BLOCK_SIZE) __global__ static void sharedMemoryCoalesced2(ArrayView<T> data, ArrayView<T> result)
{
    const int elementSize           = sizeof(T) * ElementSize;
    const int fullVectorsPerElement = elementSize / sizeof(VectorType);



#ifdef SAIGA_HAS_CONSTEXPR
    const int vectorsPerElement = CUDA::getBlockCount(elementSize, sizeof(VectorType));
    static_assert(vectorsPerElement * sizeof(VectorType) == elementSize, "T cannot be loaded with VectorType");
#else
    const int vectorsPerElement = 1;
#endif

    //    const int vectorsPerWarp = fullVectorsPerElement * SAIGA_WARP_SIZE;

    const int tileSizeBytes      = 64;
    const int tileSizeVectors    = tileSizeBytes / sizeof(VectorType);
    const int fullVectorsPerTile = fullVectorsPerElement > tileSizeVectors ? tileSizeVectors : fullVectorsPerElement;
    const int vectorsPerTile     = vectorsPerElement > tileSizeVectors ? tileSizeVectors : vectorsPerElement;
    //    const int vectorsPerTile = N > 8 ? 8 : N;
    const int fullTiles =
        fullVectorsPerElement == 0 ? fullVectorsPerElement : fullVectorsPerElement / fullVectorsPerTile;

    //    const int fullTiles = fullVectorsPerElement == 0 ? fullVectorsPerElement : fullVectorsPerElement /
    //    fullVectorsPerTile; const int tiles = 2; const int elementsPerTile = N / tiles;
    const int fullVectorsPerBlock = fullVectorsPerElement * BLOCK_SIZE;

    //    __shared__ double buffer[elementsPerBlock];
    __shared__ VectorType buffer[BLOCK_SIZE][vectorsPerTile];
    //    __shared__ T buffer[BLOCK_SIZE][N];


    T l[ElementSize];

    auto N             = data.size();
    auto Nelements     = N / ElementSize;
    auto NVectors      = N * sizeof(T) / sizeof(VectorType);
    auto requiredWarps = CUDA::getBlockCount(Nelements, SAIGA_WARP_SIZE);

    VectorType* global       = reinterpret_cast<VectorType*>(data.data());
    VectorType* globalResult = reinterpret_cast<VectorType*>(result.data());
    VectorType* local        = reinterpret_cast<VectorType*>(l);

    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    //    for(auto id = ti.thread_id * ElementSize; id < data.size(); id += ti.grid_size * ElementSize){
    for (auto wId = ti.warp_id; wId < requiredWarps; wId += ti.num_warps)
    {
        auto localMatrixId = ti.local_thread_id;  // id in shared buffer
        //        auto warpStart = ti.warp_id * vectorsPerWarp;
        auto blockStart = ti.block_id * fullVectorsPerBlock;

        auto warpOffset = ti.warp_lane * SAIGA_WARP_SIZE;  // start matrix of this warp in block local shared memory

#if 1
        for (int t = 0; t < fullTiles; ++t)
        {
            auto tileOffset = t * fullVectorsPerTile;
            // strided copy
            for (auto e = ti.lane_id; e < fullVectorsPerTile * SAIGA_WARP_SIZE; e += SAIGA_WARP_SIZE)
            {
                auto localMatrix = warpOffset + e / fullVectorsPerTile;
                auto localOffset = e % fullVectorsPerTile;
                auto globalIndex = blockStart + localMatrix * fullVectorsPerElement + tileOffset + localOffset;
                if (globalIndex < NVectors)
                {
                    buffer[localMatrix][localOffset] = global[globalIndex];
                    //                    printf("read %d %d %d \n",ti.thread_id,localMatrix,globalIndex);
                }
            }

            for (int i = 0; i < fullVectorsPerTile; ++i)
            {
                local[i + tileOffset] = buffer[localMatrixId][i];
            }
        }

#else
        // strided copy
        for (auto e = ti.lane_id; e < elementsPerWarp; e += SAIGA_WARP_SIZE)
        {
            auto localMatrix                 = ti.warp_lane * SAIGA_WARP_SIZE + e / N;
            auto localOffset                 = e % N;
            buffer[localMatrix][localOffset] = data[warpStart + e];
        }

        for (int i = 0; i < N; ++i)
        {
            l[i] = buffer[localMatrixId][i];
        }
#endif


        // add something so things don't get optimized away
        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] += 42;
        }


        //        for(int t = 0 ; t < tiles ; ++t){
        //            auto tileOffset = t * elementsPerTile;

        //            for(int i = 0; i < elementsPerTile; ++i){
        //                buffer[localMatrixId][i] = l[i + tileOffset];
        //            }

        //            //strided copy
        //            for(auto e = ti.lane_id; e < elementsPerTile * SAIGA_WARP_SIZE; e += SAIGA_WARP_SIZE){
        //                auto localMatrix = ti.warp_lane * SAIGA_WARP_SIZE + e / elementsPerTile;
        //                auto localOffset = e % elementsPerTile;
        //                result[tileOffset+warpStart+e] = buffer[localMatrix][localOffset];
        //            }



        //        }

        //        for(int i = 0; i < N; ++i){
        //            buffer[localMatrixId][i] =  l[i];
        //        }

        ////        strided copy
        //        for(auto e = ti.lane_id; e < elementsPerWarp; e += SAIGA_WARP_SIZE){
        //            auto localMatrix = ti.warp_lane * SAIGA_WARP_SIZE + e / N;
        //            auto localOffset = e % N;
        //            result[warpStart+e] = buffer[localMatrix][localOffset];
        //        }

        for (int t = 0; t < fullTiles; ++t)
        {
            auto tileOffset = t * fullVectorsPerTile;

            for (int i = 0; i < fullVectorsPerTile; ++i)
            {
                buffer[localMatrixId][i] = local[i + tileOffset];
            }
            // strided copy
            for (auto e = ti.lane_id; e < fullVectorsPerTile * SAIGA_WARP_SIZE; e += SAIGA_WARP_SIZE)
            {
                auto localMatrix = warpOffset + e / fullVectorsPerTile;
                auto localOffset = e % fullVectorsPerTile;
                auto globalIndex = blockStart + localMatrix * fullVectorsPerElement + tileOffset + localOffset;
                if (globalIndex < NVectors)
                {
                    globalResult[globalIndex] = buffer[localMatrix][localOffset];
                    //                    printf("write %d %d %d \n",ti.thread_id,localMatrix,globalIndex);
                }
            }
        }
    }
}


template <typename T, int ElementSize, unsigned int BLOCK_SIZE, typename VectorType = int4, int localWarpSize2 = -1>
__launch_bounds__(BLOCK_SIZE) __global__ static void shuffleCopy(ArrayView<T> data, ArrayView<T> result)
{
    const int localWarpSize =
        localWarpSize2 == -1 ? int(SAIGA_L2_CACHE_LINE_SIZE / sizeof(VectorType)) : localWarpSize2;
    const int vectorsPerElement = CUDA::getBlockCount(ElementSize * sizeof(T), sizeof(VectorType));

    auto N             = data.size();
    auto Nelements     = N / ElementSize;
    auto NVectors      = N * sizeof(T) / sizeof(VectorType);
    auto requiredWarps = CUDA::getBlockCount(Nelements, localWarpSize);


    //    const int localWarpSize = 2;

    CUDA::ThreadInfo<BLOCK_SIZE, localWarpSize> ti;



    // grid stride loop
    //    for(auto id = ti.thread_id * ElementSize; id < data.size(); id += ti.grid_size * ElementSize){
    for (auto wId = ti.warp_id; wId < requiredWarps; wId += ti.num_warps)
    {
        T l[ElementSize];

        //        auto matrixId = ti.thread_id;
        //        auto globalOffset = matrixId * ElementSize;
        //        auto localMatrixId = ti.local_thread_id; //id in shared buffer


        auto globalStart = wId * localWarpSize * vectorsPerElement;

        //        printf("warp %d %d %d %d \n", wId,ti.lane_id,localWarpSize,Nelements);

        VectorType* global       = reinterpret_cast<VectorType*>(data.data());
        VectorType* globalResult = reinterpret_cast<VectorType*>(result.data());
        VectorType* local        = reinterpret_cast<VectorType*>(l);

        //        loadShuffle<localWarpSize,sizeof(T)*ElementSize,VectorType>(data.data()+globalStart,local,ti.lane_id);
        loadShuffle<localWarpSize, sizeof(T) * ElementSize, VectorType>(global, local, ti.lane_id, globalStart,
                                                                        NVectors);


        for (int i = 0; i < ElementSize; ++i)
        {
            l[i] += 42;
        }

        storeShuffle<localWarpSize, sizeof(T) * ElementSize, VectorType>(globalResult, local, ti.lane_id, globalStart,
                                                                         NVectors);
    }
}

/*
__global__ static
void strangeLoop(int* data, int* out, int N){
    auto id = blockDim.x * blockIdx.x + threadIdx.x;
    auto lane = threadIdx.x % 2;

    if(id >= N)
        return;

    int local[2];

    for(int i = 0 ; i < 2 ; ++i)
        local[i] = data[id * 2 + i];

    for(int i = 0 ; i < 2 ; ++i)
        local[i] += 42;

    int tmp;

    for(int i = 0 ; i < 2 ; ++i){
        if(lane == i)
            tmp = local[i];
    }

    out[id] = tmp;
}

__global__ static
void strangeUnrolled(int* data, int* out, int N){
    auto id = blockDim.x * blockIdx.x + threadIdx.x;
    auto lane = threadIdx.x % 2;

    if(id >= N)
        return;

    int local[2];

    for(int i = 0 ; i < 2 ; ++i)
        local[i] = data[id * 2 + i];

    for(int i = 0 ; i < 2 ; ++i)
        local[i] += 42;

    int tmp;

    //manually unrolled loop
    if(lane == 0)
        tmp = local[0];
    if(lane == 1)
        tmp = local[1];

    out[id] = tmp;
}


__global__ static
void evenStrangerLoop(int* data, int* out, int N){
    auto id = blockDim.x * blockIdx.x + threadIdx.x;
    auto lane = threadIdx.x % 2;

    if(id >= N)
        return;

    int local[2];

    for(int i = 0 ; i < 2 ; ++i)
        local[i] = data[id * 2 + i];

    for(int i = 0 ; i < 2 ; ++i)
        local[i] += 42;

    int tmp;

    for(int i = 0 ; i < 2 ; ++i){
        if(lane >= i)
            tmp = local[i];
    }

    out[id] = tmp;
}
*/

// nvcc $CPPFLAGS -I ~/Master/libs/data/include/eigen3/ -ptx -lineinfo -src-in-ptx
// -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr inverse_test.cu nvcc $CPPFLAGS -I
// ~/Master/libs/data/include/eigen3/ -ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11
// --expt-relaxed-constexpr inverse_test.cu

template <typename ElementType, int ElementSize>
void coalescedCopyTest2(int ElementCount)
{
    std::cout << "Bytes per element = " << sizeof(ElementType) * ElementSize << std::endl;

    size_t readWrites = ElementSize * ElementCount * sizeof(ElementType) * 2;
    CUDA::PerformanceTestHelper test("Coalesced processing test. ElementSize: " + std::to_string(ElementSize) +
                                         " ElementCount: " + std::to_string(ElementCount),
                                     readWrites);

    thrust::host_vector<ElementType> data(ElementSize * ElementCount, 42);

    thrust::host_vector<ElementType> result(ElementSize * ElementCount + 1, -1);
    thrust::host_vector<ElementType> ref(ElementSize * ElementCount + 1, -1);

    for (int i = 0; i < int(data.size()); ++i)
    {
        data[i] = rand() % 10;
        ref[i]  = data[i] + 42;
    }


    thrust::device_vector<ElementType> d_data(data);
    thrust::device_vector<ElementType> d_result(result);


    {
        const int BLOCK_SIZE = 128;
        d_result             = result;
        float time;
        {
            CUDA::ScopedTimer t(time);
            copyUnCoalesced<ElementType, ElementSize, BLOCK_SIZE>
                <<<CUDA::getBlockCount(ElementCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("copyUnCoalesced", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);



    {
        const int BLOCK_SIZE = 128;
        d_result             = result;
        float time;
        {
            CUDA::ScopedTimer t(time);
            sharedMemoryUnCoalesced<ElementType, ElementSize, BLOCK_SIZE>
                <<<CUDA::getBlockCount(ElementCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("sharedMemoryUnCoalesced", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);

    {
        const int BLOCK_SIZE = 128;
        d_result             = result;
        float time;
        {
            CUDA::ScopedTimer t(time);
            sharedMemoryCoalesced<ElementType, ElementSize, BLOCK_SIZE>
                <<<CUDA::getBlockCount(ElementCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("sharedMemoryCoalesced", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);

    {
        const int BLOCK_SIZE = 128;
        d_result             = result;
        float time;
        {
            CUDA::ScopedTimer t(time);
            sharedMemoryCoalesced2<ElementType, ElementSize, BLOCK_SIZE>
                <<<CUDA::getBlockCount(ElementCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
            //            sharedMemoryCoalesced2<ElementType,ElementSize,BLOCK_SIZE> <<<
            //            CUDA::getBlockCount(ElementCount,BLOCK_SIZE),BLOCK_SIZE >>>(d_data,d_result);
        }
        test.addMeassurement("sharedMemoryCoalesced2", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);

    {
        const int BLOCK_SIZE = 128;
        d_result             = result;
        float time;
        {
            CUDA::ScopedTimer t(time);
            copyFullCoalesced<ElementType, ElementSize, BLOCK_SIZE>
                <<<CUDA::getBlockCount(ElementCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("copyFullCoalesced (no vector)", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);


    {
        const int BLOCK_SIZE = 128;
        d_result             = result;
        float time;
        {
            CUDA::ScopedTimer t(time);
            shuffleCopy<ElementType, ElementSize, BLOCK_SIZE>
                <<<CUDA::getBlockCount(ElementCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("shuffleCopy", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    //    for(int i = 0 ; i < ref.size() ; ++i){
    //        std::cout << ref[i] << " == " << d_result[i] << std::endl;
    //    }

    SAIGA_ASSERT(ref == d_result);

    {
        result = result;
        float time;
        {
            CUDA::ScopedTimer t(time);
            cudaMemcpy(thrust::raw_pointer_cast(d_result.data()), thrust::raw_pointer_cast(d_data.data()),
                       d_data.size() * sizeof(ElementType), cudaMemcpyDeviceToDevice);
        }
        test.addMeassurement("cudaMemcpy", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    return;
}

void coalescedCopyTest()
{
    CUDA_SYNC_CHECK_ERROR();
    //    coalescedCopyTest2<int,4>(1);
    //    coalescedCopyTest2<int,2>(1);
    //    coalescedCopyTest2<int,16>(1);
    //    coalescedCopyTest2<int,16>(3);
    //    coalescedCopyTest2<int,16>(5);
    //            coalescedCopyTest2<int,16>(1000 * 1000 + 1);
    //    coalescedCopyTest2<int,16>(32);
    coalescedCopyTest2<int, 32>(1000 * 1000 + 1);
    coalescedCopyTest2<int, 64>(1000 * 1000 + 1);
    CUDA_SYNC_CHECK_ERROR();
}

}  // namespace CUDA
}  // namespace Saiga
