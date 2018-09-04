/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/glm.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/tests/test_helper.h"
#include <random>
#include <fstream>

using Saiga::ArrayView;
using Saiga::CUDA::ThreadInfo;

std::ofstream outstrm;


HD inline
uint32_t simpleRand(uint32_t state)
{
    /* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}


template<typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static
void randomAccessSimple(ArrayView<T> data, ArrayView<T> result)
{
    ThreadInfo<BLOCK_SIZE> ti;
    if(ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for(int i = 0; i < K; ++i)
    {
        r = simpleRand(r);
        auto index = r % data.size();
        sum += data[index];
    }
    result[ti.thread_id] = sum;
}

template<typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static
void randomAccessConstRestricted(ArrayView<T> vdata, const T* __restrict__ data, ArrayView<T> result)
{
    ThreadInfo<BLOCK_SIZE> ti;
    if(ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for(int i = 0; i < K; ++i)
    {
        r = simpleRand(r);
        auto index = r % vdata.size();
        sum += data[index];
    }
    result[ti.thread_id] = sum;
}


template<typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static
void randomAccessLdg(ArrayView<T> data, ArrayView<T> result)
{

    ThreadInfo<BLOCK_SIZE> ti;
    if(ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for(int i = 0; i < K; ++i)
    {
        r = simpleRand(r);
        auto index = r % data.size();
        sum += Saiga::CUDA::ldg(data.data() + index);
    }
    result[ti.thread_id] = sum;
}


static texture<int,1,cudaReadModeElementType> dataTexture;

template<typename T, unsigned int BLOCK_SIZE, unsigned int K>
__global__ static
void randomAccessTexture( ArrayView<T> data, ArrayView<T> result)
{

    ThreadInfo<BLOCK_SIZE> ti;
    if(ti.thread_id >= result.size()) return;

    uint32_t r = ti.thread_id * 17;

    T sum = 0;
    for(int i = 0; i < K; ++i)
    {
        r = simpleRand(r);
        auto index = r % data.size();
        sum += tex1Dfetch(dataTexture,index);
    }
    result[ti.thread_id] = sum;
}


template<typename ElementType>
void randomAccessTest2(int numIndices, int numElements)
{

    const int K = 16;

    outstrm << numIndices * sizeof(int) << ",";

    size_t readWrites =
            numElements * sizeof(ElementType) +
            numElements * sizeof(int) * K;

    Saiga::CUDA::PerformanceTestHelper test(
                "Coalesced processing test. numIndices: "
                + std::to_string(numIndices)
                +" numElements: "  + std::to_string(numElements),
                readWrites);

    thrust::host_vector<int> indices(numElements * K);
    thrust::host_vector<ElementType> data(numIndices);
    thrust::host_vector<ElementType> result(numElements,0);
    thrust::host_vector<ElementType> ref(numElements);


    std::random_device rd;
        std::mt19937 mt(rd());
        std::uniform_int_distribution<int> dist1(1, 10);
        std::uniform_int_distribution<int> dist2(0, numIndices - 1);


    for(int i = 0 ; i < numIndices; ++i){
        data[i] = dist1(mt);
    }



    for(int i = 0 ; i < indices.size(); ++i){
        indices[i] = dist2(mt);
//        ref[i] = data[indices[i]];
    }



    thrust::device_vector<int> d_indices(indices);
    thrust::device_vector<ElementType> d_data(data);
    thrust::device_vector<ElementType> d_result(result);

    int its = 5;

    const int BLOCK_SIZE = 128;
    const int BLOCKS = Saiga::CUDA::getBlockCount(numElements,BLOCK_SIZE);

    {
        d_result = result;

        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            randomAccessSimple<ElementType,BLOCK_SIZE,K> <<< BLOCKS,BLOCK_SIZE >>>(d_data,d_result);
        });
        test.addMeassurement("randomAccessSimple",st.median);
        outstrm << test.bandwidth(st.median) << ",";
        CUDA_SYNC_CHECK_ERROR();
    }

//    SAIGA_ASSERT(ref == d_result);

    {
        d_result = result;

        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
//            randomAccessSimple<ElementType,BLOCK_SIZE> <<< CUDA::getBlockCount(numElements,BLOCK_SIZE),BLOCK_SIZE >>>(d_indices,d_data,d_result);
//            const ElementType* tmp = thrust::raw_pointer_cast(d_data.data());
            randomAccessConstRestricted<ElementType,BLOCK_SIZE,K> <<< BLOCKS,BLOCK_SIZE >>>(d_data,d_data.data().get(),d_result);
        });
        test.addMeassurement("randomAccessConstRestricted",st.median);
        outstrm << test.bandwidth(st.median) << ",";

        CUDA_SYNC_CHECK_ERROR();
    }

//    SAIGA_ASSERT(ref == d_result);

    {
        d_result = result;

        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            randomAccessLdg<ElementType,BLOCK_SIZE,K> <<< BLOCKS,BLOCK_SIZE >>>(d_data,d_result);

        });
        test.addMeassurement("randomAccessLdg",st.median);
        outstrm << test.bandwidth(st.median) << ",";

        CUDA_SYNC_CHECK_ERROR();
    }
#if 1

//    SAIGA_ASSERT(ref == d_result);



    {
        cudaBindTexture(0, dataTexture, thrust::raw_pointer_cast(d_data.data()), d_data.size()*sizeof(ElementType));
        d_result = result;

        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            randomAccessTexture<ElementType,BLOCK_SIZE,K> <<< BLOCKS,BLOCK_SIZE>>>(d_data,d_result);
        });
        test.addMeassurement("randomAccessTexture",st.median);
        outstrm << test.bandwidth(st.median);

        cudaUnbindTexture(dataTexture);
        CUDA_SYNC_CHECK_ERROR();
    }
#endif

//    SAIGA_ASSERT(ref == d_result);

    outstrm << endl;
    return;


}


int main(int argc, char *argv[])
{
    outstrm.open("out.csv");
    outstrm << "size,simple,cr,ldg,texture" << endl;
    for(int i = 0; i < 24; ++i)
    {
        randomAccessTest2<int>(1 << i,       1 * 1024 * 1024);
//        randomAccessTest2<int>(1024 * 1 * (i+1),       8 * 1024 * 1024);
    }
//    return;
//    randomAccessTest2<int>(1 << 0,       1 * 1024 * 1024);
//    randomAccessTest2<int>(1 << 1,       1 * 1024 * 1024);
//    randomAccessTest2<int>(1 << 2,       1 * 1024 * 1024);
//    randomAccessTest2<int>(1 << 3,       1 * 1024 * 1024);

    CUDA_SYNC_CHECK_ERROR();
    return 0;
}

