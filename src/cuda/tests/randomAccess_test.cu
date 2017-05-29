#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/cr_array_view.h"

#include <random>


namespace CUDA {



template<typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ static
void randomAccessSimple(array_view<int> indices, array_view<T> data, array_view<T> result){

    auto N = indices.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    //grid stride loop
    for(auto id = ti.thread_id ; id < N; id += ti.grid_size){
        int index = indices[id];
        auto el = data[index];
        result[id] = el;
    }
}

template<typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ static
void randomAccessConstRestricted(array_view<int> indices, const T* __restrict__ data, array_view<T> result){

    auto N = indices.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    //grid stride loop
    for(auto id = ti.thread_id ; id < N; id += ti.grid_size){
        int index = indices[id];
        auto el = data[index];
        result[id] = el;
    }
}


template<typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ static
void randomAccessLdg(array_view<int> indices, array_view<T> data, array_view<T> result){

    auto N = indices.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    //grid stride loop
    for(auto id = ti.thread_id ; id < N; id += ti.grid_size){
        int index = indices[id];
        auto el = CUDA::ldg(data.data() + index);
        result[id] = el;
    }
}

template<typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ static
void randomAccess_cr_array_view(array_view<int> indices, const cr_array_view<T> data, array_view<T> result){

    auto N = indices.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    //grid stride loop
    for(auto id = ti.thread_id ; id < N; id += ti.grid_size){
        int index = indices[id];
        auto el = data.data_[index];
        result[id] = el;
    }
}

static texture<int,1,cudaReadModeElementType> dataTexture;

template<typename T, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ static
void randomAccessTexture(array_view<int> indices, array_view<T> result){

    auto N = result.size();

    CUDA::ThreadInfo<BLOCK_SIZE> ti;
    //grid stride loop
    for(auto id = ti.thread_id ; id < N; id += ti.grid_size){
        int index = indices[id];
        auto el = tex1Dfetch(dataTexture,index);
        result[id] = el;
    }
}


/*
namespace test{

__global__ static
void test1(const int* __restrict__ data, int* result, int N){
    result[threadIdx.x] = data[threadIdx.x];
}


template<typename T>
struct my_array{
public:
    const T* __restrict__ data_;
};

__global__ static
void test2(const my_array<int> data, int* result, int N){
    result[threadIdx.x] = data.data_[threadIdx.x];
}

}
*/


//nvcc $CPPFLAGS -I ~/Master/libs/data/include/eigen3/ -ptx -lineinfo -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr inverse_test.cu
//nvcc $CPPFLAGS -I ~/Master/libs/data/include/eigen3/ -ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr randomAccess_test.cu

template<typename ElementType>
void randomAccessTest2(int numIndices, int numElements){



    size_t readWrites =  numElements * sizeof(ElementType) + numIndices * sizeof(ElementType) + numElements * sizeof(int);
    CUDA::PerformanceTestHelper test(
                "Coalesced processing test. numIndices: "
                + std::to_string(numIndices)
                +" numElements: "  + std::to_string(numElements),
                readWrites);

    thrust::host_vector<int> indices(numElements);
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



    for(int i = 0 ; i < numElements; ++i){
        indices[i] = dist2(mt);
        ref[i] = data[indices[i]];
    }



    thrust::device_vector<int> d_indices(indices);
    thrust::device_vector<ElementType> d_data(data);
    thrust::device_vector<ElementType> d_result(result);


    {
        const int BLOCK_SIZE = 128;
        d_result = result;
        float time;
        {
            CUDA::CudaScopedTimer t(time);
            randomAccessSimple<ElementType,BLOCK_SIZE> <<< CUDA::getBlockCount(numElements,BLOCK_SIZE),BLOCK_SIZE >>>(d_indices,d_data,d_result);
        }
        test.addMeassurement("randomAccessSimple",time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);

    {
        const int BLOCK_SIZE = 128;
        d_result = result;
        float time;
        {
            CUDA::CudaScopedTimer t(time);
            const ElementType* tmp = thrust::raw_pointer_cast(d_data.data());
            randomAccessConstRestricted<ElementType,BLOCK_SIZE> <<< CUDA::getBlockCount(numElements,BLOCK_SIZE),BLOCK_SIZE >>>(d_indices,tmp,d_result);
        }
        test.addMeassurement("randomAccessConstRestricted",time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);

    {
        const int BLOCK_SIZE = 128;
        d_result = result;
        float time;
        {
            CUDA::CudaScopedTimer t(time);
            randomAccessLdg<ElementType,BLOCK_SIZE> <<< CUDA::getBlockCount(numElements,BLOCK_SIZE),BLOCK_SIZE >>>(d_indices,d_data,d_result);
        }
        test.addMeassurement("randomAccessLdg",time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);

    {
        const int BLOCK_SIZE = 128;
        d_result = result;
        float time;
        {
            cr_array_view<ElementType> d;
            d.data_ = thrust::raw_pointer_cast(d_data.data());
            d.n = d_data.size();
            CUDA::CudaScopedTimer t(time);
            randomAccess_cr_array_view<ElementType,BLOCK_SIZE> <<< CUDA::getBlockCount(numElements,BLOCK_SIZE),BLOCK_SIZE >>>(d_indices,d,d_result);
        }
        test.addMeassurement("randomAccess_cr_array_view",time);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);


    {
        cudaBindTexture(0, dataTexture, thrust::raw_pointer_cast(d_data.data()), d_data.size()*sizeof(ElementType));
        const int BLOCK_SIZE = 128;
        d_result = result;
        float time;
        {
            CUDA::CudaScopedTimer t(time);
            randomAccessTexture<ElementType,BLOCK_SIZE> <<< CUDA::getBlockCount(numElements,BLOCK_SIZE),BLOCK_SIZE >>>(d_indices,d_result);
        }
        test.addMeassurement("randomAccessTexture",time);
        cudaUnbindTexture(dataTexture);
        CUDA_SYNC_CHECK_ERROR();
    }

    SAIGA_ASSERT(ref == d_result);

    return;


}

void randomAccessTest(){
    CUDA_SYNC_CHECK_ERROR();
    randomAccessTest2<int>(1,       10 * 1000 * 1000);
    randomAccessTest2<int>(100,     10 * 1000 * 1000);
    randomAccessTest2<int>(1000,    10 * 1000 * 1000);
    randomAccessTest2<int>(10000,   10 * 1000 * 1000);
    randomAccessTest2<int>(100000,  10 * 1000 * 1000);
    randomAccessTest2<int>(1000000, 10 * 1000 * 1000);
    randomAccessTest2<int>(10000000,10 * 1000 * 1000);
//    randomAccessTest2<int>(10 * 1000 * 1000, 10 * 1000 * 1000);
    CUDA_SYNC_CHECK_ERROR();
}

}
