#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"

namespace Saiga {
namespace CUDA {

template<unsigned int THREADS_PER_BLOCK>
__global__ static
void copy(void* src, void* dest, unsigned int size) {
    using vector_type = int2;
    vector_type* src2 = reinterpret_cast<vector_type*>(src);
    vector_type* dest2 = reinterpret_cast<vector_type*>(dest);

    //This copy kernel is only correct when size%sizeof(vector_type)==0
    auto numElements = size / sizeof(vector_type);

    for(auto id = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x; id < numElements ; id += gridDim.x * THREADS_PER_BLOCK){
        dest2[id] = src2[id];
    }
}


template<unsigned int ELEMENTS_PER_THREAD, unsigned int THREADS_PER_BLOCK>
__global__ static
void copyFixed(void* src, void* dest, unsigned int size) {
    using vector_type = int2;
    vector_type* src2 = reinterpret_cast<vector_type*>(src);
    vector_type* dest2 = reinterpret_cast<vector_type*>(dest);

    //This copy kernel is only correct when size%(sizeof(vector_type)*ELEMENTS_PER_THREAD)==0
    auto numElements = size / sizeof(vector_type);

    auto tid = ELEMENTS_PER_THREAD * THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;

    if(tid >= numElements)
        return;

#pragma unroll
    for(int i = 0 ; i < ELEMENTS_PER_THREAD; ++i){
        auto id = tid + THREADS_PER_BLOCK * i;
        dest2[id] = src2[id];
    }
}



//nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr integrate_test.cu


void bandwidthTest(){
    CUDA_SYNC_CHECK_ERROR();
    size_t N = 100 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int);

    CUDA::PerformanceTestHelper pth("Memcpy", readWrites);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);

    {
        float time;
        {
            CUDA::CudaScopedTimer t(time);
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()),thrust::raw_pointer_cast(src.data()),N * sizeof(int),cudaMemcpyDeviceToDevice);

        }
        pth.addMeassurement("cudaMemcpy", time);
    }

    {
        float time;
        {
            CUDA::CudaScopedTimer t(time);
            thrust::copy(src.begin(),src.end(),dest.begin());
        }
        pth.addMeassurement("thrust::copy", time);
    }

    {
        float time;
        const        size_t THREADS_PER_BLOCK = 256;
        static const size_t MAX_BLOCKS = CUDA::max_active_blocks(copy<THREADS_PER_BLOCK>,THREADS_PER_BLOCK);
        const        size_t NUM_BLOCKS = std::min<size_t>(CUDA::getBlockCount(N/sizeof(int2),THREADS_PER_BLOCK),MAX_BLOCKS);
        {
            CUDA::CudaScopedTimer t(time);
            copy<THREADS_PER_BLOCK><<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(src.data()),thrust::raw_pointer_cast(dest.data()),N * sizeof(int));
        }
        pth.addMeassurement("my memcpy (" + std::to_string(NUM_BLOCKS) + " blocks)", time);
    }

    {
        float time;
        const        size_t THREADS_PER_BLOCK = 256;
        const        int ELEMENTS_PER_THREAD = 4;
        const        size_t NUM_BLOCKS = CUDA::getBlockCount(N*sizeof(int)/(sizeof(int2)*ELEMENTS_PER_THREAD),THREADS_PER_BLOCK);
        {
            CUDA::CudaScopedTimer t(time);
            copyFixed<ELEMENTS_PER_THREAD,THREADS_PER_BLOCK><<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(src.data()),thrust::raw_pointer_cast(dest.data()),N * sizeof(int));
        }
        pth.addMeassurement("my memcpy (" + std::to_string(NUM_BLOCKS) + " blocks)", time);
    }

    {
        float time;
        const        size_t THREADS_PER_BLOCK = 256;
        const        size_t NUM_BLOCKS = CUDA::getBlockCount(N*sizeof(int)/sizeof(int2),THREADS_PER_BLOCK);
        {
            CUDA::CudaScopedTimer t(time);
            copy<THREADS_PER_BLOCK><<<NUM_BLOCKS,THREADS_PER_BLOCK>>>(thrust::raw_pointer_cast(src.data()),thrust::raw_pointer_cast(dest.data()),N * sizeof(int));
        }
        pth.addMeassurement("my memcpy (" + std::to_string(NUM_BLOCKS) + " blocks)", time);
   }
    CUDA_SYNC_CHECK_ERROR();

}

}
}
