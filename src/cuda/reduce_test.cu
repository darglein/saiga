#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"
#include "saiga/cuda/reduce.h"

//nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr integrate_test.cu

namespace CUDA {


void reduceTest(){

    /*
    int N = 10 * 1000 * 1000;
    int numIterations = 10;
    thrust::device_vector<int> v(N,1);
    thrust::host_vector<int> h(N,1);

    std::cout << "Num elements: " << N << " Memory: " << double(N)*sizeof(int)/1000000.0 << "mb" << std::endl;


    {
        ScopedTimerPrint t("cpu reduce");
        int sum = 0 ;
        for(int i = 0 ; i < N; ++i){
            sum += h[i];
        }
        SAIGA_ASSERT(sum == N);
    }
    {
        CUDA::CudaScopedTimerPrint t("thrust reduce");
        int sum = thrust::reduce(v.begin(),v.end());
        SAIGA_ASSERT(sum == N);
    }

    {
        const int blockSize = 256;
        auto numBlocks = CUDA::max_active_blocks(CUDA::reduceBlockShared<int,blockSize>,blockSize,0);
        thrust::device_vector<int> res(1);
        float time;
        {
            CUDA::CudaScopedTimer t2(time);
            for(int i = 0 ;i < numIterations; ++i){
                CUDA::reduceBlockShared<int,blockSize> <<<numBlocks,blockSize>>>(v, thrust::raw_pointer_cast(res.data()));
            }
        }
        std::cout << "reduceBlockShared - " << time/numIterations << "ms." << std::endl;
        int sum = res[0]/numIterations;
        SAIGA_ASSERT(sum == N);
    }

    {
        const int blockSize = 256;
        auto numBlocks = CUDA::max_active_blocks(CUDA::reduceBlockSharedAtomic<int,blockSize>,blockSize,0);
        thrust::device_vector<int> res(1);
        float time;
        {
            CUDA::CudaScopedTimer t2(time);
            for(int i = 0 ;i < numIterations; ++i){
                CUDA::reduceBlockSharedAtomic<int,blockSize><<<numBlocks,blockSize>>>(v,thrust::raw_pointer_cast(res.data()));
            }
        }
        std::cout << "reduceBlockSharedAtomic - " << time/numIterations << "ms." << std::endl;
        int sum = res[0]/numIterations;
        SAIGA_ASSERT(sum == N);
    }

    {
        const int blockSize = 256;
        auto numBlocks = CUDA::max_active_blocks(CUDA::reduceAtomic<int,blockSize>,blockSize,0);
        thrust::device_vector<int> res(1);
        float time;
        {
            CUDA::CudaScopedTimer t2(time);
            for(int i = 0 ;i < numIterations; ++i){
                CUDA::reduceAtomic<int,blockSize><<<numBlocks,blockSize>>>(v,thrust::raw_pointer_cast(res.data()));
            }
        }
        std::cout << "reduceAtomic - " << time/numIterations << "ms." << std::endl;
        int sum = res[0]/numIterations;
        SAIGA_ASSERT(sum == N);
    }
    */
}

}
