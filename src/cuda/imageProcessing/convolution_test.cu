/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/convolution.h"
#include "saiga/cuda/imageProcessing/filter.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"
#include "saiga/time/performanceMeasure.h"

using std::cout;
using std::endl;

namespace Saiga {
namespace CUDA {


static void checkRes(const thrust::host_vector<float>& ref, const thrust::host_vector<float>& dst){
    for(int i = 0; i < (int)dst.size();++i){
        if(std::abs(dst[i]- ref[i]) > 1e-5){
            cout << "error " << i << " " << dst[i] << "!=" << ref[i] << endl;
            SAIGA_ASSERT(0);
        }
    }
}

template<int KERNEL_RADIUS>
void convolutionTest2(int w, int h){
    CUDA_SYNC_CHECK_ERROR();

    size_t N = w * h;
    size_t readWrites = N * 2 * sizeof(float);


    thrust::device_vector<float> src(N,0.1);
    thrust::device_vector<float> dest(N,0.1);
    thrust::device_vector<float> tmp(N,0.1);

    thrust::host_vector<float> h_src = src;
    thrust::host_vector<float> h_dest = dest;
    thrust::host_vector<float> h_tmp = dest;
    thrust::host_vector<float> h_ref = dest;

    ImageView<float> imgSrc(w,h,thrust::raw_pointer_cast(src.data()));
    ImageView<float> imgDst(w,h,thrust::raw_pointer_cast(dest.data()));
    ImageView<float> imgTmp(w,h,thrust::raw_pointer_cast(tmp.data()));


    ImageView<float> h_imgSrc(w,h,thrust::raw_pointer_cast(h_src.data()));
    ImageView<float> h_imgDst(w,h,thrust::raw_pointer_cast(h_dest.data()));
    ImageView<float> h_imgTmp(w,h,thrust::raw_pointer_cast(h_tmp.data()));

    int its = 50;
    float sigma = 2.0f;
    thrust::device_vector<float> d_kernel = createGaussianBlurKernel(KERNEL_RADIUS,sigma);
    thrust::host_vector<float> h_kernel(d_kernel);

    {
        for(int y = 0; y < h; ++y){
            for(int x = 0; x < w; ++x){
                h_imgSrc(x,y) = (rand()%3) - 1;
            }
        }
        src = h_src;
    }

    Saiga::CUDA::PerformanceTestHelper pth("convolutionTest radius=" + std::to_string(KERNEL_RADIUS)
                                           + " ImageSize: " + std::to_string(w) + "x" + std::to_string(h), readWrites);

    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for(int y = 0; y < h; ++y){
                for(int x = 0; x < w; ++x){
                    float sum = 0;
                    for (int j=-KERNEL_RADIUS;j<=KERNEL_RADIUS;j++){
                        int ny = std::min(std::max(0,y+j),h-1);
                        float innerSum = 0;
                        for (int i=-KERNEL_RADIUS;i<=KERNEL_RADIUS;i++){
                            int nx = std::min(std::max(0,x+i),w-1);
                            innerSum += h_imgSrc(nx,ny) * h_kernel[i+KERNEL_RADIUS];
                        }
                        sum += innerSum * h_kernel[j+KERNEL_RADIUS];
                    }
                    h_imgDst(x,y) = sum;
                }
            }
        }
        pth.addMeassurement("CPU Convolve",time);
        h_ref = h_dest;
    }

    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for(int y = 0; y < h; ++y){
                for(int x = 0; x < w; ++x){
                    float sum = 0;
                    for (int j=-KERNEL_RADIUS;j<=KERNEL_RADIUS;j++){
                        int nx = std::min(std::max(0,x+j),w-1);
                        sum += h_imgSrc(nx,y) * h_kernel[j+KERNEL_RADIUS];
                    }
                    h_imgTmp(x,y) = sum;
                }
            }

            for(int x = 0; x < w; ++x){
                for(int y = 0; y < h; ++y){
                    float sum = 0;
                    for (int j=-KERNEL_RADIUS;j<=KERNEL_RADIUS;j++){
                        int ny = std::min(std::max(0,y+j),h-1);
                        sum += h_imgTmp(x,ny) * h_kernel[j+KERNEL_RADIUS];
                    }
                    h_imgDst(x,y) = sum;
                }
            }
        }
        pth.addMeassurement("CPU Convolve Separate",time);
        SAIGA_ASSERT(h_ref == h_dest);
    }


    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveSinglePassSeparateOuterLinear(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        });


        pth.addMeassurement("convolveSinglePassSeparateOuterLinear",st.median);
        checkRes(h_ref,thrust::host_vector<float>(dest));
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;


        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveSinglePassSeparateOuterHalo(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        });
        pth.addMeassurement("convolveSinglePassSeparateOuterHalo",st.median);
        checkRes(h_ref,thrust::host_vector<float>(dest));
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveSinglePassSeparateInner(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        });
        pth.addMeassurement("convolveSinglePassSeparateInner",st.median);
        checkRes(h_ref,thrust::host_vector<float>(dest));
    }


    {
        dest = src;
        tmp = src;
        thrust::device_vector<float> d_kernel = h_kernel;

        auto st1 = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveRow(imgSrc,imgTmp,d_kernel,KERNEL_RADIUS);
        });
        pth.addMeassurement("GPU Convolve Separate Row",st1.median);

        auto st2 = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveCol(imgTmp,imgDst,d_kernel,KERNEL_RADIUS);
        });
        pth.addMeassurement("GPU Convolve Separate Col",st2.median);
        pth.addMeassurement("GPU Convolve Separate Total",st1.median + st2.median);

        checkRes(h_ref,thrust::host_vector<float>(dest));
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()),thrust::raw_pointer_cast(src.data()),N * sizeof(int),cudaMemcpyDeviceToDevice);
        });
        pth.addMeassurement("cudaMemcpy", st.median);
    }
    CUDA_SYNC_CHECK_ERROR();

}

void convolutionTest()
{
    //    convolutionTest2<3>(17,53);
    convolutionTest2<3>(2048,1024);
    convolutionTest2<4>(2048,1024);
    convolutionTest2<5>(2048,1024);
    convolutionTest2<6>(2048,1024);
    convolutionTest2<7>(2048,1024);
}

}
}
