/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"
#include "saiga/image/templatedImage.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include <algorithm>
#include "saiga/util/tostring.h"

namespace Saiga {
namespace CUDA {


//nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr integrate_test.cu


void imageProcessingTest(){
    CUDA_SYNC_CHECK_ERROR();


    TemplatedImage<3,8,ImageElementFormat::UnsignedNormalized> img;
    //    loadImage("textures/redie.png",img);
    loadImage("textures/landscape.jpg",img);


    size_t readWrites = img.getSize() * 2;
    CUDA::PerformanceTestHelper pth("imageProcessingTest ImageSize: " + to_string(img.width) + "x" + to_string(img.height), readWrites);


    CUDA::CudaImage<uchar3> cimg(img);
    CUDA::CudaImage<uchar4> cimg4(cimg.width,cimg.height);
    CUDA::CudaImage<float> cimggray(cimg.width,cimg.height);
    CUDA::CudaImage<float> cimgtmp(cimg.width,cimg.height);
    CUDA::CudaImage<float> cimgblurred(cimg.width,cimg.height);
    CUDA::CudaImage<float> cimggrayhalf(cimg.width/2,cimg.height/2);
    CUDA::CudaImage<float> cimggraydouble(cimg.width*2,cimg.height*2);

    {
        int its = 5;
        float time;
        pth.updateBytes(cimggraydouble.size());
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::fill(cimggraydouble,0.5f);
        }
        pth.addMeassurement("fill", time/its);
    }

    {
        int its = 5;
        float time;
        pth.updateBytes(cimg.size() + cimg4.size());
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::convertRGBtoRGBA(cimg,cimg4,255);
        }
        pth.addMeassurement("convertRGBtoRGBA", time/its);
    }

    {
        int its = 5;
        float time;
        int radius = 4;
        pth.updateBytes(cimggray.size() + cimggray.size());
        auto kernel = createGaussianBlurKernel(radius,2);
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::applyFilterSeparate(cimggray,cimgtmp,cimgblurred,kernel,kernel);
        }
        pth.addMeassurement("applyFilterSeparate", time/its);
    }

    {
        int its = 5;
        float time;
        int radius = 4;
        pth.updateBytes(cimggray.size() + cimggray.size());
        auto kernel = createGaussianBlurKernel(radius,2);
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::applyFilterSeparateSinglePass(cimggray,cimgblurred,kernel);
        }
        pth.addMeassurement("applyFilterSeparateSinglePass", time/its);
    }

    {
        int its = 5;
        float time;
        int radius = 4;
        auto kernel = createGaussianBlurKernel(radius,2);
        pth.updateBytes(cimggray.size() + cimggray.size());
        //        setGaussianBlurKernel(2,radius);
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::convolveRow(cimggray,cimgtmp,kernel,radius);
        }
        pth.addMeassurement("convolveRow", time/its);
    }

    {
        int its = 5;
        float time;
        int radius = 4;
        auto kernel = createGaussianBlurKernel(radius,2);
        pth.updateBytes(cimggray.size() + cimggray.size());
        //        setGaussianBlurKernel(2,radius);
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::convolveCol(cimggray,cimgtmp,kernel,radius);
        }
        pth.addMeassurement("convolveCol", time/its);
    }

    {
        int its = 5;
        float time;
        pth.updateBytes(cimg4.size() + cimggray.size());
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::convertRGBAtoGrayscale(cimg4,cimggray);
        }
        pth.addMeassurement("convertRGBAtoGrayscale", time/its);
    }

    {
        int its = 5;
        float time;
        pth.updateBytes(cimggray.size()/4 + cimggrayhalf.size());
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::scaleDown2EveryOther(cimggray,cimggrayhalf);
        }
        pth.addMeassurement("scaleDown2EveryOther", time/its);
    }


    {
        int its = 5;
        float time;
        pth.updateBytes(cimggray.size() + cimggraydouble.size());
        {
            CUDA::CudaScopedTimer t(time);
            for(int i = 0; i < its; ++i)
                CUDA::scaleUp2Linear(cimggray,cimggraydouble);
        }
        pth.addMeassurement("scaleUp2Linear", time/its);
    }

    CUDA_SYNC_CHECK_ERROR();

}

}
}
