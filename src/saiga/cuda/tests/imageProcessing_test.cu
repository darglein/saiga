/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/core/util/tostring.h"

#include <algorithm>


namespace Saiga
{
namespace CUDA
{
// nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr
// integrate_test.cu


void imageProcessingTest()
{
#if 0
    CUDA_SYNC_CHECK_ERROR();


    TemplatedImage<uchar3> img("textures/landscape.jpg");
    //    loadImage("textures/redie.png",img);
//    loadImage("textures/landscape.jpg",img);


    size_t readWrites = img.size() * 2;
    CUDA::PerformanceTestHelper pth("imageProcessingTest ImageSize: " + to_string(img.width) + "x" + to_string(img.height), readWrites);


    CUDA::CudaImage<uchar3> cimg(img.getImageView());
    CUDA::CudaImage<uchar4> cimg4(cimg.height,cimg.width);
    CUDA::CudaImage<float> cimggray(cimg.height,cimg.width);
    CUDA::CudaImage<float> cimgtmp(cimg.height,cimg.width);
    CUDA::CudaImage<float> cimgblurred(cimg.height,cimg.width);
    CUDA::CudaImage<float> cimggrayhalf(cimg.height/2,cimg.width/2);
    CUDA::CudaImage<float> cimggraydouble(cimg.height*2,cimg.width*2);

    CUDA::CudaImage<float> cimgmulti1v(cimg.height,cimg.width*6);
    CUDA::CudaImage<float> cimgmulti2v(cimg.height,cimg.width*5);
    ImageArrayView<float> cimgmulti1( ImageView<float>(cimg.width,cimg.height,cimgmulti1v.data()), 6 );
    ImageArrayView<float> cimgmulti2( ImageView<float>(cimg.width,cimg.height,cimgmulti1v.data()), 5 );

     int its = 50;

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
                CUDA::scaleDown2EveryOther<float>(cimggray,cimggrayhalf);
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


    {
        pth.updateBytes(cimggray.size() * 3);
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            CUDA::subtract(cimggray,cimgtmp,cimgblurred);
        });
        pth.addMeassurement("subtract",st.median);
    }


     {
         pth.updateBytes(cimggray.size() * (cimgmulti1.n+cimgmulti2.n) );
         auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
         {
             CUDA::subtractMulti(cimgmulti1,cimgmulti2);
         });
         pth.addMeassurement("subtractMulti",st.median);
     }


     {
         pth.updateBytes(cimggray.size() * 2);
         auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
         {
             cudaMemcpy(cimggray.data(),cimgtmp.data(),cimggray.size(),cudaMemcpyDeviceToDevice);
         });
         pth.addMeassurement("cudaMemcpy", st.median);
     }

    CUDA_SYNC_CHECK_ERROR();
#endif
}

}  // namespace CUDA
}  // namespace Saiga
