/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/cuda/CudaInfo.h"
#include "saiga/core/image/image.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/core/tests/test.h"
#include "saiga/core/util/crash.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/tests/test.h"

using namespace Saiga;

#if defined(SAIGA_DLL_EXPORTS)
#    error build shared still defined
#endif

int main(int argc, char* argv[])
{
    catchSegFaults();

    {
        // CUDA tests
        CUDA::initCUDA();
        CUDA::convolutionTest();

        CUDA::imageProcessingTest();

        //        return 1;
        {
#if 0

            //load an image from file
//            Image img;
            TemplatedImage<ucvec3> img("textures/redie.png");
//            loadImage(img);

            //copy the image to the gpu
            CUDA::CudaImage<uchar3> cimg(img.getImageView());
            CUDA::CudaImage<uchar4> cimg4(cimg.height,cimg.width);
            CUDA::CudaImage<float> cimggray(cimg.height,cimg.width);
            CUDA::CudaImage<float> cimgtmp(cimg.height,cimg.width);
            CUDA::CudaImage<float> cimgblurred(cimg.height,cimg.width);
            CUDA::CudaImage<float> cimggrayhalf(cimg.height/2,cimg.width/2);
            CUDA::CudaImage<float> cimggraydouble(cimg.height*2,cimg.width*2);


			//test image copy from cpp file
			CUDA::CudaImage<uchar4> cimg5 = cimg4;

            CUDA::convertRGBtoRGBA(cimg,cimg4,255);
            CUDA::convertRGBAtoGrayscale(cimg4,cimggray);
            CUDA::scaleDown2EveryOther<float>(cimggray,cimggrayhalf);
            CUDA::scaleUp2Linear(cimggray,cimggraydouble);

            auto filter = CUDA::createGaussianBlurKernel(4,2);
//            CUDA::gaussianBlur(cimggray,cimgblurred,2,4);
            CUDA::applyFilterSeparate(cimggray,cimgblurred,cimgtmp,filter,filter);


            //copy back to cpu
            TemplatedImage<4,8,ImageElementFormat::UnsignedNormalized> res4;
            TemplatedImage<1,32,ImageElementFormat::FloatingPoint> resGray;
//            res4.setFormatFromImageView(cimg4);
//            CUDA::copyImage(cimg4,res4.getImageView<uchar4>(),cudaMemcpyDeviceToHost);
            CUDA::convert(cimg4,res4);

            saveImage("debug/test4.png",res4);

            TemplatedImage<1,8,ImageElementFormat::UnsignedNormalized> resGray8;

            CUDA::convert(cimggray,resGray);
//            resGray = (Image)cimggray;
//            resGray.setFormatFromImageView(cimggray);
//            CUDA::copyImage(cimg4,res4.getImageView<float>(),cudaMemcpyDeviceToHost);
            resGray8 = resGray.convertImage<1,8,ImageElementFormat::UnsignedNormalized>();
            saveImage("debug/testgray.png",resGray8);

//            resGray = (Image)cimggrayhalf;
            CUDA::convert(cimggrayhalf,resGray);
            resGray8 = resGray.convertImage<1,8,ImageElementFormat::UnsignedNormalized>();
            saveImage("debug/testgrayhalf.png",resGray8);

//            resGray = (Image)cimggraydouble;
            CUDA::convert(cimggraydouble,resGray);
            resGray8 = resGray.convertImage<1,8,ImageElementFormat::UnsignedNormalized>();
            saveImage("debug/testgraydouble.png",resGray8);

//            resGray = (Image)cimgblurred;
            CUDA::convert(cimgblurred,resGray);
            resGray8 = resGray.convertImage<1,8,ImageElementFormat::UnsignedNormalized>();
            saveImage("debug/testgrayblurred.png",resGray8);
#endif
        }
        CUDA::destroyCUDA();
    }
}
