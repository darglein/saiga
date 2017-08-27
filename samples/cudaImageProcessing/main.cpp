/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/crash.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/cusparseHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/random.h"
#include "saiga/tests/test.h"

#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/image/image.h"
#include "saiga/image/templatedImage.h"

using namespace Saiga;

int main(int argc, char *argv[]) {

    catchSegFaults();

    {
        //CUDA tests
        CUDA::initCUDA();
                CUDA::convolutionTest();

        CUDA::imageProcessingTest();

//        return 1;
        {

            //load an image from file
//            Image img;
            TemplatedImage<3,8,ImageElementFormat::UnsignedNormalized> img;
            loadImage("textures/redie.png",img);

            //copy the image to the gpu
            CUDA::CudaImage<uchar3> cimg(img.getImageView<uchar3>());
            CUDA::CudaImage<uchar4> cimg4(cimg.width,cimg.height);
            CUDA::CudaImage<float> cimggray(cimg.width,cimg.height);
            CUDA::CudaImage<float> cimgtmp(cimg.width,cimg.height);
            CUDA::CudaImage<float> cimgblurred(cimg.width,cimg.height);
            CUDA::CudaImage<float> cimggrayhalf(cimg.width/2,cimg.height/2);
            CUDA::CudaImage<float> cimggraydouble(cimg.width*2,cimg.height*2);

            CUDA::convertRGBtoRGBA(cimg,cimg4,255);
            CUDA::convertRGBAtoGrayscale(cimg4,cimggray);
            CUDA::scaleDown2EveryOther(cimggray,cimggrayhalf);
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

        }
        CUDA::destroyCUDA();
    }
}
