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
        //        CUDA::convolutionTest();


        {

            //load an image from file
//            Image img;
            TemplatedImage<3,8,ImageElementFormat::UnsignedNormalized> img;
            loadImage("textures/redie.png",img);

            //copy the image to the gpu
            CUDA::CudaImage<uchar3> cimg(img);
            CUDA::CudaImage<uchar4> cimg4(cimg.width,cimg.height);
            CUDA::CudaImage<float> cimggray(cimg.width,cimg.height);

            CUDA::convertRGBtoRGBA(cimg,cimg4,255);
            CUDA::convertRGBAtoGrayscale(cimg4,cimggray);

            //copy back to cpu
            TemplatedImage<4,8,ImageElementFormat::UnsignedNormalized> res4;
            TemplatedImage<1,32,ImageElementFormat::FloatingPoint> resGray;
            TemplatedImage<1,8,ImageElementFormat::UnsignedNormalized> resGray8;
            res4 = (Image)cimg4;
            resGray = (Image)cimggray;
            resGray8 = resGray.convertImage<1,8,ImageElementFormat::UnsignedNormalized>();

            saveImage("test4.png",res4);
            saveImage("testgray.png",resGray8);

        }
        CUDA::destroyCUDA();
    }
}
