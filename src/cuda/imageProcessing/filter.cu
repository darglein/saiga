/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/filter.h"
#include "saiga/cuda/imageProcessing/convolution.h"

namespace Saiga {
namespace CUDA {

void setGaussianBlurKernel(float sigma, int RADIUS){
    SAIGA_ASSERT(RADIUS <= MAX_RADIUS && RADIUS > 0);
    const int ELEMENTS = RADIUS * 2 + 1;
    float kernel[ELEMENTS];
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*sigma*sigma);
    for (int j=-RADIUS;j<=RADIUS;j++) {
        kernel[j+RADIUS] = (float)expf(-(double)j*j*ivar2);
        kernelSum += kernel[j+RADIUS];
    }
    for (int j=-RADIUS;j<=RADIUS;j++)
        kernel[j+RADIUS] /= kernelSum;

    CUDA::copyConvolutionKernel(Saiga::array_view<float>(kernel,RADIUS*2+1));
}

void gaussianBlur(ImageView<float> src, ImageView<float> dst, float sigma, int radius){
    setGaussianBlurKernel(sigma,radius);
    gaussianBlur(src,dst,radius);
}


void gaussianBlur(ImageView<float> src, ImageView<float> dst, int radius){
    convolve(src,dst,radius);
}

}
}


