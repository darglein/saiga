/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/imageProcessing/image.h"

namespace Saiga {
namespace CUDA {

SAIGA_GLOBAL void setGaussianBlurKernel(float sigma, int radius);

//uploads kernel and convoles images
SAIGA_GLOBAL void gaussianBlur(ImageView<float> src, ImageView<float> dst, float sigma, int radius);

//only convolves images with previously uploaded kernels
SAIGA_GLOBAL void gaussianBlur(ImageView<float> src, ImageView<float> dst, int radius);

}
}
