/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/imageProcessing/imageView.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>


namespace Saiga {
namespace CUDA {


#define MAX_RADIUS 10

SAIGA_GLOBAL void copyConvolutionKernel(Saiga::array_view<float> kernel);
SAIGA_GLOBAL void convolve(ImageView<float> src, ImageView<float> dst, int radius);

}
}
