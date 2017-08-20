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
#define MAX_KERNEL_SIZE (MAX_RADIUS*2+1)


SAIGA_GLOBAL void convolveSinglePassSeparate(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);
SAIGA_GLOBAL void convolveRow(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);
SAIGA_GLOBAL void convolveCol(ImageView<float> src, ImageView<float> dst, Saiga::array_view<float> kernel, int radius);

}
}
