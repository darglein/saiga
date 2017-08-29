/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/imageProcessing/image.h"

namespace Saiga {
namespace CUDA {

SAIGA_GLOBAL void convertRGBtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha = 255);
SAIGA_GLOBAL void convertRGBAtoRGB(ImageView<uchar4> src, ImageView<uchar3> dst);

SAIGA_GLOBAL void convertRGBAtoGrayscale(ImageView<uchar4> src, ImageView<float> dst);


SAIGA_GLOBAL void convertBGRtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha = 255);
SAIGA_GLOBAL void convertRGBAtoBGR(ImageView<uchar4> src, ImageView<uchar3> dst);

}
}
