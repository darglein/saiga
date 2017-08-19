#pragma once

#include "saiga/cuda/imageProcessing/image.h"

namespace Saiga {
namespace CUDA {

SAIGA_GLOBAL void convertRGBtoRGBA(ImageView<uchar3> src, ImageView<uchar4> dst, unsigned char alpha = 255);

SAIGA_GLOBAL void convertRGBAtoGrayscale(ImageView<uchar4> src, ImageView<float> dst);

}
}
