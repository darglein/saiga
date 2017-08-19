/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/cuda/imageProcessing/image.h"

namespace Saiga {
namespace CUDA {

SAIGA_GLOBAL void fill(ImageView<float> img, float value);

SAIGA_GLOBAL void scaleDown2EveryOther(ImageView<float> src, ImageView<float> dst);

SAIGA_GLOBAL void scaleUp2Linear(ImageView<float> src, ImageView<float> dst);


}
}
