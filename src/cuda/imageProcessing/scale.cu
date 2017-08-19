/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/scale.h"

namespace Saiga {
namespace CUDA {

void scaleDown2EveryOther(ImageView<float> src, ImageView<float> dst){
    SAIGA_ASSERT(src.width == dst.width*2 && src.height == dst.height*2);
}

void scaleUp2Linear(ImageView<float> src, ImageView<float> dst){
    SAIGA_ASSERT(src.width*2 == dst.width && src.height*2 == dst.height);

}

}
}


