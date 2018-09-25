/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"
#include "saiga/image/imageView.h"


namespace Saiga {
namespace ImageTransformation {

SAIGA_GLOBAL void addAlphaChannel(ImageView<const ucvec3> src, ImageView<ucvec4> dst, unsigned char alpha = 255);

// depth to rgb image only for visualizaion
SAIGA_GLOBAL void depthToRGBA(ImageView<const uint16_t> src, ImageView<ucvec4> dst, uint16_t minD, uint16_t maxD);
SAIGA_GLOBAL void depthToRGBA(ImageView<const float> src, ImageView<ucvec4> dst, float minD = 0, float maxD = 7);


}
}
