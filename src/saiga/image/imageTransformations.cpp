/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/image/imageTransformations.h"
#include "internal/noGraphicsAPI.h"

namespace Saiga {
namespace ImageTransformation {

void addAlphaChannel(ImageView<const ucvec3> src, ImageView<ucvec4> dst, unsigned char alpha)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for(int i = 0; i < src.height; ++i)
    {
        for(int j =0; j < src.width; ++j)
        {
            dst(i,j) = ucvec4(src(i,j),alpha);
        }
    }
}

void depthToRGBA(ImageView<const uint16_t> src, ImageView<ucvec4> dst, uint16_t minD, uint16_t maxD)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for(int i = 0; i < src.height; ++i)
    {
        for(int j =0; j < src.width; ++j)
        {
            float d = src(i,j);
            d = (d - minD) / maxD;

            dst(i,j) = ucvec4(d * 255);
        }
    }
}

}
}
