/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/image/imageTransformations.h"

namespace Saiga {

void addAlphaChannel(ImageView<ucvec3> src, ImageView<ucvec4> dst, unsigned char alpha)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for(int i = 0; i < src.height; ++i){
        for(int j =0; j < src.width; ++j)
        {
            dst(i,j) = ucvec4(src(i,j),alpha);
        }
    }
}

}
