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

            dst(i,j) = ucvec4(ucvec3(d * 255),255);
        }
    }
}

void depthToRGBA(ImageView<const float> src, ImageView<ucvec4> dst, float minD, float maxD)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for(int i = 0; i < src.height; ++i)
    {
        for(int j =0; j < src.width; ++j)
        {
            float d = src(i,j);
            d = (d - minD) / maxD;

            dst(i,j) = ucvec4(ucvec3(d * 255),255);
        }
    }
}

struct RGBATOGRAY8Trans
{
    unsigned char operator()(const ucvec4& v)
    {
        const vec3 conv(0.2126f,0.7152f,0.0722f);
        vec3 vf(v.x,v.y,v.z);
        float gray = dot(conv,vf);
        return gray;
    }
};

void RGBAToGray8(ImageView<const ucvec4> src, ImageView<unsigned char> dst)
{
    src.copyToTransform(dst,RGBATOGRAY8Trans());
}

}
}
