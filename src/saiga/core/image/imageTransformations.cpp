/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imageTransformations.h"

#include "saiga/core/util/color.h"

#include "internal/noGraphicsAPI.h"

#include "templatedImage.h"

namespace Saiga
{
namespace ImageTransformation
{
void addAlphaChannel(ImageView<const ucvec3> src, ImageView<ucvec4> dst, unsigned char alpha)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for (int i = 0; i < src.height; ++i)
    {
        for (int j = 0; j < src.width; ++j)
        {
            dst(i, j) = make_ucvec4(src(i, j), alpha);
        }
    }
}

void depthToRGBA(ImageView<const uint16_t> src, ImageView<ucvec4> dst, uint16_t minD, uint16_t maxD)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for (int i = 0; i < src.height; ++i)
    {
        for (int j = 0; j < src.width; ++j)
        {
            float d = src(i, j);
            d       = (d - minD) / maxD;

            dst(i, j) = ucvec4(d * 255, d * 255, d * 255, 255);
        }
    }
}

void depthToRGBA(ImageView<const float> src, ImageView<ucvec4> dst, float minD, float maxD)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for (int i = 0; i < src.height; ++i)
    {
        for (int j = 0; j < src.width; ++j)
        {
            float d = src(i, j);
            d       = (d - minD) / maxD;

            dst(i, j) = ucvec4(d * 255, d * 255, d * 255, 255);
        }
    }
}

struct RGBATOGRAY8Trans
{
    unsigned char operator()(const ucvec4& v)
    {
        const vec3 conv(0.2126f, 0.7152f, 0.0722f);
        vec3 vf(v[0], v[1], v[2]);
        float gray = dot(conv, vf);
        return gray;
    }
};

void RGBAToGray8(ImageView<const ucvec4> src, ImageView<unsigned char> dst)
{
    src.copyToTransform(dst, RGBATOGRAY8Trans());
}



struct RGBATOGRAYFTrans
{
    float scale;
    RGBATOGRAYFTrans(float scale) : scale(scale) {}
    float operator()(const ucvec4& v)
    {
        const vec3 conv(0.2126f, 0.7152f, 0.0722f);
        vec3 vf(v[0], v[1], v[2]);
        float gray = dot(conv, vf);
        return gray * scale;
    }
};
void RGBAToGrayF(ImageView<const ucvec4> src, ImageView<float> dst, float scale)
{
    src.copyToTransform(dst, RGBATOGRAYFTrans(scale));
}


bool saveHSV(const std::string& path, ImageView<float> img, float vmin, float vmax)
{
    TemplatedImage<float> cpy(img);
    auto vcpy = cpy.getImageView();
    vcpy.add(-vmin);
    vcpy.multWithScalar(float(1) / (vmax - vmin));

    TemplatedImage<ucvec3> simg(img.height, img.width);
    for (int i = 0; i < img.height; ++i)
    {
        for (int j = 0; j < img.width; ++j)
        {
            float f = clamp(vcpy(i, j), 0.0f, 1.0f);

            //            vec3 hsv = vec3(f,1,1);
            vec3 hsv(f * (240.0 / 360.0), 1, 1);
            Saiga::Color c(Color::hsv2rgb(hsv));
            //            unsigned char c = Saiga::iRound(f * 255.0f);
            simg(i, j)[0] = c.r;
            simg(i, j)[1] = c.g;
            simg(i, j)[2] = c.b;
        }
    }
    return simg.save(path);
}


bool save(const std::string& path, ImageView<float> img, float vmin, float vmax)
{
    TemplatedImage<float> cpy(img);
    auto vcpy = cpy.getImageView();

    vcpy.add(-vmin);
    vcpy.multWithScalar(float(1) / (vmax - vmin));

    TemplatedImage<unsigned char> simg(img.height, img.width);
    for (int i = 0; i < img.height; ++i)
    {
        for (int j = 0; j < img.width; ++j)
        {
            float f         = clamp(vcpy(i, j), 0.0f, 1.0f);
            unsigned char c = Saiga::iRound(f * 255.0f);
            simg(i, j)      = c;
        }
    }
    return simg.save(path);
}



}  // namespace ImageTransformation
}  // namespace Saiga
