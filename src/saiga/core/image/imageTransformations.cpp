/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imageTransformations.h"

#include "saiga/colorize.h"
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

void RemoveAlphaChannel(ImageView<const ucvec4> src, ImageView<ucvec3> dst)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for (int i = 0; i < src.height; ++i)
    {
        for (int j = 0; j < src.width; ++j)
        {
            dst(i, j) = src(i, j).head<3>();
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
            d       = (d - minD) / (maxD - minD);

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
            d       = (d - minD) / (maxD - minD);
            d       = clamp(d, 0.0f, 1.0f);

            dst(i, j) = ucvec4(d * 255, d * 255, d * 255, 255);
        }
    }
}

void depthToRGBA_HSV(ImageView<const float> src, ImageView<ucvec4> dst, float minD, float maxD)
{
    SAIGA_ASSERT(src.width == dst.width && src.height == dst.height);
    for (int i = 0; i < src.height; ++i)
    {
        for (int j = 0; j < src.width; ++j)
        {
            float d = src(i, j);
            d       = (d - minD) / (maxD - minD);

            d = clamp(d, 0.0f, 1.0f);

            vec3 hsv(d * (240.0 / 360.0), 1, 1);
            Saiga::Color c(Color::hsv2rgb(hsv));
            //            unsigned char c = Saiga::iRound(f * 255.0f);
            dst(i, j)[0] = c.r;
            dst(i, j)[1] = c.g;
            dst(i, j)[2] = c.b;
            dst(i, j)[3] = 255;
        }
    }
}



// const vec3 rgbToGray(0.2126f, 0.7152f, 0.0722f);
const vec3 rgbToGray(0.299f, 0.587f, 0.114f);  // opencv values

struct RGBATOGRAY8Trans
{
    unsigned char operator()(const ucvec4& v)
    {
        vec3 vf(v[0], v[1], v[2]);
        float gray = dot(rgbToGray, vf);
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
        vec3 vf(v[0], v[1], v[2]);
        float gray = dot(rgbToGray, vf);
        return gray * scale;
    }
};
void RGBAToGrayF(ImageView<const ucvec4> src, ImageView<float> dst, float scale)
{
    src.copyToTransform(dst, RGBATOGRAYFTrans(scale));
}

struct Gray8ToRGBATrans
{
    unsigned char alpha;
    Gray8ToRGBATrans(unsigned char alpha) : alpha(alpha) {}
    ucvec4 operator()(const unsigned char& v) { return ucvec4(v, v, v, alpha); }
};

void Gray8ToRGBA(ImageView<unsigned char> src, ImageView<ucvec4> dst, unsigned char alpha)
{
    src.copyToTransform(dst, Gray8ToRGBATrans(alpha));
}
struct Gray8ToRGBTrans
{
    ucvec3 operator()(const unsigned char& v) { return ucvec3(v, v, v); }
};

void Gray8ToRGB(ImageView<unsigned char> src, ImageView<ucvec3> dst)
{
    src.copyToTransform(dst, Gray8ToRGBTrans());
}

float sharpness(ImageView<const unsigned char> src)
{
    long sum = 0;
    for (auto i : src.rowRange(1))
    {
        for (auto j : src.colRange(1))
        {
            auto dx = src(i, j + 1) - src(i, j - 1);
            auto dy = src(i + 1, j) - src(i - 1, j);
            sum += std::max(dx, dy);
        }
    }
    return float(sum) / (src.w * src.h);
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

void ScaleDown2(ImageView<const ucvec4> src, ImageView<ucvec4> dst)
{
    for (int i : dst.rowRange())
    {
        for (int j : dst.colRange())
        {
            int i_src = i * 2;
            int j_src = j * 2;
            ivec4 sum = src(i_src, j_src).cast<int>() + src(i_src + 1, j_src).cast<int>() +
                        src(i_src, j_src + 1).cast<int>() + src(i_src + 1, j_src + 1).cast<int>();
            sum /= 4;
            dst(i, j) = sum.cast<unsigned char>();
        }
    }
}
TemplatedImage<unsigned char> AbsolutePixelError(ImageView<const ucvec3> img1, ImageView<const ucvec3> img2)
{
    TemplatedImage<unsigned char> result(img1.dimensions());
    for (int i : img1.rowRange())
    {
        for (int j : img1.colRange())
        {
            int diff     = (img1(i, j).cast<int>() - img2(i, j).cast<int>()).array().abs().maxCoeff();
            result(i, j) = diff;
        }
    }
    return result;
}


TemplatedImage<unsigned char> AbsolutePixelError(ImageView<const unsigned char> img1,
                                                                      ImageView<const unsigned char> img2)
{
    TemplatedImage<unsigned char> result(img1.dimensions());
    for (int i : img1.rowRange())
    {
        for (int j : img1.colRange())
        {
            int diff     = std::abs(int(img1(i, j)) - int(img2(i,j)));
            result(i, j) = diff;
        }
    }
    return result;
}

TemplatedImage<ucvec3> ErrorImage(ImageView<ucvec3> img1, ImageView<ucvec3> img2)
{
    auto absolute_diff = ImageTransformation::AbsolutePixelError(img1, img2);

    TemplatedImage<ucvec3> error_img(img1.dimensions());
    for (int i : error_img.rowRange())
    {
        for (int j : error_img.colRange())
        {
            float error = absolute_diff(i, j) / 255.f;
            vec3 c      = saturate(colorizeTurbo(error));

            error_img(i, j) = (c * 255.f).cast<unsigned char>();
        }
    }
    return error_img;
}


TemplatedImage<ucvec3> ErrorImage(ImageView<unsigned char> img1, ImageView<unsigned char> img2)
{
    auto absolute_diff = AbsolutePixelError(img1, img2);

    TemplatedImage<ucvec3> error_img(img1.dimensions());
    for (int i : error_img.rowRange())
    {
        for (int j : error_img.colRange())
        {
            float error = absolute_diff(i, j) / 255.f;
            vec3 c      = saturate(colorizeTurbo(error));

            error_img(i, j) = (c * 255.f).cast<unsigned char>();
        }
    }
    return error_img;
}


long L1Difference(ImageView<const ucvec3> img1, ImageView<const ucvec3> img2)
{
    long result = 0;
    for (int i : img1.rowRange())
    {
        for (int j : img1.colRange())
        {
            long diff = (img1(i, j).cast<int>() - img2(i, j).cast<int>()).array().abs().sum();
            result += diff;
        }
    }
    return result;
}
long L1Difference(ImageView<const unsigned char> img1, ImageView<const unsigned char> img2)
{
    long result = 0;
    for (int i : img1.rowRange())
    {
        for (int j : img1.colRange())
        {
            long diff = std::abs(img1(i, j)- img2(i, j));
            result += diff;
        }
    }
    return result;
}

}  // namespace ImageTransformation
}  // namespace Saiga
