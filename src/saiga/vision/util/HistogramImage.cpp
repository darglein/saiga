/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "HistogramImage.h"

#include "saiga/core/image/ImageDraw.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/color.h"

namespace Saiga
{
HistogramImage::HistogramImage(int inputW, int inputH, int outputW, int outputH)
    : inputW(inputW), inputH(inputH), outputW(std::min(inputW, outputW)), outputH(std::min(inputH, outputH))
{
    img.create(this->outputH, this->outputW);
    img.getImageView().set(0);
}

HistogramImage::BinIndex HistogramImage::bin(int y, int x)
{
    // pixel center
    double dx = x + 0.5;
    double dy = y + 0.5;
    // rescale
    dx = dx * ((double)outputW / (double)inputW);
    dy = dy * ((double)outputH / (double)inputH);
    // round to target
    int ox = iRound(dx - 0.5);
    int oy = iRound(dy - 0.5);

    if (!img.getImageView().inImage(oy, ox))
    {
        ox = -1;
        oy = -1;
    }
    return {oy, ox};
}

HistogramImage::BinIndex HistogramImage::add(int y, int x, int value)
{
    auto bid = bin(y, x);
    if (bid.first >= 0)
    {
        img(bid.first, bid.second) += value;
    }
    return bid;
}

void HistogramImage::writeBinary(const std::string& file, int threshold)
{
    TemplatedImage<ucvec3> outimg(outputH, outputW);

    for (int i = 0; i < outputH; ++i)
    {
        for (int j = 0; j < outputW; ++j)
        {
            if (img(i, j) >= threshold)
            {
                outimg(i, j) = ucvec3(0, 0, 255);
            }
            else
            {
                outimg(i, j) = ucvec3(255, 255, 255);
            }
        }
    }
    std::cout << outimg << std::endl;
    outimg.save(file);
}

void HistogramImage::drawGridLines(ImageView<ucvec4> imgv, bool drawHistBoxes)
{
    double dx = double(inputW) / outputW;
    double dy = double(inputH) / outputH;

    if (drawHistBoxes)
    {
        for (int i = 0; i < outputH; ++i)
        {
            for (int j = 0; j < outputW; ++j)
            {
                ivec2 low(j * dx, i * dy);
                ivec2 high((j + 1) * dx, (i + 1) * dy);

                if (img(i, j) >= 1)
                {
                    ImageDraw::FillRectangle(imgv, low, high, ucvec4(0, 0, 255, 255));
                }
            }
        }
    }


    for (auto i = 0; i < outputH + 1; ++i)
    {
        auto y = i * dy;
        ImageDraw::drawLineBresenham(imgv, vec2(0 - 0.5, y - 0.5), vec2(inputW + 0.5, y - 0.5), ucvec4(255, 0, 0, 255));
    }


    for (auto i = 0; i < outputW + 1; ++i)
    {
        auto x = i * dx;
        ImageDraw::drawLineBresenham(imgv, vec2(x - 0.5, 0 - 0.5), vec2(x - 0.5, inputH + 0.5), ucvec4(255, 0, 0, 255));
    }
}

float HistogramImage::density(int threshold)
{
    int count = 0;
    for (auto i : img.rowRange())
        for (auto j : img.colRange())
            if (img(i, j) >= threshold) count++;
    return float(count) / (img.w * img.h);
}

}  // namespace Saiga
