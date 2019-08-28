/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "HistogramImage.h"

#include "saiga/core/math/imath.h"

namespace Saiga
{
HistogramImage::HistogramImage(int inputW, int inputH, int outputW, int outputH)
    : inputW(inputW), inputH(inputH), outputW(min(inputW, outputW)), outputH(min(inputH, outputH))
{
    img.create(this->outputH, this->outputW);
    img.getImageView().set(0);
}

void HistogramImage::add(int y, int x, int value)
{
    // pixel center
    double dx = x + 0.5;
    double dy = y + 0.5;
    // rescale
    dx = dx * ((double)outputW / (double)inputW);
    dy = dy * ((double)outputH / (double)inputH);
    // round to target
    int ox = iRound(dx);
    int oy = iRound(dy);

    if (img.getImageView().inImage(oy, ox))
    {
        img.getImageView()(oy, ox) += value;
    }
}

void HistogramImage::writeBinary(const std::string& file)
{
    TemplatedImage<ucvec3> outimg(outputH, outputW);

    for (int i = 0; i < outputH; ++i)
    {
        for (int j = 0; j < outputW; ++j)
        {
            if (img(i, j) > 0)
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

float HistogramImage::density(int threshold)
{
    int count = 0;
    for (auto i : img.rowRange())
        for (auto j : img.colRange())
            if (img(i, j) >= threshold) count++;
    return float(count) / (img.w * img.h);
}

}  // namespace Saiga
