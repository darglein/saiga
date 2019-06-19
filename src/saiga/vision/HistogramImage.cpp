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
    int ox = iRound(x * ((double)outputW / (double)inputW));
    int oy = iRound(y * ((double)outputH / (double)inputH));

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

}  // namespace Saiga
