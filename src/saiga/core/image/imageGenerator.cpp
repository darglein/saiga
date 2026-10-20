/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imageGenerator.h"

#include "saiga/core/math/random.h"

#include "internal/noGraphicsAPI.h"

#include "templatedImage.h"

#include <algorithm>

namespace Saiga
{
std::shared_ptr<Image> ImageGenerator::checkerBoard(vec3 color1, vec3 color2, int quadSize, int numQuadsX,
                                                    int numQuadsY)
{
    std::shared_ptr<Image> image = std::make_shared<Image>();
    image->width                 = quadSize * numQuadsX;
    image->height                = quadSize * numQuadsY;
    image->type                  = UC3;
    image->create();

    // Converting a float outside of [0,255] to uint8_t is undefined, so the colors are clamped to [0,1] first
    // (the argument order maps NaN to 0).
    auto to_uc = [](float c) { return uint8_t(std::min(1.0f, std::max(0.0f, c)) * 255.0f); };
    uint8_t r1 = to_uc(color1[0]), g1 = to_uc(color1[1]), b1 = to_uc(color1[2]);
    uint8_t r2 = to_uc(color2[0]), g2 = to_uc(color2[1]), b2 = to_uc(color2[2]);

    for (int qx = 0; qx < numQuadsX; ++qx)
    {
        for (int qy = 0; qy < numQuadsY; ++qy)
        {
            // Neighboring quads have different colors (independent of the number of quads).
            bool first = (qx + qy) % 2 == 0;
            for (int i = 0; i < quadSize; ++i)
            {
                for (int j = 0; j < quadSize; ++j)
                {
                    image->at<ucvec3>(qy * quadSize + i, qx * quadSize + j) =
                        first ? ucvec3(r1, g1, b1) : ucvec3(r2, g2, b2);
                }
            }
        }
    }

    //    image->save("debug/checkerboard.png");



    return image;
}

std::shared_ptr<Image> ImageGenerator::randomNormalized(int width, int height)
{
    TemplatedImage<cvec4>* image = new TemplatedImage<cvec4>(height, width);
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            cvec4 texel;
            texel[0]       = linearRand(0, 255);
            texel[1]       = linearRand(0, 255);
            texel[2]       = linearRand(0, 255);
            texel[3]       = linearRand(0, 255);
            (*image)(i, j) = texel;
        }
    }
    return std::shared_ptr<Image>(image);
}

}  // namespace Saiga
