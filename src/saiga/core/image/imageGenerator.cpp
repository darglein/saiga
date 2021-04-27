/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imageGenerator.h"

#include "saiga/core/math/random.h"

#include "internal/noGraphicsAPI.h"

#include "templatedImage.h"

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

    uint8_t r1 = color1[0] * 255.0f, g1 = color1[1] * 255.0f, b1 = color1[2] * 255.0f;


    uint8_t r2 = color2[0] * 255.0f, g2 = color2[1] * 255.0f, b2 = color2[2] * 255.0f;

    bool black = true;
    for (int qx = 0; qx < numQuadsX; ++qx)
    {
        for (int qy = 0; qy < numQuadsY; ++qy)
        {
            for (int i = 0; i < quadSize; ++i)
            {
                for (int j = 0; j < quadSize; ++j)
                {
                    if (black)
                    {
                        image->at<ucvec3>(qx * quadSize + i, qy * quadSize + j) = ucvec3(r1, g1, b1);
                    }
                    else
                    {
                        image->at<ucvec3>(qx * quadSize + i, qy * quadSize + j) = ucvec3(r2, g2, b2);
                    }
                }
            }

            black = !black;
        }
        black = !black;
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
