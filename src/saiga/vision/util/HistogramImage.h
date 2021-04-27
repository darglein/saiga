/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/image.h"
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
class SAIGA_VISION_API HistogramImage
{
   public:
    using BinIndex = std::pair<int, int>;
    HistogramImage(int inputW, int inputH, int outputW, int outputH);


    // returns the bin without adding
    BinIndex bin(int y, int x);
    BinIndex add(int y, int x, int value);

    void writeBinary(const std::string& file, int threshold = 1);

    int operator()(int y, int x) { return img(y, x); }

    void drawGridLines(ImageView<ucvec4> img, bool drawHistBoxes = true);

    // all elements larger or equal than threshold divided by size
    float density(int threshold = 1);

   private:
    int inputW, inputH, outputW, outputH;
    TemplatedImage<int> img;
};
}  // namespace Saiga
