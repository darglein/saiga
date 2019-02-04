/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/image.h"
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
class HistogramImage
{
   public:
    HistogramImage(int inputW, int inputH, int outputW, int outputH);

    void add(int y, int x, int value);

    void writeBinary(const std::string& file);

   private:
    int inputW, inputH, outputW, outputH;
    TemplatedImage<int> img;
};
}  // namespace Saiga
