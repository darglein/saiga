/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"



namespace Saiga
{
struct RansacParameters
{
    int iterations;
    double threshold;
};

// double inlierProb  = 0.7;
// double successProb = 0.999;

// double k = log(1 - successProb) / log(1 - pow(inlierProb, 5));
// std::cout << k << std::endl;



}  // namespace Saiga
