/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/cuda/common.h"

namespace Saiga {

// Unsigned int to normalized signed float in the range [-1,1]
HD inline
float uintToNSFloat(unsigned int x)
{
    x = x >> (32 - 23);
    x = (x & 0x007fffff) | 0x40000000;
    return *reinterpret_cast<float*>(&x) - 3.0f;
}


// Unsigned int to normalized float in the range [0,1]
HD inline
float uintToNFloat(unsigned int x)
{
    x = x >> (32 - 23);
    x = (x & 0x007fffff) | 0x3f800000;
    return *reinterpret_cast<float*>(&x) - 1.0f;
}

}

