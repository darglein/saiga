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

//source https://www.geeksforgeeks.org/count-trailing-zero-bits-using-lookup-table/
HD inline
int countTrailingZero(int x)
{
     // Map a bit value mod 37 to its position
     static const int lookup[] = {32, 0, 1,
     26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11,
     0, 13, 4, 7, 17, 0, 25, 22, 31, 15, 29,
     10, 12, 6, 0, 21, 14, 9, 5, 20, 8, 19,
     18};

     // Only difference between (x and -x) is
     // the value of signed magnitude(leftmostbit)
     // negative numbers signed bit is 1
     return lookup[(-x & x) % 37];
}

}

