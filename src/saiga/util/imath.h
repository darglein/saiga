/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/cuda/common.h"

namespace Saiga {

//returns the smallest x number with: x * b >= a
HD SAIGA_CONSTEXPR inline
int iDivUp(int a, int b) { return (a + b - 1) / b; }

HD SAIGA_CONSTEXPR inline
int iDivDown(int a, int b) { return a / b; }

//finds the smallest number that is bigger or equal than a and divisible by b
HD SAIGA_CONSTEXPR inline
int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }

//finds the largest number that is smaller or equal than a and divisible by b
HD SAIGA_CONSTEXPR inline
int iAlignDown(int a, int b) {return a - a % b; }

HD inline
int iFloor(float value){
    int i = (int)value;
    return i - (i > value);
}

HD inline
int iCeil(float value){
    int i = (int)value;
    return i + (i < value);
}

HD inline
int iRound(float value){
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}


}

