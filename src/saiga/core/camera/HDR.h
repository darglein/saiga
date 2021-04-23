/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/math.h"

namespace Saiga
{
// Simple Polynomial Model Based on
// Vignette and Exposure Calibration and Compensation
// https://grail.cs.washington.edu/projects/vignette/vign.iccv05.pdf
//
// r_sqared is the (squared) distance to the optical center.
// if coefficients == 0 -> No vignetting exists
using VignetteCoefficients = vec3;
inline float VignetteModel(float r_squared, VignetteCoefficients coefficients)
{
    float r2 = r_squared;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    return 1.0f + coefficients[0] * r2 + coefficients[1] * r4 + coefficients[2] * r6;
}


}  // namespace Saiga
