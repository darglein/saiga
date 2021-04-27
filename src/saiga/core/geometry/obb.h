/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

namespace Saiga
{
/**
 * Oriented Bounding Box
 */
class SAIGA_CORE_API OBB
{
   public:
    // center point
    vec3 center;
    // the column vectors represent the main axis
    // and their length is the positive half extend
    mat3 orientationScale;

    void setOrientationScale(vec3 x, vec3 y, vec3 z);


    void fitToPoints(int axis, vec3* points, int count);

    void normalize2();
};

}  // namespace Saiga
