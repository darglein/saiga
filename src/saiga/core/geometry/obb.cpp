/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "obb.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
void OBB::setOrientationScale(vec3 x, vec3 y, vec3 z)
{
    (orientationScale.col(0)) = x;
    (orientationScale.col(1)) = y;
    (orientationScale.col(2)) = z;
}

void OBB::fitToPoints(int axis, vec3* points, int count)
{
    float xMin = 1e10, xMax = -1e10;

    vec3 dir = orientationScale.col(axis);
    dir      = normalize(dir);

    for (int i = 0; i < count; ++i)
    {
        float x = dot(dir, points[i]);
        xMin    = std::min(xMin, x);
        xMax    = std::max(xMax, x);
    }

    orientationScale.col(axis) = 0.5f * dir * (xMax - xMin);

    // translate center along axis
    float centerAxis = 0.5f * (xMax + xMin);
    float d          = dot(dir, center);
    center += (centerAxis - d) * dir;
}

void OBB::normalize2()
{
    orientationScale.col(0).normalize();
    orientationScale.col(1).normalize();
    orientationScale.col(2).normalize();
    //    col(orientationScale, 0) = normalize(col(orientationScale, 0));
    //    col(orientationScale, 1) = normalize(col(orientationScale, 1));
    //    col(orientationScale, 2) = normalize(col(orientationScale, 2));
}

}  // namespace Saiga
