/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "triangle.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
vec3 Triangle::center() const
{
    return (a + b + c) * float(1.0f / 3.0f);
}

float Triangle::minimalAngle()
{
    return acos(cosMinimalAngle());
}

float Triangle::cosMinimalAngle()
{
    return std::max(std::max(cosAngleAtCorner(0), cosAngleAtCorner(1)), cosAngleAtCorner(2));
}

float Triangle::angleAtCorner(int i)
{
    return acos(cosAngleAtCorner(i));
}

float Triangle::cosAngleAtCorner(int i)
{
    vec3 center = a;
    vec3 left   = b;
    vec3 right  = c;


    switch (i)
    {
        case 0:
            center = a;
            left   = b;
            right  = c;
            break;
        case 1:
            center = b;
            left   = c;
            right  = a;
            break;
        case 2:
            center = c;
            left   = a;
            right  = b;
            break;
    }

    return dot(normalize(vec3(left - center)), normalize(vec3(right - center)));
}

bool Triangle::isDegenerate()
{
    for (int i = 0; i < 3; ++i)
    {
        float a = cosAngleAtCorner(i);
        if (a <= -1 || a >= 1) return true;
    }
    return false;
    //    return !std::isfinite(angleAtCorner(0)) || !std::isfinite(angleAtCorner(1)) ||
    //    !std::isfinite(angleAtCorner(2));
}

vec3 Triangle::normal()
{
    return normalize(cross(b - a, c - a));
}

std::ostream& operator<<(std::ostream& os, const Triangle& t)
{
    os << "Triangle: " << t.a << t.b << t.c;
    return os;
}

}  // namespace Saiga
