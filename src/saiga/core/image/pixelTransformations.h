/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"


namespace Saiga
{
namespace PixelTransformation
{
inline vec3 grayTransform()
{
    return {0.2126f, 0.7152f, 0.0722f};
}


inline float toGray(const ucvec3& v, float scale = 1)
{
    vec3 vf(v[0], v[1], v[2]);
    float gray = dot(grayTransform(), vf);
    return gray * scale;
}

inline float toGray(const ucvec4& v, float scale = 1)
{
    return toGray(v.segment<3>(0).eval(), scale);
}

}  // namespace PixelTransformation
}  // namespace Saiga
