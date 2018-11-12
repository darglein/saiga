/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/math.h"


namespace Saiga {
namespace PixelTransformation {


constexpr vec3 grayTransform() { return {0.2126f,0.7152f,0.0722f}; }


inline float toGray(const ucvec3& v, float scale = 1)
{
    vec3 vf(v.x,v.y,v.z);
    float gray = dot(grayTransform(),vf);
    return gray * scale;
}

inline float toGray(const ucvec4& v, float scale = 1)
{
    return toGray(ucvec3(v),scale);
}

}
}
