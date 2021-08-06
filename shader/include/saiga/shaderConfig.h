/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#ifndef SHADER_CONFIG_H
#define SHADER_CONFIG_H


#if defined(GL_core_profile)
#    define SHADER_DEVICE
#else
#    define SHADER_HOST
#endif


#ifdef SHADER_HOST
#    include "saiga/config.h"
#    include "saiga/core/math/math.h"
using Saiga::abs;
using Saiga::clamp;
using Saiga::fract;
using Saiga::make_vec3;
using Saiga::mat3;
using Saiga::mat4;
using Saiga::mix;
using Saiga::vec2;
using Saiga::vec3;
using Saiga::vec4;
using std::max;
using std::min;
using namespace Saiga;
#    define FUNC_DECL HD inline
#else
#    define FUNC_DECL

FUNC_DECL vec3 make_vec3(float x)
{
    return vec3(x);
}

// Saturate is not defined by opengl
FUNC_DECL float saturate(float x)
{
    return clamp(x, 0.0f, 1.0f);
}

FUNC_DECL vec2 saturate(vec2 x)
{
    return clamp(x, vec2(0, 0), vec2(1, 1));
}

FUNC_DECL vec3 saturate(vec3 x)
{
    return clamp(x, vec3(0, 0, 0), vec3(1, 1, 1));
}

FUNC_DECL vec4 saturate(vec4 x)
{
    return clamp(x, vec4(0, 0, 0, 0), vec4(1, 1, 1, 1));
}
#endif



#endif
