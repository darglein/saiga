/**
 * Copyright (c) 2017 Darius Rückert
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
#    define FUNC_DECL inline

#    ifdef SAIGA_FULL_EIGEN
#        include "saiga/core/math/math.h"
using Saiga::mat3;
using Saiga::mat4;
using Saiga::vec2;
using Saiga::vec3;
using Saiga::vec4;
using namespace Saiga;
#    else
#        define GLM_FORCE_SWIZZLE
#        include "saiga/core/math/math.h"
using namespace glm;
#    endif

#else
#    define FUNC_DECL
#endif

/**
 * This is actually a usefull function, so declare it here instead of in
 * hlslDefines.h
 */
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
