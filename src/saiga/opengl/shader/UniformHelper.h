/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/core/util/FileSystem.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/shader/shaderpart.h"
#include "saiga/opengl/texture/Texture2D.h"

#include <memory>
#include <vector>

namespace Saiga
{
template <typename T>
struct UniformHelper
{
};


template <>
struct UniformHelper<float>
{
    static void Set(int l, int c, const float* v) { glUniform1fv(l, c, v); }
};


template <>
struct UniformHelper<int>
{
    static void Set(int l, int c, const int* v) { glUniform1iv(l, c, v); }
};


template <>
struct UniformHelper<vec2>
{
    static void Set(int l, int c, const vec2* v) { glUniform2fv(l, c, v->data()); }
};

template <>
struct UniformHelper<vec3>
{
    static void Set(int l, int c, const vec3* v) { glUniform3fv(l, c, v->data()); }
};

template <>
struct UniformHelper<vec4>
{
    static void Set(int l, int c, const vec4* v) { glUniform4fv(l, c, v->data()); }
};


template <>
struct UniformHelper<mat2>
{
    static void Set(int l, int c, const mat2* v) { glUniformMatrix2fv(l, c, GL_FALSE, v->data()); }
};

template <>
struct UniformHelper<mat3>
{
    static void Set(int l, int c, const mat3* v) { glUniformMatrix3fv(l, c, GL_FALSE, v->data()); }
};

template <>
struct UniformHelper<mat4>
{
    static void Set(int l, int c, const mat4* v) { glUniformMatrix4fv(l, c, GL_FALSE, v->data()); }
};

template <>
struct UniformHelper<TextureBase>
{
    static void Set(int l, int texture_unit, TextureBase* v)
    {
        v->bind(texture_unit);
        UniformHelper<int>::Set(l, 1, &texture_unit);
    }
};


}  // namespace Saiga
