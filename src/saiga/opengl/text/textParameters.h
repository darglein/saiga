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
struct SAIGA_OPENGL_API TextParameters
{
    vec4 color        = vec4(1, 1, 1, 1);
    vec4 outlineColor = vec4(0, 0, 0, 0);
    vec4 glowColor    = vec4(0, 0, 0, 0);

    vec4 outlineData  = vec4(0.5f, 0.5f, 0.5f, 0.5f);
    vec2 softEdgeData = vec2(0.5f, 0.5f);
    vec2 glowData     = vec2(0.5f, 0.5f);
    float alpha       = 1.0f;

    void setOutline(const vec4& outlineColor, float width, float smoothness);
    void setGlow(const vec4& glowColor, float width);
    void setColor(const vec4& color, float smoothness);
    void setAlpha(float alpha);
};

}  // namespace Saiga
