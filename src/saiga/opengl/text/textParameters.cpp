/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/text/textParameters.h"

namespace Saiga
{
void TextParameters::setOutline(const vec4& outlineColor, float width, float smoothness)
{
    this->outlineColor = outlineColor;
    width              = width * 0.5f;
    outlineData        = vec4(0.5f - width - smoothness, 0.5f - width + smoothness, 0.5f + width - smoothness,
                       0.5f + width + smoothness);
}

void TextParameters::setGlow(const vec4& glowColor, float width)
{
    this->glowColor = glowColor;
    width           = clamp(width, 0.0f, 1.0f) * 0.5f;
    glowData        = vec2(0.5f - width, 0.6f);
}

void TextParameters::setColor(const vec4& color, float smoothness)
{
    this->color  = color;
    softEdgeData = vec2(0.5f - smoothness, 0.5f + smoothness);
}

void TextParameters::setAlpha(float alpha)
{
    this->alpha = alpha;
}

}  // namespace Saiga
