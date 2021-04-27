/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/text/textShader.h"

namespace Saiga
{
void TextShader::checkUniforms()
{
    MVPShader::checkUniforms();

    location_texture = getUniformLocation("text");

    location_color        = getUniformLocation("color");
    location_outlineColor = getUniformLocation("outlineColor");
    location_glowColor    = getUniformLocation("glowColor");

    location_outlineData  = getUniformLocation("outlineData");
    location_softEdgeData = getUniformLocation("softEdgeData");
    location_glowData     = getUniformLocation("glowData");

    location_alphaMultiplier = getUniformLocation("alphaMultiplier");
}

void TextShader::uploadTextParameteres(const TextParameters& params)
{
    Shader::upload(location_color, params.color);
    Shader::upload(location_softEdgeData, params.softEdgeData);

    Shader::upload(location_outlineColor, params.outlineColor);
    Shader::upload(location_outlineData, params.outlineData);

    Shader::upload(location_glowColor, params.glowColor);
    Shader::upload(location_glowData, params.glowData);

    Shader::upload(location_alphaMultiplier, params.alpha);
}



void TextShader::uploadTextureAtlas(std::shared_ptr<Texture> texture)
{
    Shader::upload(location_texture, texture, 0);
}



void TextShaderFade::checkUniforms()
{
    TextShader::checkUniforms();
    location_fade = getUniformLocation("fade");
}

void TextShaderFade::uploadFade(float fade)
{
    Shader::upload(location_fade, fade);
}

}  // namespace Saiga
