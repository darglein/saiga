#include "saiga/text/textShader.h"

void TextShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_model = getUniformLocation("model");
    location_proj = getUniformLocation("proj");

    location_texture = getUniformLocation("text");

    location_color = getUniformLocation("color");
    location_outlineColor = getUniformLocation("outlineColor");
    location_glowColor = getUniformLocation("glowColor");

    location_outlineData = getUniformLocation("outlineData");
    location_softEdgeData = getUniformLocation("softEdgeData");
    location_glowData = getUniformLocation("glowData");

    location_alphaMultiplier = getUniformLocation("alphaMultiplier");

}



void TextShader::uploadTextureAtlas(Texture* texture){
    Shader::upload(location_texture,texture,0);
}

void TextShader::uploadColor(const vec4 &color, const vec2 &softEdgeData)
{
     Shader::upload(location_color,color);
     Shader::upload(location_softEdgeData,softEdgeData);
}

void TextShader::uploadOutline(const vec4 &outlineColor, const vec4 &outlineData)
{
    Shader::upload(location_outlineColor,outlineColor);
    Shader::upload(location_outlineData,outlineData);
}

void TextShader::uploadGlow(const vec4 &glowColor, const vec2 &glowData)
{
    Shader::upload(location_glowColor,glowColor);
    Shader::upload(location_glowData,glowData);
}

void TextShader::uploadAlpha(float alpha)
{
    Shader::upload(location_alphaMultiplier,alpha);
}


void TextShaderFade::checkUniforms()
{
    TextShader::checkUniforms();
     location_fade = getUniformLocation("fade");
}

void TextShaderFade::uploadFade(float fade)
{
    Shader::upload(location_fade,fade);
}
