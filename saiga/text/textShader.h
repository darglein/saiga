#pragma once

#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/texture.h"

class SAIGA_GLOBAL TextShader : public MVPShader {
public:
    GLint location_texture;
    GLint location_color, location_outlineColor, location_glowColor;
    GLint location_outlineData, location_softEdgeData, location_glowData;
     GLint location_alphaMultiplier;
    virtual void checkUniforms();

    void uploadTextureAtlas(Texture* texture);
    void uploadColor(const vec4& color, const vec2& softEdgeData);
    void uploadOutline(const vec4& outlineColor, const vec4& outlineData);
    void uploadGlow(const vec4& glowColor, const vec2& glowData);
    void uploadAlpha(float alpha);
};

class SAIGA_GLOBAL TextShaderFade : public TextShader {
public:
    GLint location_fade;

    virtual void checkUniforms();

    void uploadFade(float fade);
};
