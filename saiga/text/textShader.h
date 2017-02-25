#pragma once

#include "saiga/text/textParameters.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/texture.h"

class SAIGA_GLOBAL TextShader : public MVPShader {
public:
    GLint location_texture;
    GLint location_color, location_outlineColor, location_glowColor;
    GLint location_outlineData, location_softEdgeData, location_glowData;
     GLint location_alphaMultiplier;
    virtual void checkUniforms();

     void uploadTextParameteres(const TextParameters& params);
    void uploadTextureAtlas(std::shared_ptr<Texture> texture);
};

class SAIGA_GLOBAL TextShaderFade : public TextShader {
public:
    GLint location_fade;

    virtual void checkUniforms();

    void uploadFade(float fade);
};
