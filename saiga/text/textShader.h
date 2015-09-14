#pragma once

#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/texture.h"

class SAIGA_GLOBAL TextShader : public MVPShader {
public:
    GLint location_color, location_texture,location_strokeColor;

    virtual void checkUniforms();

    void upload(Texture* texture, const vec4 &color,const vec4 &strokeColor);
};

class SAIGA_GLOBAL TextShaderFade : public TextShader {
public:
    GLint location_fade;

    virtual void checkUniforms();

    void uploadFade(float fade);
};
