#pragma once

#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/rendering/material.h"

class SAIGA_GLOBAL MaterialShader : public MVPShader{
public:
    GLint location_colors;
    GLint location_textures, location_use_textures;
    vec3 colors[3]; //ambiend, diffuse, specular
    GLint textures[5]; //ambiend, diffuse, specular, alpha, bump
    float use_textures[5]; //1.0 if related texture is valid

    virtual void checkUniforms();
    void uploadMaterial(const Material &material);

};


