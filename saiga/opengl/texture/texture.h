#pragma once

#include "saiga/opengl/texture/raw_texture.h"

#include <string>

class SAIGA_GLOBAL basic_Texture_2D : public raw_Texture{
public:
    std::string name;
    GLubyte* data = nullptr;

    basic_Texture_2D():raw_Texture(GL_TEXTURE_2D){}
    basic_Texture_2D(const std::string &name):raw_Texture(GL_TEXTURE_2D),name(name){}
    virtual ~basic_Texture_2D(){}

    void setDefaultParameters() override;
     bool fromImage(Image &img);

};

typedef basic_Texture_2D Texture;


