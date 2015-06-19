#pragma once

#include "libhello/opengl/texture/raw_texture.h"
#include "libhello/util/loader.h"
#include "libhello/util/singleton.h"

class basic_Texture_2D : public raw_Texture{
public:
    string name;
    GLubyte* data = nullptr;

    basic_Texture_2D():raw_Texture(GL_TEXTURE_2D){}
    basic_Texture_2D(const string &name):raw_Texture(GL_TEXTURE_2D),name(name){}
    virtual ~basic_Texture_2D(){}

    void setDefaultParameters() override;
     bool fromImage(Image &img);

};

typedef basic_Texture_2D Texture;

class TextureLoader : public Loader<Texture>, public Singleton <TextureLoader>{
    friend class Singleton <TextureLoader>;
public:
    Texture* loadFromFile(const std::string &name);
};

