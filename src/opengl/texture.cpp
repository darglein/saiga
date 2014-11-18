
#include "opengl/texture.h"


Texture* TextureLoader::loadFromFile(const std::string &path){

    bool erg;
    Texture* text = new Texture();

    PNG::Image img;
    erg = PNG::readPNG( &img,path);
    if (erg){
        erg = text->fromPNG(&img);
    }

    if(erg){
        return text;
    }else{
        delete text;
    }



    return nullptr;
}

void basic_Texture_2D::setDefaultParameters(){
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

