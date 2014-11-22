
#include "opengl/texture/texture.h"


Texture* TextureLoader::loadFromFile(const std::string &path){

    bool erg;
    Texture* text = new Texture();

    PNG::Image img;
    erg = PNG::readPNG( &img,path);
    if (erg){
        Image im;
        im.convertFrom(img);
        erg = text->fromImage(im);
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

bool basic_Texture_2D::fromImage(Image &img){


    setFormat(img);

    createGlTexture();
    uploadData(img.data);
    return true;
}
