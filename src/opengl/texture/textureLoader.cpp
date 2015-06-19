
#include "opengl/texture/textureLoader.h"
#include <FreeImagePlus.h>

Texture* TextureLoader::loadFromFile(const std::string &path){

    bool erg;
    Texture* text = new Texture();

//    PNG::Image img;
//    erg = PNG::readPNG( &img,path);
//    cout<<"loading "<<path<<endl;

    fipImage fipimg;
    erg = fipimg.load(path.c_str());


    if (erg){
        Image im;
        im.convertFrom(fipimg);
        im.srgb = true;
        erg = text->fromImage(im);
    }

    if(erg){
        return text;
    }else{
        delete text;
    }



    return nullptr;
}

