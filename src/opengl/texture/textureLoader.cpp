
#include "saiga/opengl/texture/textureLoader.h"
#include <FreeImagePlus.h>

bool operator==(const TextureParameters &lhs, const TextureParameters &rhs) {
    return std::tie(lhs.srgb) == std::tie(lhs.srgb);
}

Texture* TextureLoader::loadFromFile(const std::string &path, const TextureParameters &params){

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
        im.srgb = params.srgb;
        erg = text->fromImage(im);
    }

    if(erg){
        return text;
    }else{
        delete text;
    }



    return nullptr;
}




