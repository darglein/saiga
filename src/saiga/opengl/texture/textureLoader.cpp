/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/opengl/texture/textureLoader.h"


namespace Saiga {

bool operator==(const TextureParameters &lhs, const TextureParameters &rhs) {
    return std::tie(lhs.srgb) == std::tie(rhs.srgb);
}



std::shared_ptr<Texture> TextureLoader::textureFromImage(Image &im, const TextureParameters &params) const
{
    auto text = std::make_shared<Texture>();

//    im.Format().setSrgb(params.srgb);
    bool erg = text->fromImage(im,params.srgb);
    if(erg){
        return text;
    }else{
//        delete text;
    }
    return nullptr;
}

std::shared_ptr<Texture> TextureLoader::loadFromFile(const std::string &path, const TextureParameters &params){

    bool erg;

    Image im;
    erg = im.load(path);

    if (erg)
    {
        auto text = std::make_shared<Texture>();
//        im.to8bitImage();
//        im.Format().setSrgb(params.srgb);
        erg = text->fromImage(im,params.srgb);
        return text;
    }

    return nullptr;
}


}


