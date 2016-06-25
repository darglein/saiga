
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/opengl/texture/imageConverter.h"
#include <FreeImagePlus.h>
#include "saiga/util/png_wrapper.h"

bool operator==(const TextureParameters &lhs, const TextureParameters &rhs) {
    return std::tie(lhs.srgb) == std::tie(rhs.srgb);
}



Texture* TextureLoader::textureFromImage(Image &im, const TextureParameters &params) const{
    Texture* text = new Texture();

    im.Format().setSrgb(params.srgb);
    bool erg = text->fromImage(im);
    if(erg){
        return text;
    }else{
        delete text;
    }
    return nullptr;
}

Texture* TextureLoader::loadFromFile(const std::string &path, const TextureParameters &params){

    bool erg;
    Texture* text = new Texture();

    //    PNG::Image img;
    //    erg = PNG::readPNG( &img,path);
    //    cout<<"loading "<<path<<endl;

//    fipImage fipimg;
//    erg = fipimg.load(path.c_str());
    Image im;
    erg = loadImage(path,im);

    if (erg){
        im.Format().setSrgb(params.srgb);
        erg = text->fromImage(im);
    }

    if(erg){
        return text;
    }else{
        delete text;
    }

    return nullptr;
}

bool TextureLoader::loadImage(const std::string &path, Image &outImage) const
{
    bool erg = false;

#ifdef USE_FREEIMAGE
    fipImage img;
    erg = img.load(path.c_str());
    if(erg)
        ImageConverter::convert(img,outImage);
#else

#ifdef USE_PNG
    PNG::Image img;
    erg = PNG::readPNG( &img,path);
    if(erg)
        ImageConverter::convert(img,outImage);
#endif
#endif
    return erg;
}

bool TextureLoader::saveImage(const std::string &path, Image &image) const
{

    fipImage fipimage;
    ImageConverter::convert(image,fipimage);
    return fipimage.save(path.c_str());
}




