
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/image/imageConverter.h"
#ifdef SAIGA_USE_FREEIMAGE
#include <FreeImagePlus.h>
#include "saiga/image/freeimage.h"
#endif
#include "saiga/image/png_wrapper.h"

namespace Saiga {

bool operator==(const TextureParameters &lhs, const TextureParameters &rhs) {
    return std::tie(lhs.srgb) == std::tie(rhs.srgb);
}



std::shared_ptr<Texture> TextureLoader::textureFromImage(Image &im, const TextureParameters &params) const{
    auto text = std::make_shared<Texture>();

    im.Format().setSrgb(params.srgb);
    bool erg = text->fromImage(im);
    if(erg){
        return text;
    }else{
//        delete text;
    }
    return nullptr;
}

std::shared_ptr<Texture> TextureLoader::loadFromFile(const std::string &path, const TextureParameters &params){

    bool erg;
    auto text = std::make_shared<Texture>();

    Image im;
    erg = loadImage(path,im);

    if (erg){
        im.Format().setSrgb(params.srgb);
        erg = text->fromImage(im);
    }

    if(erg){
        return text;
    }else{
//        delete text;
    }

    return nullptr;
}

bool TextureLoader::loadImage(const std::string &path, Image &outImage) const
{
    bool erg = false;

    //use libfreeimage if available, libpng otherwise
#ifdef SAIGA_USE_FREEIMAGE
    erg = FIP::load(path,outImage,0);
//    fipImage img;
//    erg = img.load(path.c_str());
//    if(erg){
//        ImageConverter::convert(img,outImage);
//    }
#else
#ifdef SAIGA_USE_PNG
    PNG::Image pngimg;
    erg = PNG::readPNG( &pngimg,path);
    if(erg)
        ImageConverter::convert(pngimg,outImage);
#endif
#endif

    if(erg){
#ifndef SAIGA_RELEASE
        std::cout<<"Loaded: "<< path << " " << outImage << std::endl;
#endif
    }else{
        std::cout << "Error: Could not load image: " << path << std::endl;
        SAIGA_ASSERT(0);
    }

    return erg;
}

bool TextureLoader::saveImage(const std::string &path, Image &image) const
{
    bool erg = false;

    //use libfreeimage if available, libpng otherwise
#ifdef SAIGA_USE_FREEIMAGE
//    fipImage fipimage;
//    ImageConverter::convert(image,fipimage);
//    erg = fipimage.save(path.c_str());
    erg = FIP::save(path,image);
#else
#ifdef SAIGA_USE_PNG
    PNG::Image pngimg;
    ImageConverter::convert(image,pngimg);
    erg = PNG::writePNG(&pngimg,path);
#endif
#endif

    if(erg){
        std::cout<<"Saved: "<< path << " " << image << std::endl;
    }else{
        std::cout << "Error: Could not save Image: " << path << std::endl;
        SAIGA_ASSERT(0);
    }


    return erg;
}

}


