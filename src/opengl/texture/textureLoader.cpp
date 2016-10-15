
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

    //use libfreeimage if available, libpng otherwise
#ifdef USE_FREEIMAGE
    fipImage img;
    erg = img.load(path.c_str());
    if(erg)
        ImageConverter::convert(img,outImage);
#else
#ifdef USE_PNG
    PNG::Image pngimg;
    erg = PNG::readPNG( &pngimg,path);
    if(erg)
        ImageConverter::convert(pngimg,outImage);
#endif
#endif

    if(erg){
        std::cout<<"Loaded: "<< path << " " << outImage << std::endl;
    }else{
        std::cout << "Error: Could not load Image: " << path << std::endl;
        assert(0);
    }

    return erg;
}

bool TextureLoader::saveImage(const std::string &path, Image &image) const
{
    bool erg = false;

    //use libfreeimage if available, libpng otherwise
#ifdef USE_FREEIMAGE
    fipImage fipimage;
    ImageConverter::convert(image,fipimage);
    erg = fipimage.save(path.c_str());
#else
#ifdef USE_PNG
    PNG::Image pngimg;
    ImageConverter::convert(image,pngimg);
    erg = PNG::writePNG(&pngimg,path);
#endif
#endif

    if(erg){
        std::cout<<"Saved: "<< path << " " << image << std::endl;
    }else{
        std::cout << "Error: Could not save Image: " << path << std::endl;
        assert(0);
    }


    return erg;
}




