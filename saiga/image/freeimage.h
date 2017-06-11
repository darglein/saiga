#pragma once

#include "saiga/config.h"
#include "saiga/image/image.h"
#include "saiga/image/imageMetaData.h"

#ifdef SAIGA_USE_FREEIMAGE




class fipImage;


namespace FIP {

//loads an image with freeimage converts it to Image and reads metadata if != 0
SAIGA_GLOBAL bool load(const std::string& path, Image& img, ImageMetadata* metaData = nullptr);
SAIGA_GLOBAL bool save(const std::string& path, Image& img);


SAIGA_GLOBAL bool loadFIP(const std::string& path, fipImage& img);

SAIGA_GLOBAL bool saveFIP(const std::string& path, fipImage& img);


SAIGA_GLOBAL extern void convert(fipImage &src, Image& dest);
SAIGA_GLOBAL extern void convert(Image src, fipImage &dest); //copy the src image because we need to flip red and blue :(

SAIGA_GLOBAL extern void getMetaData(fipImage &img, ImageMetadata& metaData);

}


#endif
