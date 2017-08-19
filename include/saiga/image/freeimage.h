/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/image/image.h"
#include "saiga/image/imageMetaData.h"

#ifdef SAIGA_USE_FREEIMAGE

class fipImage;

namespace Saiga {
namespace FIP {

//loads an image with freeimage converts it to Image and reads metadata if != 0
SAIGA_GLOBAL bool load(const std::string& path, Image& img, ImageMetadata* metaData = nullptr);
//writes an image to file with freeimage
SAIGA_GLOBAL bool save(const std::string& path, const Image& img);

//helper functions if you want to actually have fipimages
//these are used by load and save from above
SAIGA_GLOBAL bool loadFIP(const std::string& path, fipImage& img);
SAIGA_GLOBAL bool saveFIP(const std::string& path, const fipImage& img);

//conersion between saiga's image and freeimage's fipimage
//these are used by load and save from above
SAIGA_GLOBAL extern void convert(const fipImage &src, Image& dest);
SAIGA_GLOBAL extern void convert(const Image &src, fipImage &dest); //copy the src image because we need to flip red and blue :(

//reads the meta data of a fipimage
//is used by load from above
SAIGA_GLOBAL extern void getMetaData(fipImage &img, ImageMetadata& metaData);
SAIGA_GLOBAL extern void printAllMetaData(fipImage &img);

}

}

#endif
