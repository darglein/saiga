/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/image/image.h"

namespace Saiga {

//class fipImage;
namespace PNG{
class PngImage;
}

/**
 * Converts between Image types from the image loader libraries for example 'libpng' to
 * saigas 'Image' type.
 * @brief The ImageConverter class
 */
class SAIGA_GLOBAL ImageConverter{
public:
#ifdef SAIGA_USE_PNG
    static void convert(PNG::PngImage& src, Image& dst);
    static void convert(Image &src, PNG::PngImage &dst);
#endif

#ifdef SAIGA_USE_FREEIMAGE
//    static void convert(fipImage &src, Image& dest);
//    static void convert(Image src, fipImage &dest); //copy the src image because we need to flip red and blue :(
#endif
};

}
