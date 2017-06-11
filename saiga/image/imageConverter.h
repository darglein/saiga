#pragma once


#include "saiga/opengl/opengl.h"
#include "saiga/image/image.h"

class fipImage;
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
//    static void convert(PNG::PngImage &src, Image& dest);
//    static void convert(Image& src, PNG::PngImage &dest);
#endif

#ifdef SAIGA_USE_FREEIMAGE
//    static void convert(fipImage &src, Image& dest);
//    static void convert(Image src, fipImage &dest); //copy the src image because we need to flip red and blue :(
#endif
};
