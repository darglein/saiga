/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

/*
 * credits for write PNG:
 * http://www.libpng.org/pub/png/book/chapter15.html
 * credits for read PNG:
 * http://www.libpng.org/pub/png/book/chapter13.html
 * http://blog.nobel-joergensen.com/2010/11/07/loading-a-png-as-texture-in-opengl-using-libpng/
 */

#include "managedImage.h"

#ifdef SAIGA_USE_PNG

namespace Saiga
{
namespace PNG
{
using uchar = unsigned char;

struct SAIGA_CORE_API PngImage
{
    // image size
    size_t width, height;

    // number of bits per color. 8 for basic rgb(a) images
    int bit_depth;
    // PNG_COLOR_TYPE_GRAY,PNG_COLOR_TYPE_GRAY_ALPHA,PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGB_ALPHA
    int color_type;

    // raw image data
    std::vector<uchar> data;
    uchar* data2;

    int rowAlignment = 4;
    size_t bytesPerRow;


    void* rowPtr(int i) { return data.data() + bytesPerRow * i; }
    ImageType saigaType() const;
    void fromSaigaType(ImageType t);
};


//    SAIGA_LOCAL void pngVersionInfo();

//    SAIGA_LOCAL bool readPNG (PngImage *img, const std::string &path, bool invertY = false);
//    SAIGA_LOCAL bool writePNG(PngImage *img, const std::string &path, bool invertY = false);

//    SAIGA_LOCAL void convert(PNG::PngImage& src, Image& dst);
//    SAIGA_LOCAL void convert(Image &src, PNG::PngImage &dst);


SAIGA_LOCAL bool save(const Image& img, const std::string& path, bool invertY = false);
SAIGA_LOCAL bool load(Image& img, const std::string& path, bool invertY = false);

}  // namespace PNG
}  // namespace Saiga

#endif
