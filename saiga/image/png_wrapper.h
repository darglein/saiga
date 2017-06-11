#pragma once

/*
 * credits for write PNG:
 * http://www.libpng.org/pub/png/book/chapter15.html
 * credits for read PNG:
 * http://www.libpng.org/pub/png/book/chapter13.html
 * http://blog.nobel-joergensen.com/2010/11/07/loading-a-png-as-texture-in-opengl-using-libpng/
 */

#include <saiga/config.h>

#ifdef SAIGA_USE_PNG


#include <png.h>
#include <vector>


namespace PNG{
using uchar = unsigned char;

    struct SAIGA_GLOBAL PngImage{
        //image size
        png_uint_32 width, height;

        //number of bits per color. 8 for basic rgb(a) images
        int bit_depth;
        //PNG_COLOR_TYPE_GRAY,PNG_COLOR_TYPE_GRAY_ALPHA,PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGB_ALPHA
        int color_type;

        //raw image data
        std::vector<uchar> data;

        int rowAlignment = 4;
        int bytesPerRow;

        //temp variables for libpng. Don't modify them!!!
        uchar **row_pointers;
        void *png_ptr;
        void *info_ptr;
        FILE *infile;
        FILE *outfile;
        jmp_buf jmpbuf;
    };


    SAIGA_GLOBAL void pngVersionInfo();

    SAIGA_GLOBAL bool readPNG (PngImage *img, const std::string &path, bool invertY = true);
    SAIGA_GLOBAL bool writePNG(PngImage *img, const std::string &path, bool invertY = true);


}

#endif

