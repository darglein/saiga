#pragma once

/*
 * credits for write PNG:
 * http://www.libpng.org/pub/png/book/chapter15.html
 * credits for read PNG:
 * http://www.libpng.org/pub/png/book/chapter13.html
 * http://blog.nobel-joergensen.com/2010/11/07/loading-a-png-as-texture-in-opengl-using-libpng/
 */

#include <saiga/config.h>

#ifdef USE_PNG


#include <png.h>
#include <iostream>

typedef unsigned char uchar;



namespace PNG{

    struct Image{
        png_uint_32 width, height;
        int bit_depth;  //number of bits per color. 8 for basic rgb(a) images
        int color_type; //PNG_COLOR_TYPE_GRAY,PNG_COLOR_TYPE_GRAY_ALPHA,PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGB_ALPHA
        uchar* data;

        //temp variables for libpng. Don't modify them!!!
        uchar **row_pointers;
        void *png_ptr;
        void *info_ptr;
        FILE *infile;
        FILE *outfile;
        jmp_buf jmpbuf;
    };

    void writepng_error_handler(png_structp png_ptr, png_const_charp msg);

    void pngVersionInfo();

    bool readPNG(Image *img,const std::string &path, bool invertY=true);
    bool writePNG(Image *img,const std::string &path, bool invertY=true);


}

#endif

