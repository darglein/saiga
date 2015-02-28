#pragma once

/*
 * credits for write PNG:
 * http://www.libpng.org/pub/png/book/chapter15.html
 * credits for read PNG:
 * http://www.libpng.org/pub/png/book/chapter13.html
 * http://blog.nobel-joergensen.com/2010/11/07/loading-a-png-as-texture-in-opengl-using-libpng/
 */

#ifdef USE_PNG

#include <png.h>
#include <iostream>

typedef unsigned char uchar;



class PNG{
public:
    struct Image{
        png_uint_32 width, height;
        int bit_depth;
        int color_type;
        uchar* data;


        uchar **row_pointers;
        void *png_ptr;
        void *info_ptr;
        FILE *infile;
        FILE *outfile;
        jmp_buf jmpbuf;


    };
    static void writepng_error_handler(png_structp png_ptr, png_const_charp msg);

    static void pngVersionInfo();

    static bool readPNG(Image *img,const std::string &path, bool invertY=true);
    static bool writePNG(Image *img,const std::string &path, bool invertY=true);


};

#endif

