/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "png_wrapper.h"

#include "saiga/core/util/assert.h"

#include <cstring>  // for memcpy
#include <iostream>

#ifdef SAIGA_USE_PNG
#    include "internal/noGraphicsAPI.h"

#    include <png.h>
#    include <zlib.h>
namespace Saiga
{
namespace LibPNG
{
ImageType saigaType(int color_type, int bit_depth)
{
    int channels = -1;
    switch (color_type)
    {
        case PNG_COLOR_TYPE_GRAY:
            channels = 1;
            break;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            channels = 2;
            break;
        case PNG_COLOR_TYPE_RGB:
            channels = 3;
            break;
        case PNG_COLOR_TYPE_RGB_ALPHA:
            channels = 4;
            break;
        default:
            std::cout << "unknown png color type: " << color_type << std::endl;
            SAIGA_EXIT_ERROR("Unknown Color Type");
    }

    ImageElementType elementType = ImageElementType::IET_ELEMENT_UNKNOWN;
    switch (bit_depth)
    {
        case 1:
        case 2:
        case 4:
            // We have set png_expand from 1,2,4 bit to 8-bit
            SAIGA_ASSERT(PNG_COLOR_TYPE_GRAY == PNG_COLOR_TYPE_GRAY);
        case 8:
            elementType = ImageElementType::IET_UCHAR;
            break;
        case 16:
            elementType = ImageElementType::IET_USHORT;
            break;
        case 32:
            elementType = ImageElementType::IET_UINT;
            break;
        default:
            std::cout << "unknown bit depth: " << bit_depth << std::endl;
            SAIGA_EXIT_ERROR("Unknown bit depth");
    }


    return getType(channels, elementType);
}

struct PNGLoadStore
{
    // temp variables for libpng. Don't modify them!!!
    png_byte** row_pointers;
    void* png_ptr;
    void* info_ptr;
    FILE* infile;
    FILE* outfile;
    jmp_buf jmpbuf;
};

static void writepng_error_handler(png_structp png_ptr, png_const_charp msg)
{
    PNGLoadStore* image;

    /* This function, aside from the extra step of retrieving the "error
     * pointer" (below) and the fact that it exists within the application
     * rather than within libpng, is essentially identical to libpng's
     * default error handler.  The second point is critical:  since both
     * setjmp() and longjmp() are called from the same code, they are
     * guaranteed to have compatible notions of how big a jmp_buf is,
     * regardless of whether _BSD_SOURCE or anything else has (or has not)
     * been defined. */

    fprintf(stderr, "writepng libpng error: %s\n", msg);
    fflush(stderr);

    image = static_cast<PNGLoadStore*>(png_get_error_ptr(png_ptr));
    if (image == NULL)
    { /* we are completely hosed now */
        fprintf(stderr, "writepng severe error:  jmpbuf not recoverable; terminating.\n");
        fflush(stderr);
        SAIGA_ASSERT(0);
    }

    longjmp(image->jmpbuf, 1);
}

bool load(const std::string& path, Image& img, bool invertY)
{
    PNGLoadStore pngls;

    png_structp png_ptr;
    png_infop info_ptr;

    unsigned int sig_read = 0;


    if ((pngls.infile = fopen(path.c_str(), "rb")) == NULL) return false;

    /* Create and initialize the png_struct
     * with the desired error handler
     * functions.  If you want to use the
     * default stderr and longjump method,
     * you can supply NULL for the last
     * three parameters.  We also supply the
     * the compiler header file version, so
     * that we know if the application
     * was compiled with a compatible version
     * of the library.  REQUIRED
     */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);


    if (png_ptr == NULL)
    {
        fclose(pngls.infile);
        return false;
    }

    /* Allocate/initialize the memory
     * for image information.  REQUIRED. */
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        fclose(pngls.infile);
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return false;
    }

    /* Set error handling if you are
     * using the setjmp/longjmp method
     * (this is the normal method of
     * doing things with libpng).
     * REQUIRED unless you  set up
     * your own error handlers in
     * the png_create_read_struct()
     * earlier.
     */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        /* Free all of the memory associated
         * with the png_ptr and info_ptr */
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(pngls.infile);
        /* If we get here, we had a
         * problem reading the file */
        return false;
    }

    /* Set up the output control if
     * you are using standard C streams */
    png_init_io(png_ptr, pngls.infile);

    /* If we have already
     * read some of the signature */
    png_set_sig_bytes(png_ptr, sig_read);

    png_read_info(png_ptr, info_ptr);

    png_uint_32 pw, ph;
    int bit_depth;
    int color_type;
    int interlace_type;
    png_get_IHDR(png_ptr, info_ptr, &pw, &ph, &bit_depth, &color_type, &interlace_type, NULL, NULL);
    SAIGA_ASSERT(interlace_type == PNG_INTERLACE_NONE);

    unsigned int row_bytes = png_get_rowbytes(png_ptr, info_ptr);

    // we want to row-align the image in our output data
    int rowAlignment = 4;
    int rowPadding   = (rowAlignment - (row_bytes % rowAlignment)) % rowAlignment;
    int bytesPerRow  = row_bytes + rowPadding;



    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png_ptr);

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png_ptr);

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png_ptr);


    if (bit_depth > 8)
    {
        png_set_swap(png_ptr);
    }

    png_set_packing(png_ptr);

    //    img->data.resize(img->bytesPerRow * img->height);
    img.create(ph, pw, bytesPerRow, saigaType(color_type, bit_depth));
    img.makeZero();


    for (int i = 0; i < img.height; i++)
    {
        auto rowPtr = (png_byte*)img.rowPtr(invertY ? img.height - i - 1 : i);
        png_read_row(png_ptr, rowPtr, nullptr);
    }

    /* Clean up after the read,
     * and free any memory allocated */
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    /* Close the file */
    fclose(pngls.infile);

    /* That's it */
    return true;
}


/* returns 0 for success, 2 for libpng problem, 4 for out of memory, 11 for
 *  unexpected pnmtype; note that outfile might be stdout */

static int writepng_init(const Image& img, PNGLoadStore* pngls, Compression compression)
{
    png_structp png_ptr; /* note:  temporary variables! */
    png_infop info_ptr;


    /* could also replace libpng warning-handler (final NULL), but no need: */

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, writepng_error_handler, NULL);
    if (!png_ptr) return 4; /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        png_destroy_write_struct(&png_ptr, NULL);
        return 4; /* out of memory */
    }


    /* setjmp() must be called in every function that calls a PNG-writing
     * libpng function, unless an alternate error handler was installed--
     * but compatible error handlers must either use longjmp() themselves
     * (as in this program) or exit immediately, so here we go: */

    if (setjmp(pngls->jmpbuf))
    {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return 2;
    }


    /* make sure outfile is (re)opened in BINARY mode */

    png_init_io(png_ptr, pngls->outfile);


    /* set the compression levels--in general, always want to leave filtering
     * turned on (except for palette images) and allow all of the filters,
     * which is the default; want 32K zlib window, unless entire image buffer
     * is 16K or smaller (unknown here)--also the default; usually want max
     * compression (NOT the default); and remaining compression flags should
     * be left alone */

    // png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
    /*
    >> this is default for no filtering; Z_FILTERED is default otherwise:
    png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);
    >> these are all defaults:
    png_set_compression_mem_level(png_ptr, 8);
    png_set_compression_window_bits(png_ptr, 15);
    png_set_compression_method(png_ptr, 8);
 */


    /* set the image parameters appropriately */



    int bit_depth  = bitsPerChannel(img.type);
    int color_type = 0;

    switch (channels(img.type))
    {
        case 1:
            color_type = PNG_COLOR_TYPE_GRAY;
            break;
        case 2:
            color_type = PNG_COLOR_TYPE_GRAY_ALPHA;
            break;
        case 3:
            color_type = PNG_COLOR_TYPE_RGB;
            break;
        case 4:
            color_type = PNG_COLOR_TYPE_RGB_ALPHA;
            break;
        default:
            SAIGA_EXIT_ERROR("Invalid color type!");
    }



    png_set_IHDR(png_ptr, info_ptr, img.width, img.height, bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);


    switch (compression)
    {
        case Compression::fast:
            png_set_compression_strategy(png_ptr, Z_RLE);
            png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);
            png_set_compression_level(png_ptr, Z_BEST_SPEED);
            break;
        case Compression::medium:
            png_set_compression_strategy(png_ptr, Z_RLE);
            png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
            png_set_compression_level(png_ptr, Z_DEFAULT_COMPRESSION);
            break;
        case Compression::best:
            png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);
            png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
            png_set_compression_level(png_ptr, Z_BEST_COMPRESSION);
            break;
        default:
            SAIGA_EXIT_ERROR("Unknown Compression");
    }



    /* write all chunks up to (but not including) first IDAT */
    png_write_info(png_ptr, info_ptr);


    /* if we wanted to write any more text info *after* the image data, we
     * would set up text struct(s) here and call png_set_text() again, with
     * just the new data; png_set_tIME() could also go here, but it would
     * have no effect since we already called it above (only one tIME chunk
     * allowed) */


    /* set up the transformations:  for now, just pack low-bit-depth pixels
     * into bytes (one, two or four pixels per byte) */
    png_set_packing(png_ptr);

    /*  png_set_shift(png_ptr, &sig_bit);  to scale low-bit-depth values */

    png_set_swap(png_ptr);

    //    std::cout << "bit depth " << bit_depth << std::endl;

    /* make sure we save our pointers for use in writepng_encode_image() */

    pngls->png_ptr  = png_ptr;
    pngls->info_ptr = info_ptr;

    SAIGA_ASSERT(pngls->png_ptr);
    SAIGA_ASSERT(pngls->info_ptr);

    return 0;
}



static void writepng_encode_image(const Image& img, PNGLoadStore* pngls, bool invertY)
{
    png_structp png_ptr = (png_structp)pngls->png_ptr;
    png_infop info_ptr  = (png_infop)pngls->info_ptr;


    //    std::vector<png_byte*> rows(img.height);

    for (int i = 0; i < img.height; i++)
    {
        auto rowPtr = (png_byte*)img.rowPtr(invertY ? img.height - i - 1 : i);
        //        rows[i]     = rowPtr;
        png_write_row(png_ptr, rowPtr);
    }

    //    png_write_image(png_ptr, rows.data());

    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
}


bool save(const std::string& path, const Image& img, bool invertY, Compression compression)
{
    PNGLoadStore pngls;

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
        std::cout << "could not open file: " << path.c_str() << std::endl;
        return false;
    }

    pngls.outfile = fp;


    if (writepng_init(img, &pngls, compression) != 0)
    {
        std::cout << "error write png init" << std::endl;
    }

    writepng_encode_image(img, &pngls, invertY);

    fclose(fp);

    return true;
}


}  // namespace LibPNG
}  // namespace Saiga
#endif
