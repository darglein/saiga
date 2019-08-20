/**
 * Copyright (c) 2017 Darius Rückert
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
namespace Saiga
{
namespace PNG
{
struct PNGLoadStore
{
    // temp variables for libpng. Don't modify them!!!
    uchar** row_pointers;
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


bool readPNG(PngImage* img, const std::string& path, bool invertY)
{
    PNGLoadStore pngls;

    png_structp png_ptr;
    png_infop info_ptr;

    unsigned int sig_read = 0;
    int interlace_type;

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

    /*
     * If you have enough memory to read
     * in the entire image at once, and
     * you need to specify only
     * transforms that can be controlled
     * with one of the PNG_TRANSFORM_*
     * bits (this presently excludes
     * dithering, filling, setting
     * background, and doing gamma
     * adjustment), then you can read the
     * entire image (including pixels)
     * into the info structure with this
     * call
     *
     * PNG_TRANSFORM_STRIP_16 |
     * PNG_TRANSFORM_PACKING  forces 8 bit
     * PNG_TRANSFORM_EXPAND forces to
     *  expand a palette into RGB
     */
    png_read_png(png_ptr, info_ptr,
                 //                 PNG_TRANSFORM_STRIP_16 | //Strip 16-bit samples to 8 bits
                 PNG_TRANSFORM_SWAP_ENDIAN |  // png byte order is big endian!
                     PNG_TRANSFORM_PACKING |  // Expand 1, 2 and 4-bit samples to bytes
                     PNG_TRANSFORM_EXPAND     // Perform set_expand()
                 ,
                 NULL);



    png_uint_32 pw, ph;
    png_get_IHDR(png_ptr, info_ptr, &pw, &ph, &img->bit_depth, &img->color_type, &interlace_type, NULL, NULL);
    SAIGA_ASSERT(interlace_type == PNG_INTERLACE_NONE);
    img->width  = pw;
    img->height = ph;


    unsigned int row_bytes = png_get_rowbytes(png_ptr, info_ptr);

    // we want to row-align the image in our output data
    int rowPadding   = (img->rowAlignment - (row_bytes % img->rowAlignment)) % img->rowAlignment;
    img->bytesPerRow = row_bytes + rowPadding;

    img->data.resize(img->bytesPerRow * img->height);

    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);



    if (invertY)
    {
        for (unsigned int i = 0; i < img->height; i++)
        {
            memcpy(img->data.data() + (img->bytesPerRow * (img->height - 1 - i)), row_pointers[i], row_bytes);
        }
    }
    else
    {
        for (unsigned int i = 0; i < img->height; i++)
        {
            memcpy(img->data.data() + (img->bytesPerRow * i), row_pointers[i], row_bytes);
        }
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

static int writepng_init(const Image& img, PNGLoadStore* pngls)
{
    png_structp png_ptr; /* note:  temporary variables! */
    png_infop info_ptr;
    int interlace_type;


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



    interlace_type = PNG_INTERLACE_NONE;  // PNG_INTERLACE_ADAM7


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
            SAIGA_ASSERT(0);
    }



    png_set_IHDR(png_ptr, info_ptr, img.width, img.height, bit_depth, color_type, interlace_type,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    // Higher is more compression
    // PNG_TEXT_Z_DEFAULT_STRATEGY
    png_set_compression_level(png_ptr, 1);

    // One of the following
    // PNG_FAST_FILTERS, PNG_FILTER_NONE, PNG_ALL_FILTERS
    png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_NONE);


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

    if (bit_depth == 16 || bit_depth == 32) png_set_swap(png_ptr);

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


    for (int i = 0; i < img.height; i++)
    {
        auto j      = invertY ? img.height - i - 1 : i;
        auto offset = j * img.pitchBytes;
        auto rowPtr = img.data8() + offset;

#    if 0
        std::vector<unsigned char> dataTest(img.pitchBytes);
        for (auto c : img.colRange())
        {
            png_save_uint_16(dataTest.data() + c * sizeof(short), ((unsigned short*)rowPtr)[c]);
        }
#    endif
        png_write_row(png_ptr, rowPtr);
        //        png_write_row(png_ptr, dataTest.data());
    }


    png_write_end(png_ptr, NULL);

    png_destroy_write_struct(&png_ptr, &info_ptr);
}

#    if 0
bool writePNG(PngImage *img, const std::string &path, bool invertY)
{
    PNGLoadStore pngls;

    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
        std::cout << "could not open file: " << path.c_str() << std::endl;
        return false;
    }

    pngls.outfile = fp;


    if(writepng_init(img,&pngls)!=0)
    {
        std::cout<<"error write png init"<<std::endl;
    }

    writepng_encode_image(img,&pngls,invertY);


    fclose(fp);

    return true;

}

#    endif
ImageType PngImage::saigaType() const
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
    }
    SAIGA_ASSERT(channels != -1);

    ImageElementType elementType = IET_ELEMENT_UNKNOWN;
    switch (bit_depth)
    {
        case 8:
            elementType = IET_UCHAR;
            break;
        case 16:
            elementType = IET_USHORT;
            break;
        case 32:
            elementType = IET_UINT;
            break;
    }
    SAIGA_ASSERT(elementType != IET_ELEMENT_UNKNOWN);

    return getType(channels, elementType);
}

void PngImage::fromSaigaType(ImageType t)
{
    switch (channels(t))
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
            color_type = PNG_COLOR_TYPE_RGB;
            SAIGA_ASSERT(0);
    }

    switch (elementType(t))
    {
        case IET_UCHAR:
            bit_depth = 8;
            break;
        case IET_USHORT:
            bit_depth = 16;
            break;
        case IET_UINT:
            bit_depth = 32;
            break;
        default:
            bit_depth = 0;
            SAIGA_ASSERT(0);
    }
}


void convert(PNG::PngImage& src, Image& dest)
{
    dest.width  = src.width;
    dest.height = src.height;

    dest.type = src.saigaType();


    dest.create();

    for (int i = 0; i < dest.rows; ++i)
    {
        memcpy(dest.rowPtr(i), src.rowPtr(i), std::min(dest.pitchBytes, src.bytesPerRow));
    }
}

void convert(Image& src, PNG::PngImage& dest)
{
    dest.width  = src.width;
    dest.height = src.height;

    dest.fromSaigaType(src.type);

    // The rows must be 4-aligned
    SAIGA_ASSERT(src.pitchBytes % 4 == 0);

    dest.bytesPerRow = iAlignUp(elementSize(src.type) * src.width, 4);
    //    dest.data.resize(dest.bytesPerRow * src.height);

    dest.data2 = src.data8();

    for (int i = 0; i < src.rows; ++i)
    {
        //        memcpy(dest.rowPtr(i),src.rowPtr(i), std::min(src.pitchBytes,dest.bytesPerRow));
    }
}

bool save(const Image& img, const std::string& path, bool invertY)
{
    //    PNG::PngImage pngimg;
    //    PNG::convert(img,pngimg);
    //    return PNG::writePNG(&pngimg,path,invertY);

    PNGLoadStore pngls;

    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
        std::cout << "could not open file: " << path.c_str() << std::endl;
        return false;
    }

    pngls.outfile = fp;


    if (writepng_init(img, &pngls) != 0)
    {
        std::cout << "error write png init" << std::endl;
    }

    writepng_encode_image(img, &pngls, invertY);


    fclose(fp);

    return true;
}

bool load(Image& img, const std::string& path, bool invertY)
{
    PNG::PngImage pngimg;
    bool erg = PNG::readPNG(&pngimg, path, invertY);
    if (erg) PNG::convert(pngimg, img);
    return erg;
}

}  // namespace PNG
}  // namespace Saiga
#endif
