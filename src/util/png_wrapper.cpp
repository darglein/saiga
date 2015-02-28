#include "util/png_wrapper.h"

#ifdef USE_PNG

bool PNG::readPNG(Image *img, const std::string &path, bool invertY){
    png_structp png_ptr;
    png_infop info_ptr;

    unsigned int sig_read = 0;
    int  interlace_type;

    if ((img->infile = fopen(path.c_str(), "rb")) == NULL)
        return false;

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
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                     NULL, NULL, NULL);

    if (png_ptr == NULL) {
        fclose(img->infile);
        return false;
    }

    /* Allocate/initialize the memory
     * for image information.  REQUIRED. */
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        fclose(img->infile);
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
    if (setjmp(png_jmpbuf(png_ptr))) {
        /* Free all of the memory associated
         * with the png_ptr and info_ptr */
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(img->infile);
        /* If we get here, we had a
         * problem reading the file */
        return false;
    }

    /* Set up the output control if
     * you are using standard C streams */
    png_init_io(png_ptr, img->infile);

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
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_STRIP_16 | PNG_TRANSFORM_PACKING | PNG_TRANSFORM_EXPAND, NULL);


    png_get_IHDR(png_ptr, info_ptr, &img->width, &img->height, &img->bit_depth, &img->color_type,
                 &interlace_type, NULL, NULL);


    unsigned int row_bytes = png_get_rowbytes(png_ptr, info_ptr);
    img->data = new unsigned char[row_bytes * img->height];

    png_bytepp row_pointers = png_get_rows(png_ptr, info_ptr);



    if(invertY){
        for (unsigned int i = 0; i < img->height; i++) {
            memcpy(img->data+(row_bytes * (img->height-1-i)), row_pointers[i], row_bytes);
        }
    }else{
        for (unsigned int i = 0; i < img->height; i++) {
            memcpy(img->data+(row_bytes * i), row_pointers[i], row_bytes);
        }
    }

    /* Clean up after the read,
     * and free any memory allocated */
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    /* Close the file */
    fclose(img->infile);

    /* That's it */
    return true;
}


/* returns 0 for success, 2 for libpng problem, 4 for out of memory, 11 for
 *  unexpected pnmtype; note that outfile might be stdout */

int writepng_init(PNG::Image *image)
{
    png_structp  png_ptr;       /* note:  temporary variables! */
    png_infop  info_ptr;
    int  interlace_type;


    /* could also replace libpng warning-handler (final NULL), but no need: */

    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, image,PNG::writepng_error_handler, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */

    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        return 4;   /* out of memory */
    }


    /* setjmp() must be called in every function that calls a PNG-writing
     * libpng function, unless an alternate error handler was installed--
     * but compatible error handlers must either use longjmp() themselves
     * (as in this program) or exit immediately, so here we go: */

    if (setjmp(image->jmpbuf)) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        return 2;
    }


    /* make sure outfile is (re)opened in BINARY mode */

    png_init_io(png_ptr, image->outfile);


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




    interlace_type = PNG_INTERLACE_NONE; //PNG_INTERLACE_ADAM7

    png_set_IHDR(png_ptr, info_ptr, image->width, image->height,
                 image->bit_depth, image->color_type, interlace_type,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    //    if (mainprog_ptr->gamma > 0.0)
    //        png_set_gAMA(png_ptr, info_ptr, mainprog_ptr->gamma);

    //    if (mainprog_ptr->have_bg) {   /* we know it's RGBA, not gray+alpha */
    //        png_color_16  background;

    //        background.red = mainprog_ptr->bg_red;
    //        background.green = mainprog_ptr->bg_green;
    //        background.blue = mainprog_ptr->bg_blue;
    //        png_set_bKGD(png_ptr, info_ptr, &background);
    //    }

    //    if (mainprog_ptr->have_time) {
    //        png_time  modtime;

    //        png_convert_from_time_t(&modtime, mainprog_ptr->modtime);
    //        png_set_tIME(png_ptr, info_ptr, &modtime);
    //    }

    //    if (mainprog_ptr->have_text) {
    //        png_text  text[6];
    //        int  num_text = 0;

    //        if (mainprog_ptr->have_text & TEXT_TITLE) {
    //            text[num_text].compression = PNG_TEXT_COMPRESSION_NONE;
    //            text[num_text].key = "Title";
    //            text[num_text].text = mainprog_ptr->title;
    //            ++num_text;
    //        }
    //        if (mainprog_ptr->have_text & TEXT_AUTHOR) {
    //            text[num_text].compression = PNG_TEXT_COMPRESSION_NONE;
    //            text[num_text].key = "Author";
    //            text[num_text].text = mainprog_ptr->author;
    //            ++num_text;
    //        }
    //        if (mainprog_ptr->have_text & TEXT_DESC) {
    //            text[num_text].compression = PNG_TEXT_COMPRESSION_NONE;
    //            text[num_text].key = "Description";
    //            text[num_text].text = mainprog_ptr->desc;
    //            ++num_text;
    //        }
    //        if (mainprog_ptr->have_text & TEXT_COPY) {
    //            text[num_text].compression = PNG_TEXT_COMPRESSION_NONE;
    //            text[num_text].key = "Copyright";
    //            text[num_text].text = mainprog_ptr->copyright;
    //            ++num_text;
    //        }
    //        if (mainprog_ptr->have_text & TEXT_EMAIL) {
    //            text[num_text].compression = PNG_TEXT_COMPRESSION_NONE;
    //            text[num_text].key = "E-mail";
    //            text[num_text].text = mainprog_ptr->email;
    //            ++num_text;
    //        }
    //        if (mainprog_ptr->have_text & TEXT_URL) {
    //            text[num_text].compression = PNG_TEXT_COMPRESSION_NONE;
    //            text[num_text].key = "URL";
    //            text[num_text].text = mainprog_ptr->url;
    //            ++num_text;
    //        }
    //        png_set_text(png_ptr, info_ptr, text, num_text);
    //    }


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


    /* make sure we save our pointers for use in writepng_encode_image() */

    image->png_ptr = png_ptr;
    image->info_ptr = info_ptr;


    /* OK, that's all we need to do for now; return happy */

    return 0;
}





/* returns 0 for success, 2 for libpng (longjmp) problem */

int writepng_encode_image(PNG::Image *image,bool invertY)
{
    png_structp png_ptr = (png_structp)image->png_ptr;
    png_infop info_ptr = (png_infop)image->info_ptr;


    /* as always, setjmp() must be called in every function that calls a
     * PNG-writing libpng function */

    if (setjmp(image->jmpbuf)) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        image->png_ptr = NULL;
        image->info_ptr = NULL;
        return 2;
    }

    int color_channels = 0;
    switch(image->color_type){
    case PNG_COLOR_TYPE_GRAY:
        color_channels = 1;
        break;
    case PNG_COLOR_TYPE_RGB:
        color_channels = 3;
        break;
    case PNG_COLOR_TYPE_RGB_ALPHA:
        color_channels = 4;
        break;
    }

    int bytes_per_pixel = color_channels * image->bit_depth/8;



    //the row pointers point to the first byte of each row of the image
    image->row_pointers = new uchar*[image->height];

    if(invertY){
        for(unsigned int i=0;i<image->height;i++){
            int offset = i*image->width*bytes_per_pixel;
            image->row_pointers[image->height-i-1] = image->data+offset;
        }
    }else{


        for(unsigned int i=0;i<image->height;i++){
            int offset = i*image->width*bytes_per_pixel;
            image->row_pointers[i] = image->data+offset;
        }
    }
    /* and now we just write the whole image; libpng takes care of interlacing
     * for us */

    png_write_image(png_ptr, image->row_pointers);


    /* since that's it, we also close out the end of the PNG file now--if we
     * had any text or time info to write after the IDATs, second argument
     * would be info_ptr, but we optimize slightly by sending NULL pointer: */

    png_write_end(png_ptr, NULL);

    return 0;
}





/* returns 0 if succeeds, 2 if libpng problem */

int writepng_encode_row(PNG::Image *image)  /* NON-interlaced only! */
{
    png_structp png_ptr = (png_structp)image->png_ptr;
    png_infop info_ptr = (png_infop)image->info_ptr;


    /* as always, setjmp() must be called in every function that calls a
     * PNG-writing libpng function */

    if (setjmp(image->jmpbuf)) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        image->png_ptr = NULL;
        image->info_ptr = NULL;
        return 2;
    }


    /* image_data points at our one row of image data */

    png_write_row(png_ptr, image->data);

    return 0;
}





/* returns 0 if succeeds, 2 if libpng problem */

int writepng_encode_finish(PNG::Image *image)   /* NON-interlaced! */
{
    png_structp png_ptr = (png_structp)image->png_ptr;
    png_infop info_ptr = (png_infop)image->info_ptr;


    /* as always, setjmp() must be called in every function that calls a
     * PNG-writing libpng function */

    if (setjmp(image->jmpbuf)) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        image->png_ptr = NULL;
        image->info_ptr = NULL;
        return 2;
    }


    /* close out PNG file; if we had any text or time info to write after
     * the IDATs, second argument would be info_ptr: */

    png_write_end(png_ptr, NULL);

    return 0;
}





void writepng_cleanup(PNG::Image *image)
{
    png_structp png_ptr = (png_structp)image->png_ptr;
    png_infop info_ptr = (png_infop)image->info_ptr;

    if (png_ptr && info_ptr)
        png_destroy_write_struct(&png_ptr, &info_ptr);
}





void PNG::writepng_error_handler(png_structp png_ptr, png_const_charp msg)
{
    PNG::Image  *image;

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

    image = static_cast<PNG::Image*>(png_get_error_ptr(png_ptr));
    if (image == NULL) {         /* we are completely hosed now */
        fprintf(stderr,
                "writepng severe error:  jmpbuf not recoverable; terminating.\n");
        fflush(stderr);
        exit(99);
    }

    longjmp(image->jmpbuf, 1);
}

void PNG::pngVersionInfo()
{
    std::cout<<"libpng version: "<<png_libpng_ver<< std::endl;
    //std::cout<< "zlib version: "<<zlib_version<< std::endl;
}

bool PNG::writePNG(Image *img, const std::string &path, bool invertY){
    std::cout<<"write png: "<<path.c_str()<<std::endl;

    FILE *fp = fopen(path.c_str(), "wb");
    if (!fp)
    {
		std::cout << "could not open file: " << path.c_str() << std::endl;
        return false;
    }

    img->outfile = fp;




    if(writepng_init(img)!=0){
        std::cout<<"error write png init"<<std::endl;
    }
    if(writepng_encode_image(img,invertY)!=0){
        std::cout<<"error write encode image"<<std::endl;
    }


    writepng_cleanup(img);

    fclose(fp);

    return true;

}

#endif
