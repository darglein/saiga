/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/image/imageFormat.h"
#include "saiga/cuda/imageProcessing/imageView.h"
#include <stdint.h>
#include <vector>

namespace Saiga {

class SAIGA_GLOBAL Image{
public:
    using byte_t = unsigned char;


    //image dimensions
    int width = 0;
    int height = 0;
    int pitch;
    //raw image data
    std::vector<byte_t> data;
protected:
    //size of data in bytes
    //    int size = 0;

    ImageFormat format;

    //Alignment of the beginning of each row. Allowed values: 1,2,4,8
    int rowAlignment = 4;
public:

    Image(){}
    Image(ImageFormat _format) : format(_format) {}
    Image(ImageFormat format, int w, int h, void* data);
    Image(ImageFormat format, int w, int h, int p, void* data);


    //raw image data
    byte_t* getRawData();
    const byte_t* getRawData() const;

    //byte offset of the given texel in the raw data
    int position(int x, int y);
    //pointer to the beginning of a given texel
    byte_t* positionPtr(int x, int y);


    void setPixel(int x, int y, void* data);
    void setPixel(int x, int y, uint8_t data);
    void setPixel(int x, int y, uint16_t data);
    void setPixel(int x, int y, uint32_t data);

    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b);


    template<typename T>
    ImageView<T> getImageView(){
        ImageView<T> res(width,height,getBytesPerRow(),getRawData());
        return res;
    }

    template<typename T>
    T getPixel(int x, int y){
        return *(T*)positionPtr(x,y);
    }


    //zeros out all bytes
    void makeZero();

    //allocates the memory to hold this image.
    void create(byte_t* initialData = nullptr);

    //resizes the image. The data is undefined.
    void resize(int w, int h);

    //resizes the image. This adds 0 rows and columns to the right and bottom. The top left is the original image.
    //if the new image is smaller the image is cropped to the new size.
    void resizeCopy(int w, int h);

    void setSubImage(int x, int y, Image &src);
    void setSubImage(int x, int y , int w , int h , uint8_t* data);
    void getSubImage(int x, int y, int w, int h, Image &out);



    //swaps the red and blue color channel. Usefull for RGB->BGR conversion
    void flipRB();

    //flips the rows. Some images require the origin to be in the top left and some in the bottom left
    void flipY();

    //convert to 8-bit bitmap if image is in heigher bitdepth
    void to8bitImage();

    //removes alpha channel if this image has one
    void removeAlpha();

    //======================================================



    size_t getSize();


    ImageFormat& Format();
    const ImageFormat& Format() const;
    int getBytesPerRow() const;
};

/**
 * Loads an image from file.
 * Uses libfreeimage if possible and libpng otherwise.
 *
 * Returns true when the images was loaded successfully.
 */
SAIGA_GLOBAL bool loadImage(const std::string &path, Image& outImage);

/**
 * Saves an image to file.
 * Uses libfreeimage if possible and libpng otherwise.
 *
 * Returns false when the images was loaded successfully.
 */
SAIGA_GLOBAL bool saveImage(const std::string &path, const Image& image);


SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const Image& f);

}
