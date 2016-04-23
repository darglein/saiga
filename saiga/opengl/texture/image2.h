#pragma once


#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/imageFormat.h"
#include <vector>

class Image2{
public:
    typedef unsigned char byte_t;

    //raw image data
    std::vector<byte_t> data;

    //image dimensions
    int size = 0; //size of data in bytes
    int width = 0;
    int height = 0;

    //alignment and helper values
    int rowAlignment = 4;
    int bytesPerRow = 0;
    int bytesPerTexel = 0;

    //image format
    bool srgb = false;
    int channels = 0;
    int bitDepth = 0;
    ImageFormat format;

    //opengl type for the image format
    GLenum glInternalType = GL_NONE;
    GLenum glType = GL_NONE;
};


template<int CHANNELS, int BITDEPTH, ImageFormat FORMAT, bool SRGB=false>
class TemplatedImage : public Image2{
public:
    typedef Texel<CHANNELS,BITDEPTH,FORMAT> texel_t;
    typedef GLImageFormatMap<CHANNELS,BITDEPTH,FORMAT,SRGB> imageFormat_t;

    TemplatedImage();
    TemplatedImage(int width, int height);

    void create();

    texel_t& getTexel(int x, int y);
    void setTexel(int x, int y, texel_t t);

    //pointer to the beginning of this row
    texel_t* rowPointer(int y);

    void flipRB();
};


template<int CHANNELS, int BITDEPTH, ImageFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage()
{
    this->channels = CHANNELS;
    this->bitDepth = BITDEPTH;
    this->format = FORMAT;
    this->srgb = SRGB;
    this->glType = texel_t::type;
    this->glInternalType = imageFormat_t::type;
    this->bytesPerTexel = sizeof(texel_t);
}

template<int CHANNELS, int BITDEPTH, ImageFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage(int width, int height)
    : TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>()
{
    this->width = width;
    this->height = height;
}

template<int CHANNELS, int BITDEPTH, ImageFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::create(){
    bytesPerRow = width*bytesPerTexel;
    int rowPadding = (rowAlignment - (bytesPerRow % rowAlignment)) % rowAlignment;
    bytesPerRow += rowPadding;
    size = bytesPerRow * height;
    data.resize(size);
}

template<int CHANNELS, int BITDEPTH, ImageFormat FORMAT, bool SRGB>
typename TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::texel_t& TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::getTexel(int x, int y){
    return *(rowPointer(y) + x);
}

template<int CHANNELS, int BITDEPTH, ImageFormat FORMAT, bool SRGB>
typename TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::texel_t* TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::rowPointer(int y){
    return reinterpret_cast<texel_t*>(&data[bytesPerRow*y]);
}

template<int CHANNELS, int BITDEPTH, ImageFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::flipRB(){
    static_assert(CHANNELS>=3,"The image must have atleast 3 channels.");
    for(int y = 0 ; y < height ; ++y){
        for(int x = 0 ; x < height ; ++x){
            texel_t& texel = getTexel(x,y);
            typename texel_t::elementType tmp = texel.r;
            texel.r = texel.b;
            texel.b = tmp;
        }
    }
}
