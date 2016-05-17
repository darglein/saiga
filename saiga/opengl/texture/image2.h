#pragma once


#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/imageFormat.h"
#include "saiga/opengl/texture/image.h"
#include <vector>


template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB=false>
class SAIGA_GLOBAL TemplatedImage : public Image{
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


template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage()
{
    this->format = ImageFormat(CHANNELS,BITDEPTH,FORMAT,SRGB);
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage(int width, int height)
    : TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>()
{
    this->width = width;
    this->height = height;
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::create(){
    bytesPerRow = width*format.bytesPerPixel();
    int rowPadding = (rowAlignment - (bytesPerRow % rowAlignment)) % rowAlignment;
    bytesPerRow += rowPadding;
    size = bytesPerRow * height;
    data.resize(size);
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
typename TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::texel_t& TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::getTexel(int x, int y){
    return *(rowPointer(y) + x);
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::setTexel(int x, int y, texel_t t){
    getTexel(x,y) = t;
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
typename TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::texel_t* TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::rowPointer(int y){
    return reinterpret_cast<texel_t*>(&data[bytesPerRow*y]);
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
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
