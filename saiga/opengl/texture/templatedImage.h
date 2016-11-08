#pragma once


#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/imageFormat.h"
#include "saiga/opengl/texture/image.h"
#include "saiga/opengl/texture/templatedImageTypes.h"
#include "saiga/util/color.h"
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


    void toSRGB();
    void toLinearRGB();

    TemplatedImage<CHANNELS,32,ImageElementFormat::FloatingPoint,false> convertToFloatImage();


    template<int OTHER_CHANNELS, int OTHER_BITDEPTH, ImageElementFormat OTHER_FORMAT, bool OTHER_SRGB=false>
    TemplatedImage<OTHER_CHANNELS,OTHER_BITDEPTH,OTHER_FORMAT,OTHER_SRGB> convertImage();
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
    create();
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
        for(int x = 0 ; x < width ; ++x){
            texel_t& texel = getTexel(x,y);
            typename texel_t::elementType tmp = texel.r;
            texel.r = texel.b;
            texel.b = tmp;
        }
    }
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::toSRGB()
{
    for(int y = 0 ; y < height ; ++y){
        for(int x = 0 ; x < width ; ++x){
            auto& texel1 = getTexel(x,y);
            vec4 c = texel1.toVec4();
            c = vec4(Color::linearrgb2srgb(vec3(c)),c.w);
            texel1.fromVec4(c);
        }
    }
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::toLinearRGB()
{
    for(int y = 0 ; y < height ; ++y){
        for(int x = 0 ; x < width ; ++x){
            auto& texel1 = getTexel(x,y);
            vec4 c = texel1.toVec4();
            c = vec4(Color::srgb2linearrgb(vec3(c)),c.w);
            texel1.fromVec4(c);
        }
    }
}


template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS, 32, ImageElementFormat::FloatingPoint, false> TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::convertToFloatImage()
{
    TemplatedImage<CHANNELS, 32, ImageElementFormat::FloatingPoint, false> img(width,height);
    for(int y = 0 ; y < height ; ++y){
        for(int x = 0 ; x < width ; ++x){
            auto& texel1 = getTexel(x,y);
            auto& texel2 = img.getTexel(x,y);
            texel2.fromVec4(texel1.toVec4());
        }
    }
    return img;
}


template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
template<int OTHER_CHANNELS, int OTHER_BITDEPTH, ImageElementFormat OTHER_FORMAT, bool OTHER_SRGB>
TemplatedImage<OTHER_CHANNELS,OTHER_BITDEPTH,OTHER_FORMAT,OTHER_SRGB> TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::convertImage()
{
    TemplatedImage<OTHER_CHANNELS,OTHER_BITDEPTH,OTHER_FORMAT,OTHER_SRGB> img(width,height);
    for(int y = 0 ; y < height ; ++y){
        for(int x = 0 ; x < width ; ++x){
            auto& texel1 = getTexel(x,y);
            auto& texel2 = img.getTexel(x,y);
            texel2.fromVec4(texel1.toVec4());
        }
    }
    return img;
}

