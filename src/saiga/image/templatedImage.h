/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/opengl/opengl.h"
#include "saiga/image/imageFormat.h"
#include "saiga/image/image.h"
#include "saiga/image/templatedImageTypes.h"
#include "saiga/util/color.h"
#include <vector>

namespace Saiga {

template<int CHANNELS, int BITDEPTH = 8, ImageElementFormat FORMAT = ImageElementFormat::UnsignedNormalized, bool SRGB=false>
class TemplatedImage : public Image{
public:
    using TexelType = Texel<CHANNELS,BITDEPTH,FORMAT>;
    using ImageFormatType = GLImageFormatMap<CHANNELS,BITDEPTH,FORMAT,SRGB>;

    TemplatedImage();
    TemplatedImage(int w, int h, void* data = nullptr);
    TemplatedImage(int w, int h, int p, void* data = nullptr);
    TemplatedImage(const Image& img);


    TexelType& getTexel(int x, int y);
    void setTexel(int x, int y, TexelType t);

    TexelType& operator()(int x, int y) { return getTexel(x,y); }

    //pointer to the beginning of this row
    TexelType* rowPointer(int y);

    void flipRB();


    void toSRGB();
    void toLinearRGB();

    TemplatedImage<CHANNELS,32,ImageElementFormat::FloatingPoint,false> convertToFloatImage();


    template<int OTHER_CHANNELS, int OTHER_BITDEPTH, ImageElementFormat OTHER_FORMAT, bool OTHER_SRGB=false>
    TemplatedImage<OTHER_CHANNELS,OTHER_BITDEPTH,OTHER_FORMAT,OTHER_SRGB> convertImage();
};


template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage()
    : Image(ImageFormat(CHANNELS,BITDEPTH,FORMAT,SRGB))
{
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage(int w, int h, void* data)
    : Image(ImageFormat(CHANNELS,BITDEPTH,FORMAT,SRGB),w,h,data)
{
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage(int w, int h, int p, void* data)
    : Image(ImageFormat(CHANNELS,BITDEPTH,FORMAT,SRGB),w,h,p,data)
{
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TemplatedImage(const Image& img)
    : Image(img)
{
    SAIGA_ASSERT(CHANNELS == img.Format().getChannels());
    SAIGA_ASSERT(BITDEPTH == img.Format().getBitDepth());
    SAIGA_ASSERT(FORMAT == img.Format().getElementFormat());
    SAIGA_ASSERT(SRGB == img.Format().getSrgb());
}


template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
typename TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TexelType& TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::getTexel(int x, int y){
    return *(rowPointer(y) + x);
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::setTexel(int x, int y, TexelType t){
    getTexel(x,y) = t;
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
typename TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::TexelType* TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::rowPointer(int y){
    return reinterpret_cast<TexelType*>(&data[pitch*y]);
}

template<int CHANNELS, int BITDEPTH, ImageElementFormat FORMAT, bool SRGB>
void TemplatedImage<CHANNELS,BITDEPTH,FORMAT,SRGB>::flipRB(){
    static_assert(CHANNELS>=3,"The image must have atleast 3 channels.");
    for(int y = 0 ; y < height ; ++y){
        for(int x = 0 ; x < width ; ++x){
            TexelType& texel = getTexel(x,y);
            typename TexelType::elementType tmp = texel.r;
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

}
