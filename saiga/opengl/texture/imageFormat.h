#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/glImageFormatMap.h"


class SAIGA_GLOBAL ImageFormat{
private:
    int channels;
    int bitDepth;
    ImageElementFormat elementFormat;
    bool srgb;

public:
    //the default format is RGBA 8bit normalized
    ImageFormat(int channels = 4, int bitDepth = 8, ImageElementFormat elementFormat = ImageElementFormat::UnsignedNormalized, bool srgb = false);

    //basic getters and setters
    int getChannels() const;
    void setChannels(int value);
    int getBitDepth() const;
    void setBitDepth(int value);
    ImageElementFormat getElementFormat() const;
    void setElementFormat(const ImageElementFormat &value);
    bool getSrgb() const;
    void setSrgb(bool value);

    int bytesPerChannel();
    int bytesPerPixel();
    int bitsPerPixel();

    //match to the parameters of glTexImage2D(...)
    //https://www.opengl.org/sdk/docs/man/html/glTexImage2D.xhtml
    GLenum getGlInternalFormat() const;
    GLenum getGlFormat() const;
    GLenum getGlType() const;
};

SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, const ImageFormat& f);


template<int CHANNELS, int bitDepth, ImageElementFormat format>
class Texel{
public:
};



template<int bitDepth, ImageElementFormat format>
class Texel<1,bitDepth,format>{
public:
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    const static GLenum type = GL_R;
    union{
        elementType r;
        elementType x;
    };
};

template<int bitDepth, ImageElementFormat format>
class Texel<2,bitDepth,format>{
public:
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    const static GLenum type = GL_RG;
    union{
        elementType r;
        elementType x;
    };
    union{
        elementType g;
        elementType y;
    };
};

template<int bitDepth, ImageElementFormat format>
class Texel<3,bitDepth,format>{
public:
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    const static GLenum type = GL_RGB;
    union{
        elementType r;
        elementType x;
    };
    union{
        elementType g;
        elementType y;
    };
    union{
        elementType b;
        elementType z;
    };
};

template<int bitDepth, ImageElementFormat format>
class Texel<4,bitDepth,format>{
public:
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    const static GLenum type = GL_RGBA;
    union{
        elementType r;
        elementType x;
    };
    union{
        elementType g;
        elementType y;
    };
    union{
        elementType b;
        elementType z;
    };
    union{
        elementType a;
        elementType w;
    };
};
