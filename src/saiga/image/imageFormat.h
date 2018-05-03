/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/opengl.h"
#include "saiga/image/glImageFormatMap.h"

namespace Saiga {


enum ImageChannel : int
{
    C1 = 0,C2,C3,C4
};

enum ImageElementType : int
{
    UCHAR = 0, SHORT, INT, FLOAT, DOUBLE, ELEMENT_UNKNOWN
};


static const int ImageElementTypeSize[] =
{
    1,2,4,4,8
};

static const GLenum ImageElementTypeGL[] =
{
    GL_UNSIGNED_BYTE,GL_UNSIGNED_SHORT,GL_UNSIGNED_INT,GL_FLOAT,GL_DOUBLE, GL_INVALID_ENUM
};

enum ImageType : int
{
    UC1 = 0, UC2, UC3, UC4,
    S1, S2, S3, S4,
    I1, I2, I3, I4,
    F1, F2, F3, F4,
    D1, D2, D3, D4,
    TYPE_UNKNOWN
};

static const GLenum ImageTypeInternalGL[] =
{
    GL_R8, GL_RG8, GL_RGB8, GL_RGBA8,
    GL_R16, GL_RG16, GL_RGB16, GL_RGBA16,
    GL_R32UI, GL_RG32UI, GL_RGB32UI, GL_RGBA32UI,
    GL_R32F, GL_RG32F, GL_RGB32F, GL_RGBA32F,
    GL_INVALID_ENUM, GL_INVALID_ENUM, GL_INVALID_ENUM, GL_INVALID_ENUM
};

template<typename T>
struct SAIGA_GLOBAL ImageTypeTemplate{
};

template<> struct ImageTypeTemplate<unsigned char>{const static ImageType type = UC1;};
template<> struct ImageTypeTemplate<cvec2>{const static ImageType type = UC2;};
template<> struct ImageTypeTemplate<cvec3>{const static ImageType type = UC3;};
template<> struct ImageTypeTemplate<cvec4>{const static ImageType type = UC4;};



inline ImageType getType(ImageChannel channels, ImageElementType elementType)
{
    return ImageType(int(elementType) * 4 + int(channels));
}

inline ImageType getType(int channels, ImageElementType elementType)
{
    return ImageType(int(elementType) * 4 + int(channels-1));
}

inline int channels(ImageType type)
{
    return (int(type) % 4) + 1;
}

inline int elementType(ImageType type)
{
    return int(type) / 4;
}

inline int elementSize(ImageType type)
{
    return channels(type) * ImageElementTypeSize[elementType(type)];
}

inline GLenum getGlInternalFormat(ImageType type, bool srgb = false)
{
    GLenum t =ImageTypeInternalGL[type];
    if(srgb)
    {
        //currently there are only 2 srgb formats.
        if(t == GL_RGB8)
            t = GL_SRGB8;
        if(t == GL_RGBA8)
            t = GL_SRGB8_ALPHA8;
    }
    return t;
}

inline GLenum getGlFormat(ImageType type)
{
    static const GLenum formats[] = {GL_RED,GL_RG,GL_RGB,GL_RGBA};
    return formats[channels(type)-1];
}

inline GLenum getGlType(ImageType type)
{
    return ImageElementTypeGL[elementType(type)];
}

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

}
