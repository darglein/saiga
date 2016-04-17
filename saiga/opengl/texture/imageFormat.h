#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/glImageFormatMap.h"

template<int CHANNELS, int bitDepth, ImageFormat format>
class Texel{
public:
};



template<int bitDepth, ImageFormat format>
class Texel<1,bitDepth,format>{
public:
    typedef typename GLImageFormatMapElementType<bitDepth,format>::elementType elementType;
    const static GLenum type = GL_R;
    union{
        elementType r;
        elementType x;
    };
};

template<int bitDepth, ImageFormat format>
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

template<int bitDepth, ImageFormat format>
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

template<int bitDepth, ImageFormat format>
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
