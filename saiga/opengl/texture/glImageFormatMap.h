#pragma once

#include "saiga/opengl/opengl.h"

enum class ImageFormat{
    UnsignedNormalized,
    SignedNormalized,
    UnsignedIntegral,
    SignedIntegral,
    FloatingPoint
};


template<int bitDepth, ImageFormat format>
struct GLImageFormatMapElementType{
};

template<> struct GLImageFormatMapElementType<8,ImageFormat::UnsignedNormalized>{using elementType = GLubyte;};
template<> struct GLImageFormatMapElementType<8,ImageFormat::UnsignedIntegral>{using elementType = GLubyte;};
template<> struct GLImageFormatMapElementType<8,ImageFormat::SignedNormalized>{using elementType = GLbyte;};
template<> struct GLImageFormatMapElementType<8,ImageFormat::SignedIntegral>{using elementType = GLbyte;};

template<> struct GLImageFormatMapElementType<16,ImageFormat::UnsignedNormalized>{using elementType = GLushort;};
template<> struct GLImageFormatMapElementType<16,ImageFormat::UnsignedIntegral>{using elementType = GLushort;};
template<> struct GLImageFormatMapElementType<16,ImageFormat::SignedNormalized>{using elementType = GLshort;};
template<> struct GLImageFormatMapElementType<16,ImageFormat::SignedIntegral>{using elementType = GLshort;};

template<> struct GLImageFormatMapElementType<32,ImageFormat::UnsignedNormalized>{using elementType = GLuint;};
template<> struct GLImageFormatMapElementType<32,ImageFormat::UnsignedIntegral>{using elementType = GLuint;};
template<> struct GLImageFormatMapElementType<32,ImageFormat::SignedNormalized>{using elementType = GLint;};
template<> struct GLImageFormatMapElementType<32,ImageFormat::SignedIntegral>{using elementType = GLint;};

template<> struct GLImageFormatMapElementType<32,ImageFormat::FloatingPoint>{using elementType = GLfloat;};

template<int CHANNELS, int bitDepth, ImageFormat format, bool srgb>
struct GLImageFormatMap{
};

template<> struct GLImageFormatMap<1,8,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_R8;};
template<> struct GLImageFormatMap<2,8,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_RG8;};
template<> struct GLImageFormatMap<3,8,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_RGB8;};
template<> struct GLImageFormatMap<4,8,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_RGBA8;};

template<> struct GLImageFormatMap<1,8,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_R8_SNORM;};
template<> struct GLImageFormatMap<2,8,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_RG8_SNORM;};
template<> struct GLImageFormatMap<3,8,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_RGB8_SNORM;};
template<> struct GLImageFormatMap<4,8,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_RGBA8_SNORM;};

template<> struct GLImageFormatMap<1,8,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_R8UI;};
template<> struct GLImageFormatMap<2,8,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RG8UI;};
template<> struct GLImageFormatMap<3,8,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RGB8UI;};
template<> struct GLImageFormatMap<4,8,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RGBA8UI;};

template<> struct GLImageFormatMap<1,8,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_R8I;};
template<> struct GLImageFormatMap<2,8,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RG8I;};
template<> struct GLImageFormatMap<3,8,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RGB8I;};
template<> struct GLImageFormatMap<4,8,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RGBA8I;};

template<> struct GLImageFormatMap<1,16,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_R16;};
template<> struct GLImageFormatMap<2,16,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_RG16;};
template<> struct GLImageFormatMap<3,16,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_RGB16;};
template<> struct GLImageFormatMap<4,16,ImageFormat::UnsignedNormalized,false>{const static GLenum type = GL_RGBA16;};

template<> struct GLImageFormatMap<1,16,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_R16_SNORM;};
template<> struct GLImageFormatMap<2,16,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_RG16_SNORM;};
template<> struct GLImageFormatMap<3,16,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_RGB16_SNORM;};
template<> struct GLImageFormatMap<4,16,ImageFormat::SignedNormalized,false>{const static GLenum type = GL_RGBA16_SNORM;};

template<> struct GLImageFormatMap<1,16,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_R16UI;};
template<> struct GLImageFormatMap<2,16,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RG16UI;};
template<> struct GLImageFormatMap<3,16,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RGB16UI;};
template<> struct GLImageFormatMap<4,16,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RGBA16UI;};

template<> struct GLImageFormatMap<1,16,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_R16I;};
template<> struct GLImageFormatMap<2,16,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RG16I;};
template<> struct GLImageFormatMap<3,16,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RGB16I;};
template<> struct GLImageFormatMap<4,16,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RGBA16I;};

//Note: signed and unsigned normalized 32bit formats do not exist.

template<> struct GLImageFormatMap<1,32,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_R32UI;};
template<> struct GLImageFormatMap<2,32,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RG32UI;};
template<> struct GLImageFormatMap<3,32,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RGB32UI;};
template<> struct GLImageFormatMap<4,32,ImageFormat::UnsignedIntegral,false>{const static GLenum type = GL_RGBA32UI;};

template<> struct GLImageFormatMap<1,32,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_R32I;};
template<> struct GLImageFormatMap<2,32,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RG32I;};
template<> struct GLImageFormatMap<3,32,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RGB32I;};
template<> struct GLImageFormatMap<4,32,ImageFormat::SignedIntegral,false>{const static GLenum type = GL_RGBA32I;};

template<> struct GLImageFormatMap<1,16,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_R16F;};
template<> struct GLImageFormatMap<2,16,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_RG16F;};
template<> struct GLImageFormatMap<3,16,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_RGB16F;};
template<> struct GLImageFormatMap<4,16,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_RGBA16F;};

template<> struct GLImageFormatMap<1,32,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_R32F;};
template<> struct GLImageFormatMap<2,32,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_RG32F;};
template<> struct GLImageFormatMap<3,32,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_RGB32F;};
template<> struct GLImageFormatMap<4,32,ImageFormat::FloatingPoint,false>{const static GLenum type = GL_RGBA32F;};

//srgb formats
template<> struct GLImageFormatMap<3,8,ImageFormat::UnsignedNormalized,true>{const static GLenum type = GL_SRGB8;};
template<> struct GLImageFormatMap<4,8,ImageFormat::UnsignedNormalized,true>{const static GLenum type = GL_SRGB8_ALPHA8;};

