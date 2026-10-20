/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "glImageFormat.h"



namespace Saiga
{
static constexpr GLenum ImageElementTypeGL[] = {GL_BYTE,         GL_UNSIGNED_BYTE, GL_SHORT,  GL_UNSIGNED_SHORT, GL_INT,
                                                GL_UNSIGNED_INT, GL_FLOAT,         GL_DOUBLE, GL_INVALID_ENUM};


// Signed types use the SNORM formats: an unsigned normalized format would clamp negative values to 0.
static constexpr GLenum ImageTypeInternalGL[] = {GL_R8_SNORM,     GL_RG8_SNORM,    GL_RGB8_SNORM,   GL_RGBA8_SNORM,
                                                 GL_R8,           GL_RG8,          GL_RGB8,         GL_RGBA8,

                                                 GL_R16_SNORM,    GL_RG16_SNORM,   GL_RGB16_SNORM,  GL_RGBA16_SNORM,
                                                 GL_R16,          GL_RG16,         GL_RGB16,        GL_RGBA16,

                                                 GL_R32I,         GL_RG32I,        GL_RGB32I,       GL_RGBA32I,
                                                 GL_R32UI,        GL_RG32UI,       GL_RGB32UI,      GL_RGBA32UI,

                                                 GL_R32F,         GL_RG32F,        GL_RGB32F,       GL_RGBA32F,
                                                 GL_INVALID_ENUM, GL_INVALID_ENUM, GL_INVALID_ENUM, GL_INVALID_ENUM};

// Unnormalized integer formats of the 8 and 16 bit types
static constexpr GLenum ImageTypeInternalIntegerGL[] = {
    GL_R8I,  GL_RG8I,  GL_RGB8I,  GL_RGBA8I,  GL_R8UI,  GL_RG8UI,  GL_RGB8UI,  GL_RGBA8UI,
    GL_R16I, GL_RG16I, GL_RGB16I, GL_RGBA16I, GL_R16UI, GL_RG16UI, GL_RGB16UI, GL_RGBA16UI};



GLenum getGlInternalFormat(ImageType type, bool srgb, bool integral)
{
    GLenum t = ImageTypeInternalGL[type];
    if (integral && type < static_cast<int>(std::size(ImageTypeInternalIntegerGL)))
    {
        t = ImageTypeInternalIntegerGL[type];
    }
    if (srgb)
    {
        // currently there are only 2 srgb formats.
        if (t == GL_RGB8) t = GL_SRGB8;
        if (t == GL_RGBA8) t = GL_SRGB8_ALPHA8;
    }
    return t;
}

GLenum getGlFormat(ImageType type)
{
    static const GLenum formats[] = {GL_RED, GL_RG, GL_RGB, GL_RGBA};
    return formats[channels(type) - 1];
}

GLenum getGlFormatInteger(ImageType type)
{
    static const GLenum formats[] = {GL_RED_INTEGER, GL_RG_INTEGER, GL_RGB_INTEGER, GL_RGBA_INTEGER};
    return formats[channels(type) - 1];
}

GLenum getGlType(ImageType type)
{
    return ImageElementTypeGL[(int)elementType(type)];
}
std::string BindlessImageTypeName(GLenum gl_type)
{
    switch (gl_type)
    {
        case gl::GLenum::GL_RGBA16F:
            return "rgba16f";
        case gl::GLenum::GL_RGBA32F:
            return "rgba32f";
        default:
            throw std::runtime_error("Unknown gl_type " + std::to_string((int)gl_type));
    }
}

}  // namespace Saiga
