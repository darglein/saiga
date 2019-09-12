/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/texture/Texture.h"

#include "saiga/opengl/error.h"
namespace Saiga
{
// ===========================

void Texture2D::setDefaultParameters()
{
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}


bool Texture2D::fromImage(const Image& img, bool srgb, bool flipY)
{
    setFormat(img.type, srgb);
    width  = img.width;
    height = img.height;
    TextureBase::create();

    if (flipY)
    {
        std::vector<char> data(img.pitchBytes * img.h);
        for (int i = 0; i < img.h; ++i)
        {
            memcpy(&data[i * img.pitchBytes], img.rowPtr(img.h - i - 1), img.pitchBytes);
        }
        upload(data.data());
    }
    else
    {
        upload(img.data());
    }

    return true;
}

void Texture2D::updateFromImage(const Image& img)
{
    upload(img.data());
}

void Texture2D::create(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type,
                       const void* data)
{
    TextureBase::create(color_type, internal_format, data_type);
    this->width  = width;
    this->height = height;
    upload(data);
}

void Texture2D::upload(const void* data)
{
    bind();
    glTexImage2D(target,                               // target
                 0,                                    // level, 0 = base, no minimap,
                 static_cast<GLint>(internal_format),  // internalformat
                 width,                                // width
                 height,                               // height
                 0,
                 color_type,  // format
                 data_type,   // type
                 data);
    setDefaultParameters();
    assert_no_glerror();
    unbind();
}



void Texture2D::resize(int width, int height)
{
    this->width  = width;
    this->height = height;
    upload(nullptr);
}
void Texture2D::uploadSubImage(int x, int y, int width, int height, void* data)
{
    bind();
    glTexSubImage2D(target, 0, x, y, width, height, color_type, data_type, data);
    assert_no_glerror();
    unbind();
}

//====================================================================================


void MultisampledTexture2D::setDefaultParameters()
{
    //    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    //    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    //    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    //    glTexParameteri(target, GL_TEXTURE_WRAP_T,static_cast<GLint>( GL_CLAMP_TO_EDGE));
}


void MultisampledTexture2D::uploadData(const void* data)
{
    bind();
    glTexImage2DMultisample(target, samples, internal_format, width, height,
                            GL_TRUE  // fixedsamplelocations
    );
    setDefaultParameters();
    unbind();
}

}  // namespace Saiga
