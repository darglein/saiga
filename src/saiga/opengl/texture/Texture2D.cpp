/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Texture2D.h"

#include "saiga/opengl/error.h"
#include "saiga/opengl/texture/Texture.h"

namespace Saiga
{
// ===========================


void Texture1D::create(int size, GLenum color_type, GLenum internal_format, GLenum data_type, const void* data)
{
    TextureBase::create(color_type, internal_format, data_type);
    this->size = size;
    bind();
    glTexImage1D(target,                               // target
                 0,                                    // level, 0 = base, no minimap,
                 static_cast<GLint>(internal_format),  // internalformat
                 size,                                 // width
                 0,                                    // border
                 color_type,                           // format
                 data_type,                            // type
                 data);
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    assert_no_glerror();
    unbind();
}

Texture2D::Texture2D(const Image& img, bool srgb, bool flipY, bool integer) : TextureBase(GL_TEXTURE_2D)
{
    setFormat(img.type, srgb, integer);
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
}


void Texture2D::setDefaultParameters()
{
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
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
