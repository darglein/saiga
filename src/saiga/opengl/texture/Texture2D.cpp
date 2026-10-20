/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Texture2D.h"

#include "saiga/opengl/error.h"
#include "saiga/opengl/glImageFormat.h"
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
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(defaultFilter()));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(defaultFilter()));
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

    auto upload_fn = [this](const void* data) { upload(data); };
    if (flipY)
    {
        std::vector<char> data(img.pitchBytes * img.h);
        for (int i = 0; i < img.h; ++i)
        {
            memcpy(&data[i * img.pitchBytes], img.rowPtr(img.h - i - 1), img.pitchBytes);
        }
        uploadWithPitch(img.w, img.h, img.type, img.pitchBytes, data.data(), upload_fn);
    }
    else
    {
        uploadWithPitch(img.w, img.h, img.type, img.pitchBytes, img.data(), upload_fn);
    }
}


void Texture2D::setDefaultParameters()
{
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(defaultFilter()));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(defaultFilter()));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}


void Texture2D::updateFromImage(const Image& img)
{
    // The upload reads width * height pixels of the texture's format from the image.
    SAIGA_ASSERT(getGlType(img.type) == data_type &&
                     (getGlFormat(img.type) == color_type || getGlFormatInteger(img.type) == color_type),
                 "The image type does not match the texture format");
    width  = img.w;
    height = img.h;
    uploadWithPitch(img.w, img.h, img.type, img.pitchBytes, img.data(), [this](const void* data) { upload(data); });
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
    if (new_texture)
    {
        // Only a new texture gets the default parameters. A resize or update keeps the parameters of the user.
        setDefaultParameters();
        new_texture = false;
    }
    else
    {
        // The mipmaps no longer match level 0: after a resize a mipmap filter would make the texture incomplete, and
        // after an update of the same size it would sample the mipmaps of the old image.
        GLint min_filter = 0;
        glGetTexParameteriv(target, GL_TEXTURE_MIN_FILTER, &min_filter);
        if (min_filter == GL_NEAREST_MIPMAP_NEAREST || min_filter == GL_NEAREST_MIPMAP_LINEAR)
        {
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_NEAREST));
        }
        else if (min_filter == GL_LINEAR_MIPMAP_NEAREST || min_filter == GL_LINEAR_MIPMAP_LINEAR)
        {
            glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
        }
    }
    assert_no_glerror();
    unbind();
}



void Texture2D::resize(int width, int height, void* data)
{
    this->width  = width;
    this->height = height;
    upload(data);
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
