/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "TextureBase.h"

#include "saiga/opengl/error.h"
#include "saiga/opengl/glImageFormat.h"

namespace Saiga
{
TextureBase::~TextureBase()
{
    destroy();
}

void TextureBase::create(GLenum color_type, GLenum internal_format, GLenum data_type)
{
    this->color_type      = color_type;
    this->data_type       = data_type;
    this->internal_format = internal_format;
    create();
}



void TextureBase::create()
{
    destroy();
    /* init_resources */
    glGenTextures(1, &id);
    glBindTexture(target, id);
    glBindTexture(target, 0);
    assert_no_glerror();
}

void TextureBase::destroy()
{
    if (id != 0)
    {
        glDeleteTextures(1, &id);
        id = 0;
    }
}



bool TextureBase::download(void* data)
{
    if (id == 0)
    {
        return false;
    }

    bind();
    glGetTexImage(target, 0, color_type, data_type, data);
    assert_no_glerror();
    unbind();
    return true;
}

void TextureBase::bind()
{
    glBindTexture(target, id);
    assert_no_glerror();
}

void TextureBase::bind(int location)
{
    glActiveTexture(GL_TEXTURE0 + location);
    assert_no_glerror();
    bind();
}


void TextureBase::unbind()
{
    glBindTexture(target, 0);
    assert_no_glerror();
}

void TextureBase::bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access,
                                   GLenum format)
{
    glBindImageTexture(imageUnit, id, level, layered, layer, access, format);
}

void TextureBase::bindImageTexture(GLuint imageUnit, GLint level, GLboolean layered, GLint layer, GLenum access)
{
    bindImageTexture(imageUnit, level, layered, layer, access, internal_format);
}

void TextureBase::bindImageTexture(GLuint imageUnit, GLenum access)
{
    bindImageTexture(imageUnit, 0, GL_FALSE, 0, access);
}



void TextureBase::setWrap(GLenum param)
{
    bind();
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(param));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(param));
    unbind();
    assert_no_glerror();
}
void TextureBase::setFiltering(GLenum param)
{
    bind();
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(param));
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(param));
    unbind();
    assert_no_glerror();
}

void TextureBase::setParameter(GLenum name, GLenum param)
{
    bind();
    glTexParameteri(target, name, static_cast<GLint>(param));
    unbind();
    assert_no_glerror();
}

void TextureBase::generateMipmaps()
{
    bind();
    glGenerateMipmap(target);
    unbind();

    setParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    assert_no_glerror();
}

void TextureBase::setBorderColor(vec4 color)
{
    bind();
    glTexParameterfv(target, GL_TEXTURE_BORDER_COLOR, &color[0]);
    unbind();
    assert_no_glerror();
}


void TextureBase::setFormat(ImageType type, bool srgb, bool integer)
{
    SAIGA_ASSERT(!srgb);
    //    SAIGA_ASSERT(0);
    internal_format = getGlInternalFormat(type, srgb);
    data_type       = getGlType(type);
    if (integer)
    {
        color_type = getGlFormatInteger(type);
    }
    else
    {
        color_type = getGlFormat(type);
    }
}

#if 0
void raw_Texture::setFormat(const ImageFormat &format){
    internal_format = format.getGlInternalFormat();
    color_type = format.getGlFormat();
    data_type = format.getGlType();
}

void raw_Texture::setFormat(const Image &image)
{
    setFormat(image.Format());
    width = image.width;
    height = image.height;
}
#endif


}  // namespace Saiga
