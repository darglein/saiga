/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/texture/CubeTexture.h"

#include "saiga/opengl/error.h"

namespace Saiga
{
void TextureCube::uploadData(GLenum target, const void* data)
{
    //    bind(0);
    bind();
    glTexImage2D(target,
                 0,                                    // level, 0 = base, no minimap,
                 static_cast<GLint>(internal_format),  // internalformat
                 width,                                // width
                 height,                               // height
                 0,                                    // border, always 0 in OpenGL ES
                 color_type,                           // format
                 data_type,                            // type
                 data);
    setDefaultParameters();
    unbind();

    assert_no_glerror();
}



void TextureCube::uploadData(const void* data)
{
    bind();
    for (int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,   // target
                     0,                                    // level, 0 = base, no minimap,
                     static_cast<GLint>(internal_format),  // internalformat
                     width,                                // width
                     height,                               // height
                     0,
                     color_type,  // format
                     data_type,   // type
                     data);
    }
    assert_no_glerror();
    setDefaultParameters();
    unbind();
}


void TextureCube::setDefaultParameters()
{
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_LINEAR));
    glTexParameteri(target, GL_TEXTURE_WRAP_R, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}



bool TextureCube::fromImage(std::vector<Image>& images)
{
    SAIGA_ASSERT(images.size() == 6);

    auto& refImage = images.front();

    setFormat(refImage.type);
    width  = refImage.width;
    height = refImage.height;

    TextureBase::create();

    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_X, images[1].data());
    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_Z, images[0].data());
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_X, images[3].data());
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_Z, images[2].data());

    uploadData(GL_TEXTURE_CUBE_MAP_POSITIVE_Y, images[4].data());
    uploadData(GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, images[5].data());

    assert_no_glerror();
    return true;
}

void TextureCube::create(int width, int height, GLenum color_type, GLenum internal_format, GLenum data_type,
                         const void* data)
{
    TextureBase::create(color_type, internal_format, data_type);
    this->width  = width;
    this->height = height;
    uploadData(data);
}

}  // namespace Saiga
