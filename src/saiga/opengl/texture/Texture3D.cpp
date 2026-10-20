/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/texture/Texture3D.h"

#include "saiga/opengl/error.h"

namespace Saiga
{
Texture3D::Texture3D(GLenum target) : TextureBase(target)
{
    SAIGA_ASSERT(target == GL_TEXTURE_3D || target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY);
}

void Texture3D::create(int width, int height, int depth, GLenum color_type, GLenum internal_format, GLenum data_type)
{
    //    std::cout <<"Texture3D::create" << std::endl;
    this->width           = width;
    this->height          = height;
    this->depth           = depth;
    this->color_type      = color_type;
    this->data_type       = data_type;
    this->internal_format = internal_format;

    TextureBase::create();
    bind(0);
    glTexImage3D(target, 0, static_cast<GLint>(internal_format), width, height, depth, 0, color_type, data_type,
                 nullptr);
    setDefaultParameters();
    unbind();

    assert_no_glerror();
}

void Texture3D::setDefaultParameters()
{
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(defaultFilter()));
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(defaultFilter()));
    glTexParameteri(target, GL_TEXTURE_WRAP_R, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(target, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
}

void Texture3D::uploadSubImage(int x, int y, int z, int width, int height, int depth, void* data)
{
    bind();
    glTexSubImage3D(target, 0, x, y, z, width, height, depth, color_type, data_type, data);
    assert_no_glerror();
    unbind();
}


bool Texture3D::fromImage(std::vector<Image>& images)
{
    if (images.empty()) return false;
    // All slices are uploaded with the size and format of the first image
    for (auto& img : images)
    {
        if (img.width != images[0].width || img.height != images[0].height || img.type != images[0].type)
        {
            return false;
        }
    }

    width  = images[0].width;
    height = images[0].height;
    depth  = images.size();
    setFormat(images[0].type);


    TextureBase::create();
    bind(0);
    //    glTexStorage3D(target, 1, internal_format, width, height, depth);
    glTexImage3D(target, 0, static_cast<GLint>(internal_format), width, height, depth, 0, color_type, data_type,
                 nullptr);

    assert_no_glerror();
    for (int i = 0; i < depth; i++)
    {
        Image& img = images[i];
        uploadWithPitch(img.w, img.h, img.type, img.pitchBytes, img.data(),
                        [&](const void* data) { uploadSubImage(0, 0, i, width, height, 1, const_cast<void*>(data)); });
    }


    unbind();
    return true;
}

}  // namespace Saiga
