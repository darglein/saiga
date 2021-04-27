/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/texture/TextureBase.h"

#include <vector>

namespace Saiga
{
class SAIGA_OPENGL_API Texture3D : public TextureBase
{
   public:
    int width, height;
    int depth;

    Texture3D(GLenum target = GL_TEXTURE_3D);
    virtual ~Texture3D() {}

    void create(int width, int height, int depth, GLenum color_type, GLenum internal_format, GLenum data_type);
    void uploadSubImage(int x, int y, int z, int width, int height, int depth, void* data);

    void setDefaultParameters();

    bool fromImage(std::vector<Image>& images);
};

}  // namespace Saiga
