/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/texture/raw_texture.h"
#include <vector>

namespace Saiga {

class SAIGA_GLOBAL Texture3D : public raw_Texture{

public:
    int depth;

    Texture3D(GLenum target=GL_TEXTURE_3D);
    virtual ~Texture3D(){}

    void createEmptyTexture(int width, int height, int depth, GLenum color_type, GLenum internal_format, GLenum data_type);
    void uploadSubImage(int x, int y, int z, int width, int height, int depth, GLubyte *data);

    void setDefaultParameters() override;

    bool fromImage(std::vector<Image> &images);
};

}
