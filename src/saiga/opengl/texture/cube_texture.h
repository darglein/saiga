/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/texture/raw_texture.h"

namespace Saiga
{
/*
 *  From Stackoverflow:
 *
 * Cube Maps have been specified to follow the RenderMan specification (for whatever reason),
 *  and RenderMan assumes the images' origin being in the upper left, contrary to the usual
 * OpenGL behaviour of having the image origin in the lower left. That's why things get swapped
 *  in the Y direction. It totally breaks with the usual OpenGL semantics and doesn't make
 * sense at all. But now we're stuck with it.
 *
 * -> Swap Y before creating a cube texture from a image
 */
class SAIGA_GLOBAL TextureCube : public raw_Texture
{
   public:
    TextureCube() : raw_Texture(GL_TEXTURE_CUBE_MAP) {}
    virtual ~TextureCube() {}


    void setDefaultParameters() override;
    void uploadData(const void* data) override;

    void uploadData(GLenum target, const void* data);


    bool fromImage(Image& img);
    bool fromImage(std::vector<Image>& images);
};

}  // namespace Saiga
