/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/quality.h"
#include "saiga/opengl/framebuffer.h"

namespace Saiga
{
class SAIGA_OPENGL_API GBuffer : public Framebuffer
{
   public:
    GBuffer() {}
    void init(int w, int h, bool srgb_color);

    std::shared_ptr<Texture> getTextureColor() { return this->colorBuffers[0]; }
    std::shared_ptr<Texture> getTextureNormal() { return this->colorBuffers[1]; }
    std::shared_ptr<Texture> getTextureMaterial() { return this->colorBuffers[2]; }
};

}  // namespace Saiga
