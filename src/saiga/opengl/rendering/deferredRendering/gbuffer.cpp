/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/vertexBuffer.h"

namespace Saiga
{
void GBuffer::init(int w, int h, bool srgb_color)
{
    destroy();
    this->create();

    std::shared_ptr<Texture> color = std::make_shared<Texture>();
    if (srgb_color)
    {
        color->create(w, h, GL_RGBA, GL_SRGB8_ALPHA8, GL_UNSIGNED_BYTE);
    }
    else
    {
        color->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
    }
    attachTexture(color);


    std::shared_ptr<Texture> normal = std::make_shared<Texture>();
    normal->create(w, h, GL_RG, GL_RG8, GL_UNSIGNED_BYTE);
    //            normal->create(w, h, GL_RG, GL_RG16, GL_UNSIGNED_SHORT);
    attachTexture(normal);


    std::shared_ptr<Texture> data = std::make_shared<Texture>();
    data->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
    attachTexture(data);

    std::shared_ptr<Texture> depth_stencil = std::make_shared<Texture>();
    depth_stencil->create(w, h, GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8);
    attachTextureDepthStencil(depth_stencil);

    drawToAll();

    check();
    unbind();
}


}  // namespace Saiga
