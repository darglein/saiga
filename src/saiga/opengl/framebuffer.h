/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/texture.h"

#include <vector>

namespace Saiga
{
// todo: remove
typedef std::shared_ptr<raw_Texture> framebuffer_texture_t;

class SAIGA_GLOBAL Framebuffer
{
   protected:
    GLuint id = 0;

    std::vector<framebuffer_texture_t> colorBuffers;
    framebuffer_texture_t depthBuffer   = nullptr;
    framebuffer_texture_t stencilBuffer = nullptr;

   public:
    Framebuffer();
    ~Framebuffer();
    Framebuffer(Framebuffer const&) = delete;
    Framebuffer& operator=(Framebuffer const&) = delete;

    void attachTexture(framebuffer_texture_t texture);
    void attachTextureDepth(framebuffer_texture_t texture);
    void attachTextureStencil(framebuffer_texture_t texture);
    void attachTextureDepthStencil(framebuffer_texture_t texture);

    void destroy();
    void create();

    void bind();
    void unbind();
    void check();


    /**
     * Specifies a list of color buffers to be drawn into.
     * Maps to glDrawBuffers.
     *
     * drawToAll: to all attached color buffers is drawn. The order is given by the attachment order.
     *      So the first attached texture has the ID 0. To map different ids to textures use drawTo()
     *
     * drawToNone: to no color buffer is drawn.
     *
     * drawTo: to the passed buffer ids is drawn.
     */

    void drawToAll();
    void drawToNone();
    void drawTo(std::vector<int> colorBufferIds);


    void blitDepth(int otherId);
    void blitColor(int otherId);

    framebuffer_texture_t getTextureStencil() { return this->stencilBuffer; }
    framebuffer_texture_t getTextureDepth() { return this->depthBuffer; }
    framebuffer_texture_t getTextureColor(int _id) { return this->colorBuffers[_id]; }
    GLuint getId() { return id; }

    /**
     * Resizes all attached textures.
     */
    void resize(int width, int height);
};

}  // namespace Saiga
