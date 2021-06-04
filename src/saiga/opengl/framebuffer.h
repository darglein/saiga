/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/texture/Texture.h"

#include <memory>
#include <vector>

namespace Saiga
{

class SAIGA_OPENGL_API Framebuffer
{
   protected:
    GLuint id = 0;

    std::vector<std::shared_ptr<Texture>> colorBuffers;
    std::shared_ptr<Texture> depthBuffer   = nullptr;
    std::shared_ptr<Texture> stencilBuffer = nullptr;

   public:
    Framebuffer();
    ~Framebuffer();
    Framebuffer(Framebuffer const&) = delete;
    Framebuffer& operator=(Framebuffer const&) = delete;

    void MakeDefaultFramebuffer();

    void attachTexture(std::shared_ptr<Texture> texture);
    void attachTextureDepth(std::shared_ptr<Texture> texture);
    void attachTextureStencil(std::shared_ptr<Texture> texture);
    void attachTextureDepthStencil(std::shared_ptr<Texture> texture);

    void destroy();
    void create();

    void bind();
    void unbind();
    void check();
    static void bindDefaultFramebuffer() { glBindFramebuffer(GL_FRAMEBUFFER, 0); }


    /**
     * Specifies a list of color buffers to be drawn into.
     * Maps to glDrawBuffers.
     *
     * drawToAll: to all attached color buffers is drawn. The order is given by the attachment order.
     *      So the first attached texture has the ID 0. To map different ids to textures use drawTo()
     *
     * drawToNone: to no color buffer is drawn. only depth + stencil if they are attached.
     *
     * drawTo: to the passed buffer ids is drawn.
     */

    void drawToAll();
    void drawToNone();
    void drawTo(std::vector<int> colorBufferIds);


    void blitDepth(int otherId);
    void blitColor(int otherId);

    std::shared_ptr<Texture> getTextureStencil() { return this->stencilBuffer; }
    std::shared_ptr<Texture> getTextureDepth() { return this->depthBuffer; }
    std::shared_ptr<Texture> getTextureColor(int _id) { return this->colorBuffers[_id]; }
    GLuint getId() { return id; }

    /**
     * Resizes all attached textures.
     */
    void resize(int width, int height);
};

}  // namespace Saiga
