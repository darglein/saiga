/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/framebuffer.h"

#include "saiga/opengl/error.h"

namespace Saiga
{
Framebuffer::Framebuffer() {}

Framebuffer::~Framebuffer()
{
    destroy();

    //    if(depthBuffer == stencilBuffer){
    //        delete depthBuffer;
    //    }else{

    //        delete depthBuffer;
    //        delete stencilBuffer;
    //    }
    //    for(framebuffer_texture_t t : colorBuffers){
    //        delete t;
    //    }
}

void Framebuffer::MakeDefaultFramebuffer()
{
    destroy();
    id = 0;
}

void Framebuffer::create()
{
    if (id)
    {
        std::cerr << "Warning Framebuffer already created!" << std::endl;
    }
    glGenFramebuffers(1, &id);
    bind();
    //    glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_WIDTH, 1000);
    //    glFramebufferParameteri(GL_DRAW_FRAMEBUFFER, GL_FRAMEBUFFER_DEFAULT_HEIGHT, 1000);
}

void Framebuffer::destroy()
{
    if (!id) return;
    glDeleteFramebuffers(1, &id);
    id = 0;

    colorBuffers.clear();
    depthBuffer   = nullptr;
    stencilBuffer = nullptr;
}



void Framebuffer::bind()
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);
    assert_no_glerror();
}

void Framebuffer::unbind()
{
    bindDefaultFramebuffer();
}

void Framebuffer::check()
{
    glBindFramebuffer(GL_FRAMEBUFFER, id);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        switch (status)
        {
            case GL_FRAMEBUFFER_COMPLETE:
                std::cerr << ("GL_FRAMEBUFFER_COMPLETE\n") << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
                std::cerr << ("GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER\n") << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                std::cerr << ("GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT\n") << std::endl;
                break;
            case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                std::cerr << ("GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT\n") << std::endl;
                break;
            case GL_FRAMEBUFFER_UNSUPPORTED:
                std::cerr << ("GL_FRAMEBUFFER_UNSUPPORTED\n") << std::endl;
                break;
            default:
                std::cerr << "Unknown issue " << (int)status << std::endl;
                break;
        }

        std::cerr << "Framebuffer error!" << std::endl;
        SAIGA_ASSERT(0);
    }
}

void Framebuffer::drawToAll()
{
    int count = colorBuffers.size();
    if (count == 0)
    {
        drawToNone();
        return;
    }
    std::vector<GLenum> DrawBuffers(count);
    for (int i = 0; i < count; ++i)
    {
        DrawBuffers[i] = GL_COLOR_ATTACHMENT0 + i;
    }
    glDrawBuffers(count, &DrawBuffers[0]);
}

void Framebuffer::drawToNone()
{
    glDrawBuffer(GL_NONE);
}

void Framebuffer::drawTo(std::vector<int> colorBufferIds)
{
    int count = colorBufferIds.size();
    std::vector<GLenum> DrawBuffers(count);
    for (int i = 0; i < count; ++i)
    {
        DrawBuffers[i] = GL_COLOR_ATTACHMENT0 + colorBufferIds[i];
    }
    glDrawBuffers(count, &DrawBuffers[0]);
}



void Framebuffer::attachTexture(std::shared_ptr<Texture> texture)
{
    bind();
    int index = colorBuffers.size();
    colorBuffers.push_back(texture);
    GLenum cid = GL_COLOR_ATTACHMENT0 + index;
    glFramebufferTexture2D(GL_FRAMEBUFFER, cid, texture->getTarget(), texture->getId(), 0);
}

void Framebuffer::attachTextureDepth(std::shared_ptr<Texture> texture)
{
    bind();
    depthBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, texture->getTarget(), texture->getId(), 0);
}

void Framebuffer::attachTextureStencil(std::shared_ptr<Texture> texture)
{
    bind();
    stencilBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, texture->getTarget(), texture->getId(), 0);
}


void Framebuffer::attachTextureDepthStencil(std::shared_ptr<Texture> texture)
{
    bind();
    depthBuffer   = texture;
    stencilBuffer = texture;
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, texture->getTarget(), texture->getId(), 0);
}

void Framebuffer::blitDepth(int otherId)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, otherId);
    glBlitFramebuffer(0, 0, depthBuffer->getWidth(), depthBuffer->getHeight(), 0, 0, depthBuffer->getWidth(),
                      depthBuffer->getHeight(), GL_DEPTH_BUFFER_BIT, GL_NEAREST);
}

void Framebuffer::blitColor(int otherId)
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, otherId);
    glBlitFramebuffer(0, 0, colorBuffers[0]->getWidth(), colorBuffers[0]->getHeight(), 0, 0,
                      colorBuffers[0]->getWidth(), colorBuffers[0]->getHeight(), GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void Framebuffer::resize(int width, int height)
{
    if (depthBuffer) depthBuffer->resize(width, height);
    if (stencilBuffer) stencilBuffer->resize(width, height);
    for (auto t : colorBuffers) t->resize(width, height);
}

}  // namespace Saiga
