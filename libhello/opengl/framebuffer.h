#pragma once


#include "libhello/opengl/opengl.h"

#include <iostream>
#include <vector>


#include "libhello/opengl/texture/texture.h"

class Framebuffer{
public:
    GLuint id = 0;

    //there can be multiple color buffers, but only 1 depth and stencil buffer
    std::vector<Texture*> colorBuffers;
    Texture* depthBuffer = nullptr;
    Texture* stencilBuffer = nullptr;
public:
    Framebuffer();
    ~Framebuffer();
    void attachTexture(Texture* texture);
    void attachTextureDepth(Texture* texture);
    void attachTextureStencil(Texture *texture);
    void attachTextureDepthStencil(Texture* texture);
    void makeToDeferredFramebuffer(int w, int h);

    void destroy();
    void create();

    void bind();
    void unbind();
    void check();

    void blitDepth(int otherId);
    void blitColor(int otherId);

};

