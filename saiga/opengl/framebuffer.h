#pragma once


#include "saiga/opengl/opengl.h"

#include <iostream>
#include <vector>


#include "saiga/opengl/texture/texture.h"

class SAIGA_GLOBAL Framebuffer{
protected:
    GLuint id = 0;

    //there can be multiple color buffers, but only 1 depth and stencil buffer
    std::vector<Texture*> colorBuffers;
    Texture* depthBuffer = nullptr;
    Texture* stencilBuffer = nullptr;
public:
    Framebuffer();
    ~Framebuffer();
    Framebuffer(Framebuffer const&) = delete;
    Framebuffer& operator=(Framebuffer const&) = delete;

    void attachTexture(Texture* texture);
    void attachTextureDepth(Texture* texture);
    void attachTextureStencil(Texture *texture);
    void attachTextureDepthStencil(Texture* texture);

    void destroy();
    void create();

    void bind();
    void unbind();
    void check();

    void blitDepth(int otherId);
    void blitColor(int otherId);

    Texture* getTextureStencil(){return this->stencilBuffer;}
    Texture* getTextureDepth(){return this->depthBuffer;}
    Texture* getTextureColor(int id){return this->colorBuffers[id];}
    GLuint getId(){return id;}
    /**
    * Resizes all attached textures.
    */
    void resize(int width , int height);

};

