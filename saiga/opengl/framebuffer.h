#pragma once


#include "saiga/opengl/opengl.h"

#include <iostream>
#include <vector>
#include <memory>

#include "saiga/opengl/texture/texture.h"

class SAIGA_GLOBAL Framebuffer{
protected:
    GLuint id = 0;

    //there can be multiple color buffers, but only 1 depth and stencil buffer
//    std::vector<std::shared_ptr<raw_Texture*>> colorBuffers;
//    std::shared_ptr<raw_Texture*> depthBuffer = nullptr;
//    std::shared_ptr<raw_Texture*> stencilBuffer = nullptr;


        std::vector<raw_Texture*> colorBuffers;
        raw_Texture* depthBuffer = nullptr;
        raw_Texture* stencilBuffer = nullptr;
public:
    Framebuffer();
    ~Framebuffer();
    Framebuffer(Framebuffer const&) = delete;
    Framebuffer& operator=(Framebuffer const&) = delete;

    void attachTexture(raw_Texture* texture, GLenum textTarget=GL_TEXTURE_2D);
    void attachTextureDepth(raw_Texture* texture, GLenum textTarget=GL_TEXTURE_2D);
    void attachTextureStencil(raw_Texture* texture, GLenum textTarget=GL_TEXTURE_2D);
    void attachTextureDepthStencil(raw_Texture* texture, GLenum textTarget=GL_TEXTURE_2D);

    void destroy();
    void create();

    void bind();
    void unbind();
    void check();

    void blitDepth(int otherId);
    void blitColor(int otherId);

    raw_Texture* getTextureStencil(){return this->stencilBuffer;}
    raw_Texture* getTextureDepth(){return this->depthBuffer;}
    raw_Texture* getTextureColor(int id){return this->colorBuffers[id];}
    GLuint getId(){return id;}
    /**
    * Resizes all attached textures.
    */
    void resize(int width , int height);

};

