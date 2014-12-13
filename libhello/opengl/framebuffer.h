#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H


#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>
#include <GL/glu.h>

#include <iostream>

#include "libhello/opengl/texture/texture.h"
#include "libhello/util/error.h"

class Framebuffer{
public:
    GLuint id = 0;

    //there can be multiple color buffers, but only 1 depth and stencil buffer
    std::vector<Texture*> colorBuffers;
    Texture* depthBuffer;
    Texture* stencilBuffer;
public:
    Framebuffer();

    void attachTexture(Texture* texture);
    void attachTextureDepth(Texture* texture);
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

#endif // FRAMEBUFFER_H
