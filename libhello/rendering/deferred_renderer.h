#pragma once

#include "libhello/rendering/renderer.h"
#include "libhello/rendering/lighting/deferred_lighting.h"
#include "libhello/opengl/framebuffer.h"
#include "libhello/opengl/mesh.h"

class Deferred_Renderer : public Renderer{
public:
    Camera** currentCamera;
    Framebuffer deferred_framebuffer;
    DeferredShader* deferred_shader;

    int width,height;

    DeferredLighting lighting;
    Deferred_Renderer():Renderer(),lighting(deferred_framebuffer){}
    void init(DeferredShader* deferred_shader, int w, int h);
    void setDeferredMixer(DeferredShader* deferred_shader);
    void setSize(int width, int height){this->width=width;this->height=height;}

    void render_intern();
    virtual void render() = 0;
    virtual void renderOverlay() = 0;
    virtual void renderLighting() = 0;
};


