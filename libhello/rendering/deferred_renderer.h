#pragma once

#include "libhello/rendering/renderer.h"
#include "libhello/rendering/lighting/deferred_lighting.h"
#include "libhello/opengl/framebuffer.h"
#include "libhello/opengl/mesh.h"


class PostProcessingShader : public Shader{
public:
    GLuint location_texture, location_screenSize;
    PostProcessingShader(const string &multi_file) : Shader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadTexture(raw_Texture* texture);
    virtual void uploadScreenSize(vec4 size);
};



class Deferred_Renderer : public Renderer{
    public:
    Camera** currentCamera;

    bool postProcessing = false;
    PostProcessingShader* postProcessingShader;
    IndexedVertexBuffer<VertexNT,GLuint> quadMesh;

    Framebuffer deferred_framebuffer;
    Framebuffer mix_framebuffer;
    Framebuffer postProcess_framebuffer;

    DeferredShader* deferred_shader;

    int width,height;

    DeferredLighting lighting;
    Deferred_Renderer():Renderer(),lighting(deferred_framebuffer){}
    void init(DeferredShader* deferred_shader, int w, int h);
    void setDeferredMixer(DeferredShader* deferred_shader);
    void setSize(int width, int height){this->width=width;this->height=height;}

    void render_intern();
    void renderGBuffer(Camera *cam);
    void renderDepthMaps(Camera *cam);
    void renderLighting(Camera *cam);
    void postProcess();

    virtual void render(Camera *cam) = 0;
    virtual void renderDepth(Camera *cam) = 0;
    virtual void renderOverlay(Camera *cam) = 0;

    virtual void cudaPostProcessing() = 0;
    virtual void initCudaPostProcessing(Texture* src, Texture* dest) = 0;
//    virtual void renderLighting() = 0;


    void enablePostProcessing(){postProcessing=true;}
    void disablePostProcessing(){postProcessing=false;}
};


