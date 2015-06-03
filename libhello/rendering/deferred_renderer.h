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
    virtual void uploadAdditionalUniforms(){}
};

class SSAOShader : public DeferredShader{
public:
    int location_invProj;
    SSAOShader(const string &multi_file) : DeferredShader(multi_file){}
    virtual void checkUniforms();
    void uploadInvProj(mat4 &mat);
};


class Deferred_Renderer : public Renderer{
    public:
    Camera** currentCamera;

    bool postProcessing = false;
    bool ssao = false;

    PostProcessingShader* postProcessingShader = nullptr;
    SSAOShader* ssaoShader = nullptr;

    IndexedVertexBuffer<VertexNT,GLuint> quadMesh;

    Framebuffer deferred_framebuffer;
    Framebuffer mix_framebuffer;
    Framebuffer postProcess_framebuffer; //unused
    Framebuffer ssao_framebuffer;

    DeferredShader* deferred_shader;

    int width,height;

    DeferredLighting lighting;
    Deferred_Renderer();
    virtual ~Deferred_Renderer();
    void init(DeferredShader* deferred_shader, int w, int h);
    void setDeferredMixer(DeferredShader* deferred_shader);
    void setSize(int width, int height){this->width=width;this->height=height;}

    void toggleSSAO();

    void render_intern(float interpolation = 0.f);
    void renderGBuffer(Camera *cam, float interpolation = 0.f);
    void renderDepthMaps(Camera *cam);
    void renderLighting(Camera *cam);
    void renderSSAO(Camera *cam);
    void postProcess();

    virtual void render(Camera *cam, float interpolation) = 0;
    virtual void renderDepth(Camera *cam) = 0;
    virtual void renderOverlay(Camera *cam, float interpolation) = 0;
    virtual void renderFinal(Camera *cam, float interpolation) = 0; //directly renders to screen (after post processing)

    virtual void cudaPostProcessing() = 0;
    virtual void initCudaPostProcessing(Texture* src, Texture* dest) = 0;
//    virtual void renderLighting() = 0;


    void enablePostProcessing(){postProcessing=true;}
    void disablePostProcessing(){postProcessing=false;}
};


