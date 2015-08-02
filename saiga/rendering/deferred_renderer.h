#pragma once

#include "saiga/rendering/postProcessor.h"
#include "saiga/rendering/lighting/deferred_lighting.h"
#include "saiga/opengl/framebuffer.h"


class RendererInterface;


class SAIGA_GLOBAL SSAOShader : public DeferredShader{
public:
    int location_invProj;
    int location_filterRadius,location_distanceThreshold;

    float distanceThreshold = 1.0f;
    vec2 filterRadius = vec2(10.0f) / vec2(1600,900);

    SSAOShader(const std::string &multi_file) : DeferredShader(multi_file){}
    virtual void checkUniforms();
    void uploadInvProj(mat4 &mat);
    void uploadData();
};




class SAIGA_GLOBAL Deferred_Renderer{
public:
    enum DeferredTimings{
        GEOMETRYPASS = 0,
        SSAO,
        DEPTHMAPS,
        LIGHTING,
        POSTPROCESSING,
        LIGHTACCUMULATION,
        OVERLAY,
        FINAL,
        TOTAL,
        COUNT
    };
private:
    std::vector<FilteredGPUTimer> timers;
    bool useTimers = true;

    void startTimer(DeferredTimings timer){if(useTimers)timers[timer].startTimer();}
    void stopTimer(DeferredTimings timer){if(useTimers)timers[timer].stopTimer();}

public:


    float getTime(DeferredTimings timer){return timers[timer].getTimeMS();}
    void printTimings();


    RendererInterface* renderer;



    bool wireframe = false;
    float wireframeLineSize = 1;


    Camera** currentCamera;

    bool ssao = false;

    SSAOShader* ssaoShader = nullptr;

    IndexedVertexBuffer<VertexNT,GLuint> quadMesh;

    Framebuffer deferred_framebuffer;

    Framebuffer ssao_framebuffer;
    PostProcessor postProcessor;

    DeferredShader* deferred_shader;

    int width,height;

    DeferredLighting lighting;
    Deferred_Renderer();
    virtual ~Deferred_Renderer();
    void init(DeferredShader* deferred_shader, int w, int h);
    void setDeferredMixer(DeferredShader* deferred_shader);
    void setSize(int width, int height){this->width=width;this->height=height;}

    void toggleSSAO();

    void render_intern();
    void renderGBuffer(Camera *cam);
    void renderDepthMaps(Camera *cam);
    void renderLighting(Camera *cam);
    void renderSSAO(Camera *cam);

    virtual void cudaPostProcessing() {}
    virtual void initCudaPostProcessing(Texture* src, Texture* dest) {}



};


