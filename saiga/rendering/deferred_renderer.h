#pragma once

#include "saiga/rendering/postProcessor.h"
#include "saiga/rendering/lighting/deferred_lighting.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"

class Program;


class SAIGA_GLOBAL SSAOShader : public DeferredShader{
public:
    GLint location_invProj;
    GLint location_filterRadius,location_distanceThreshold;

    float distanceThreshold = 1.0f;
    vec2 filterRadius = vec2(10.0f) / vec2(1600,900);

    virtual void checkUniforms();
    void uploadInvProj(mat4 &mat);
    void uploadData();
};

struct SAIGA_GLOBAL RenderingParameters{
    /**
     * If srgbWrites is enabled all writes to srgb textures will cause a linear->srgb converesion.
     * Important to note is that writes to the default framebuffer also be converted to srgb.
     * This means if srgbWrites is enabled all shader inputs must be converted to linear rgb.
     * For textures use the srgb flag.
     * For vertex colors and uniforms this conversion must be done manually with Color::srgb2linearrgb()
     *
     * If srgbWrites is disabled the gbuffer and postprocessor are not allowed to have srgb textures.
     *
     * Note: If srgbWrites is enabled, you can still use a non-srgb gbuffer and post processor.
     */
    bool srgbWrites = true;

    /**
     * When true the depth of the gbuffer is blitted to the default framebuffer.
     */
    bool writeDepthToDefaultFramebuffer = false;

    /**
     * When true the depth of the gbuffer is blitted to the default framebuffer.
     */
    bool writeDepthToOverlayBuffer = true;

    GBufferParameters gbp;
    PostProcessorParameters ppp;
    RenderingParameters(){}
    RenderingParameters(bool srgbWrites,GBufferParameters gbp,PostProcessorParameters ppp):
        srgbWrites(srgbWrites),gbp(gbp),ppp(ppp){}
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


    Program* renderer;



    bool wireframe = false;
    float wireframeLineSize = 1;

    bool offsetGeometry = false;
    float offsetFactor = 1.0f, offsetUnits = 1.0f;


    Camera** currentCamera;

    bool ssao = false;

    SSAOShader* ssaoShader = nullptr;
    MVPTextureShader* blitDepthShader;

    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;

    GBuffer deferred_framebuffer;
    Framebuffer ssao_framebuffer;

    PostProcessor postProcessor;

    RenderingParameters params;
    int width,height;

    DeferredLighting lighting;
    Deferred_Renderer(int w, int h, RenderingParameters params);
	Deferred_Renderer& operator=(Deferred_Renderer& l) = delete;
    virtual ~Deferred_Renderer();
    void init( int w, int h);
    void setSize(int width, int height){this->width=width;this->height=height;}
    void resize(int width, int height);

    void toggleSSAO();

    void render_intern();
    void renderGBuffer(Camera *cam);
    void renderDepthMaps(); //render the scene from the lights perspective (don't need user camera here)
    void renderLighting(Camera *cam);
    void renderSSAO(Camera *cam);

    void writeGbufferDepthToCurrentFramebuffer();


};


