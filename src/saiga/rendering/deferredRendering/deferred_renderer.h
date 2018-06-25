/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/renderer.h"
#include "saiga/rendering/deferredRendering/postProcessor.h"
#include "saiga/rendering/deferredRendering/lighting/deferred_lighting.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/deferredRendering/gbuffer.h"
#include "saiga/rendering/deferredRendering/lighting/ssao.h"
#include "saiga/smaa/SMAA.h"
#include "saiga/rendering/overlay/deferredDebugOverlay.h"


namespace Saiga {



struct SAIGA_GLOBAL DeferredRenderingParameters  : public RenderingParameters{

    /**
     * When true the depth of the gbuffer is blitted to the default framebuffer.
     */
    bool writeDepthToDefaultFramebuffer = false;

    /**
     * When true the depth of the gbuffer is blitted to the default framebuffer.
     */
    bool writeDepthToOverlayBuffer = true;

    /**
     * Mark all pixels rendered in the geometry pass in the stencil buffer. These pixels then will not be affected by directional lighting.
     * This is especially good when alot of pixels do not need to be lit.
     * For example when huge parts of the screeen is covered by the skybox.
     */
    bool maskUsedPixels = true;


    float renderScale = 1.0f; //a render scale of 2 is equivalent to 4xSSAA

    bool useGPUTimers = true; //meassure gpu times of individual passes. This can decrease the overall performance


    bool useSSAO = false;

    bool useSMAA = false;
    SMAA::Quality smaaQuality = SMAA::Quality::SMAA_PRESET_HIGH;

    vec4 lightingClearColor = vec4(0,0,0,0);

    int shadowSamples = 16;

    bool offsetGeometry = false;
    float offsetFactor = 1.0f, offsetUnits = 1.0f;
    bool blitLastFramebuffer = true;

    GBufferParameters gbp;
    PostProcessorParameters ppp;
};


class SAIGA_GLOBAL Deferred_Renderer : public Renderer{
public:

    DeferredLighting lighting;

    Deferred_Renderer(OpenGLWindow& window, DeferredRenderingParameters _params = DeferredRenderingParameters());
    Deferred_Renderer& operator=(Deferred_Renderer& l) = delete;
    virtual ~Deferred_Renderer();

    void render(Camera *cam) override;
    void renderImGui(bool* p_open = NULL);


    enum DeferredTimings{
        TOTAL = 0,
        GEOMETRYPASS,
        SSAOT,
        DEPTHMAPS,
        LIGHTING,
        POSTPROCESSING,
        LIGHTACCUMULATION,
        OVERLAY,
        FINAL,
        SMAATIME,
        COUNT,
    };

    float getTime(DeferredTimings timer){ if (!params.useGPUTimers && timer != TOTAL) return 0; return timers[timer].getTimeMS();}
    float getUnsmoothedTimeMS(DeferredTimings timer){ if (!params.useGPUTimers && timer != TOTAL) return 0; return timers[timer].MultiFrameOpenGLTimer::getTimeMS();}
    float getTotalRenderTime() { return getUnsmoothedTimeMS(Deferred_Renderer::DeferredTimings::TOTAL); }

    void printTimings() override;
    void resize(int outputWidth, int outputHeight) override;

private:
    int renderWidth,renderHeight;
    std::shared_ptr<SSAO> ssao;
    std::shared_ptr<SMAA> smaa;
    GBuffer gbuffer;
    PostProcessor postProcessor;
    DeferredRenderingParameters params;
    std::shared_ptr<MVPTextureShader>  blitDepthShader;
    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;
    std::vector<FilteredMultiFrameOpenGLTimer> timers;
    std::shared_ptr<Texture> blackDummyTexture;
    bool showLightingImgui = false;
    bool renderDDO = false;
    DeferredDebugOverlay ddo;


    void renderGBuffer(Camera *cam);
    void renderDepthMaps(); //render the scene from the lights perspective (don't need user camera here)
    void renderLighting(Camera *cam);
    void renderSSAO(Camera *cam);

    void writeGbufferDepthToCurrentFramebuffer();

    void startTimer(DeferredTimings timer){if(params.useGPUTimers || timer==TOTAL)timers[timer].startTimer();}
    void stopTimer(DeferredTimings timer){if(params.useGPUTimers || timer == TOTAL)timers[timer].stopTimer();}
};

}
