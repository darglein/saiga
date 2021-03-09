/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"
#include "saiga/opengl/rendering/deferredRendering/postProcessor.h"
#include "saiga/opengl/rendering/lighting/deferred_lighting.h"
#include "saiga/opengl/rendering/lighting/ssao.h"
#include "saiga/opengl/rendering/overlay/deferredDebugOverlay.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/smaa/SMAA.h"


namespace Saiga
{
struct SAIGA_OPENGL_API DeferredRenderingParameters : public RenderingParameters
{
    /**
     * When true the depth of the gbuffer is blitted to the default framebuffer.
     */
    bool writeDepthToDefaultFramebuffer = false;

    /**
     * When true the depth of the gbuffer is blitted to the default framebuffer.
     */
    bool writeDepthToOverlayBuffer = true;

    /**
     * Mark all pixels rendered in the geometry pass in the stencil buffer. These pixels then will not be affected by
     * directional lighting. This is especially good when alot of pixels do not need to be lit. For example when huge
     * parts of the screeen is covered by the skybox.
     */
    bool maskUsedPixels = true;


    bool useGPUTimers = true;  // meassure gpu times of individual passes. This can decrease the overall performance


    bool useSSAO = false;

    bool useSMAA              = false;
    SMAA::Quality smaaQuality = SMAA::Quality::SMAA_PRESET_HIGH;

    vec4 lightingClearColor = vec4(0, 0, 0, 0);

    int shadowSamples = 4;

    bool offsetGeometry = false;
    float offsetFactor = 1.0f, offsetUnits = 1.0f;
    bool blitLastFramebuffer = true;

    GBufferParameters gbp;
    PostProcessorParameters ppp;

    void fromConfigFile(const std::string& file) {}
};


class SAIGA_OPENGL_API DeferredRenderer : public OpenGLRenderer
{
   public:
    using ParameterType = DeferredRenderingParameters;

    DeferredLighting lighting;
    PostProcessor postProcessor;
    DeferredRenderingParameters params;

    DeferredRenderer(OpenGLWindow& window, DeferredRenderingParameters _params = DeferredRenderingParameters());
    DeferredRenderer& operator=(DeferredRenderer& l) = delete;
    virtual ~DeferredRenderer() {}

    void renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera) override;
    void renderImgui() override;


    enum DeferredTimings
    {
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

    float getTime(DeferredTimings timer)
    {
        if (!params.useGPUTimers && timer != TOTAL) return 0;
        return timers[timer].getTimeMS();
    }
    float getUnsmoothedTimeMS(DeferredTimings timer)
    {
        if (!params.useGPUTimers && timer != TOTAL) return 0;
        return timers[timer].MultiFrameOpenGLTimer::getTimeMS();
    }
    float getTotalRenderTime() override { return getUnsmoothedTimeMS(DeferredRenderer::DeferredTimings::TOTAL); }

    void printTimings() override;
    void Resize(int outputWidth, int outputHeight);

    int getRenderWidth() { return renderWidth; }
    int getRenderHeight() { return renderHeight; }


   public:
    int renderWidth, renderHeight;
    std::shared_ptr<SSAO> ssao;
    std::shared_ptr<SMAA> smaa;
    GBuffer gbuffer;

    std::shared_ptr<MVPTextureShader> blitDepthShader;
    IndexedVertexBuffer<VertexNT, uint32_t> quadMesh;
    std::vector<FilteredMultiFrameOpenGLTimer> timers;
    std::shared_ptr<Texture> blackDummyTexture;
    bool showLightingImgui = false;
    bool renderDDO         = false;
    DeferredDebugOverlay ddo;


    void clearGBuffer();
    void renderGBuffer(const std::pair<Camera*, ViewPort>& camera);
    void renderDepthMaps();
    void renderLighting(const std::pair<Camera*, ViewPort>& camera);
    void renderSSAO(const std::pair<Camera*, ViewPort>& camera);


    TemplatedImage<ucvec4> DownloadRender();

    // Download depth map and convert to float
    // The result should be in the range [0,1]
    TemplatedImage<float> DownloadDepth();


    void writeGbufferDepthToCurrentFramebuffer();

    void startTimer(DeferredTimings timer)
    {
        if (params.useGPUTimers || timer == TOTAL) timers[timer].startTimer();
    }
    void stopTimer(DeferredTimings timer)
    {
        if (params.useGPUTimers || timer == TOTAL) timers[timer].stopTimer();
    }
};

}  // namespace Saiga
