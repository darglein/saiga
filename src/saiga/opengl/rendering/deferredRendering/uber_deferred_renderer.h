/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"
#include "saiga/opengl/rendering/lighting/uber_deferred_lighting.h"
#include "saiga/opengl/rendering/renderer.h"


namespace Saiga
{
struct SAIGA_OPENGL_API UberDeferredRenderingParameters : public RenderingParameters
{
    int maximumNumberOfDirectionalLights = 256;
    int maximumNumberOfPointLights       = 256;
    int maximumNumberOfSpotLights        = 256;

    /**
     * When true the depth of the gbuffer is blitted to the default framebuffer.
     */
    bool writeDepthToDefaultFramebuffer = false;

    /**
     * When true the depth of the gbuffer is blitted to the overlay framebuffer.
     */
    bool writeDepthToOverlayBuffer = true;

    /**
     * Mark all pixels rendered in the geometry pass in the stencil buffer. These pixels then will not be affected by
     * lighting. This is especially good when alot of pixels do not need to be lit. For example when huge
     * parts of the screen is covered by the skybox.
     */
    bool maskUsedPixels = true;


    float renderScale = 1.0f;  // a render scale of 2 is equivalent to 4xSSAA

    bool useGPUTimers = true;  // meassure gpu times of individual passes. This can decrease the overall performance


    vec4 lightingClearColor = vec4(0, 0, 0, 0);

    int shadowSamples = 4;

    bool offsetGeometry = false;
    float offsetFactor = 1.0f, offsetUnits = 1.0f;
    bool blitLastFramebuffer = true;

    GBufferParameters gbp;

    void fromConfigFile(const std::string& file) {}
};

class SAIGA_OPENGL_API UberDeferredRenderer : public OpenGLRenderer
{
   public:
    using InterfaceType = RenderingInterface;
    using ParameterType = UberDeferredRenderingParameters;

    UberDeferredLighting lighting;
    UberDeferredRenderingParameters params;

    UberDeferredRenderer(OpenGLWindow& window,
                         UberDeferredRenderingParameters _params = UberDeferredRenderingParameters());
    UberDeferredRenderer& operator=(UberDeferredRenderer& l) = delete;
    virtual ~UberDeferredRenderer();

    void renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera) override;
    void renderImgui() override;



    inline void setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights)
    {
        // TODO Paul: Refactor!
        params.maximumNumberOfDirectionalLights = maxDirectionalLights;
        params.maximumNumberOfPointLights       = maxPointLights;
        params.maximumNumberOfSpotLights        = maxSpotLights;

        params.maximumNumberOfDirectionalLights = std::max(0, params.maximumNumberOfDirectionalLights);
        params.maximumNumberOfPointLights       = std::max(0, params.maximumNumberOfPointLights);
        params.maximumNumberOfSpotLights        = std::max(0, params.maximumNumberOfSpotLights);

        lighting.setLightMaxima(params.maximumNumberOfDirectionalLights, params.maximumNumberOfPointLights,
                                params.maximumNumberOfSpotLights);
    }

    void Resize(int outputWidth, int outputHeight);

    int getRenderWidth() { return renderWidth; }
    int getRenderHeight() { return renderHeight; }

   protected:
    int renderWidth, renderHeight;
    GBuffer gbuffer;

    std::shared_ptr<MVPTextureShader> blitDepthShader;
    UnifiedMeshBuffer quadMesh;
    std::shared_ptr<Texture> blackDummyTexture;
    bool showLightingImgui = false;
    bool cullLights        = false;


    void clearGBuffer();
    void renderGBuffer(const std::pair<Camera*, ViewPort>& camera);
    void renderLighting(const std::pair<Camera*, ViewPort>& camera);

    void writeGbufferDepthToCurrentFramebuffer();
};

}  // namespace Saiga
