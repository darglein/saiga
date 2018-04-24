/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/postProcessor.h"
#include "saiga/rendering/lighting/deferred_lighting.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"
#include "saiga/rendering/lighting/ssao.h"
#include "saiga/smaa/SMAA.h"
#include "saiga/rendering/overlay/deferredDebugOverlay.h"

namespace Saiga {

class Program;



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

    /**
     * Mark all pixels rendered in the geometry pass in the stencil buffer. These pixels then will not be affected by directional lighting.
     * This is especially good when alot of pixels do not need to be lit.
     * For example when huge parts of the screeen is covered by the skybox.
     */
    bool maskUsedPixels = true;


    float renderScale = 1.0f; //a render scale of 2 is equivalent to 4xSSAA

    bool useGPUTimers = true; //meassure gpu times of individual passes. This can decrease the overall performance

	bool useGlFinish = false; //adds a 'glfinish' at the end of the rendering. usefull for debugging.

    bool useSSAO = false;

    bool useSMAA = false;
    SMAA::Quality smaaQuality = SMAA::Quality::SMAA_PRESET_HIGH;

    vec4 gbufferClearColor = vec4(0,0,0,0);
    vec4 lightingClearColor = vec4(0,0,0,0);

    int shadowSamples = 16;

    GBufferParameters gbp;
    PostProcessorParameters ppp;
    RenderingParameters(){}
    RenderingParameters(bool srgbWrites,GBufferParameters gbp,PostProcessorParameters ppp):
        srgbWrites(srgbWrites),gbp(gbp),ppp(ppp){}
};


class SAIGA_GLOBAL Deferred_Renderer{
public:
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
private:
    std::vector<FilteredMultiFrameOpenGLTimer> timers;

    void startTimer(DeferredTimings timer){if(params.useGPUTimers || timer==TOTAL)timers[timer].startTimer();}
    void stopTimer(DeferredTimings timer){if(params.useGPUTimers || timer == TOTAL)timers[timer].stopTimer();}

    bool blitLastFramebuffer = true;

     std::shared_ptr<Texture> blackDummyTexture;

    bool renderDDO = false;
    DeferredDebugOverlay ddo;
    UniformBuffer cameraBuffer;
public:

    //for imgui
    bool showLightingImgui = false;

    void bindCamera(Camera* cam);

    float getTime(DeferredTimings timer){ if (!params.useGPUTimers && timer != TOTAL) return 0; return timers[timer].getTimeMS();}
    float getUnsmoothedTimeMS(DeferredTimings timer){ if (!params.useGPUTimers && timer != TOTAL) return 0; return timers[timer].MultiFrameOpenGLTimer::getTimeMS();}

    void printTimings();


    Program* renderer;



    bool wireframe = false;
    float wireframeLineSize = 1;

    bool offsetGeometry = false;
    float offsetFactor = 1.0f, offsetUnits = 1.0f;



    int windowWidth, windowHeight;
    int width,height;

    Camera** currentCamera;

    std::shared_ptr<SSAO> ssao;

    std::shared_ptr<SMAA> smaa;

    std::shared_ptr<MVPTextureShader>  blitDepthShader;

    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;

    GBuffer gbuffer;


    PostProcessor postProcessor;

    RenderingParameters params;


    DeferredLighting lighting;
    Deferred_Renderer(int windowWidth, int windowHeight, RenderingParameters _params);
	Deferred_Renderer& operator=(Deferred_Renderer& l) = delete;
    virtual ~Deferred_Renderer();
    void resize(int windowWidth, int windowHeight);


    void render_intern();
    void renderGBuffer(Camera *cam);
    void renderDepthMaps(); //render the scene from the lights perspective (don't need user camera here)
    void renderLighting(Camera *cam);
    void renderSSAO(Camera *cam);

    void writeGbufferDepthToCurrentFramebuffer();

    void renderImGui(bool* p_open = NULL);

};

}
