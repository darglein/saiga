/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/opengl/UnifiedMeshBuffer.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"
#include "saiga/opengl/rendering/deferredRendering/postProcessor.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/vertex.h"

namespace Saiga
{
/*
 *
 *
 *                           |input|------------------·
 *                              v                     |
 *                    [ SMAAEdgeDetection ]           |
 *                              v                     |
 *                          |edgesTex|                |
 *                              v                     |
 *              [ SMAABlendingWeightCalculation ]     |
 *                              v                     |
 *                          |blendTex|                |
 *                              v                     |
 *                [ SMAANeighborhoodBlending ] <------·
 *                              v
 *                           |output|
 *
 *
 */

class SAIGA_OPENGL_API SMAABlendingWeightCalculationShader : public Shader
{
   public:
    GLint location_edgeTex, location_areaTex, location_searchTex;
    virtual void checkUniforms();
    virtual void uploadTextures(std::shared_ptr<TextureBase> edgeTex, std::shared_ptr<TextureBase> areaTex,
                                std::shared_ptr<TextureBase> searchTex);
};

class SAIGA_OPENGL_API SMAANeighborhoodBlendingShader : public Shader
{
   public:
    GLint location_colorTex, location_blendTex;
    virtual void checkUniforms();
    virtual void uploadTextures(std::shared_ptr<TextureBase> colorTex, std::shared_ptr<TextureBase> blendTex);
};

class SMAA
{
   public:
    enum class Quality : int
    {
        SMAA_PRESET_LOW = 0,  //(%60 of the quality)
        SMAA_PRESET_MEDIUM,   //(%80 of the quality)
        SMAA_PRESET_HIGH,     //(%95 of the quality)
        SMAA_PRESET_ULTRA,    //(%99 of the quality)
    };


    SMAA(int w, int h);
    void loadShader(SMAA::Quality _quality);
    void resize(int w, int h);
    void render(std::shared_ptr<Texture> input, Framebuffer& output);

    void renderImGui();

   protected:
    // mark pixel in first pass and use it in second pass. The last pass is executed on all pixels.
    std::shared_ptr<Texture> stencilTex;

    // RGBA temporal render targets
    std::shared_ptr<Texture> edgesTex;
    Framebuffer edgesFb;

    std::shared_ptr<Texture> blendTex;
    Framebuffer blendFb;

    // supporting precalculated textures
    std::shared_ptr<Texture> areaTex;
    std::shared_ptr<Texture> searchTex;

    bool shaderLoaded = false;
    std::shared_ptr<PostProcessingShader> smaaEdgeDetectionShader;
    std::shared_ptr<SMAABlendingWeightCalculationShader> smaaBlendingWeightCalculationShader;
    std::shared_ptr<SMAANeighborhoodBlendingShader> smaaNeighborhoodBlendingShader;


    UnifiedMeshBuffer quadMesh;
    ivec2 screenSize;

    Quality quality = Quality::SMAA_PRESET_HIGH;
};



}  // namespace Saiga
