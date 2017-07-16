#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/postProcessor.h"

namespace Saiga {

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

class SAIGA_GLOBAL SMAABlendingWeightCalculationShader : public Shader{
public:
    GLint location_edgeTex, location_areaTex, location_searchTex;
    virtual void checkUniforms();
    virtual void uploadTextures(std::shared_ptr<raw_Texture> edgeTex, std::shared_ptr<raw_Texture> areaTex, std::shared_ptr<raw_Texture> searchTex);
};

class SAIGA_GLOBAL SMAANeighborhoodBlendingShader : public Shader{
public:
    GLint location_colorTex, location_blendTex;
    virtual void checkUniforms();
    virtual void uploadTextures(std::shared_ptr<raw_Texture> colorTex, std::shared_ptr<raw_Texture> blendTex);
};

class SMAA{
public:
    enum class Quality{
        SMAA_PRESET_LOW,          //(%60 of the quality)
        SMAA_PRESET_MEDIUM,       //(%80 of the quality)
        SMAA_PRESET_HIGH,         //(%95 of the quality)
        SMAA_PRESET_ULTRA,        //(%99 of the quality)
    };


    SMAA();
    void init(int w, int h, Quality _quality);
    void loadShader();
    void resize(int w, int h);
    void render(framebuffer_texture_t input, Framebuffer& output);

private:
    //mark pixel in first pass and use it in second pass. The last pass is executed on all pixels.
    framebuffer_texture_t stencilTex;

    //RGBA temporal render targets
    framebuffer_texture_t edgesTex;
    Framebuffer edgesFb;

    framebuffer_texture_t blendTex;
    Framebuffer blendFb;

    //supporting precalculated textures
    framebuffer_texture_t areaTex;
    framebuffer_texture_t searchTex;

    bool shaderLoaded = false;
    std::shared_ptr<PostProcessingShader>  smaaEdgeDetectionShader;
    std::shared_ptr<SMAABlendingWeightCalculationShader>  smaaBlendingWeightCalculationShader;
    std::shared_ptr<SMAANeighborhoodBlendingShader>  smaaNeighborhoodBlendingShader;


    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;
    glm::ivec2 screenSize;

    Quality quality = Quality::SMAA_PRESET_HIGH;
};

}
