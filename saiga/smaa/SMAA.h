#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/postProcessor.h"

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
    virtual void uploadTextures(raw_Texture* edgeTex, raw_Texture* areaTex, raw_Texture* searchTex);
};

class SAIGA_GLOBAL SMAANeighborhoodBlendingShader : public Shader{
public:
    GLint location_colorTex, location_blendTex;
    virtual void checkUniforms();
    virtual void uploadTextures(raw_Texture* colorTex, raw_Texture* blendTex);
};

class SMAA{
public:

    //RGBA temporal render targets
    Texture* edgesTex;
    Framebuffer edgesFb;

    Texture* blendTex;
    Framebuffer blendFb;

    //supporting precalculated textures
    Texture* areaTex;
    Texture* searchTex;

    PostProcessingShader* smaaEdgeDetectionShader;
    SMAABlendingWeightCalculationShader* smaaBlendingWeightCalculationShader;
    SMAANeighborhoodBlendingShader* smaaNeighborhoodBlendingShader;


    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;
    vec2 screenSize;

    SMAA(int w, int h);
    void render(framebuffer_texture_t input, Framebuffer& output);
};
