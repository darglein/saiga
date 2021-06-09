/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/smaa/SMAA.h"

#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/image/imageGenerator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/smaa/AreaTex.h"
#include "saiga/opengl/smaa/SearchTex.h"
#include "saiga/opengl/opengl_helper.h"

namespace Saiga
{
void SMAABlendingWeightCalculationShader::checkUniforms()
{
    Shader::checkUniforms();
    location_edgeTex   = Shader::getUniformLocation("edgeTex");
    location_areaTex   = Shader::getUniformLocation("areaTex");
    location_searchTex = Shader::getUniformLocation("searchTex");
}

void SMAABlendingWeightCalculationShader::uploadTextures(std::shared_ptr<TextureBase> edgeTex,
                                                         std::shared_ptr<TextureBase> areaTex,
                                                         std::shared_ptr<TextureBase> searchTex)
{
    edgeTex->bind(0);
    Shader::upload(location_edgeTex, 0);

    areaTex->bind(1);
    Shader::upload(location_areaTex, 1);

    searchTex->bind(2);
    Shader::upload(location_searchTex, 2);
}

void SMAANeighborhoodBlendingShader::checkUniforms()
{
    Shader::checkUniforms();
    location_colorTex = Shader::getUniformLocation("colorTex");
    location_blendTex = Shader::getUniformLocation("blendTex");
}

void SMAANeighborhoodBlendingShader::uploadTextures(std::shared_ptr<TextureBase> colorTex,
                                                    std::shared_ptr<TextureBase> blendTex)
{
    colorTex->bind(0);
    Shader::upload(location_colorTex, 0);

    blendTex->bind(1);
    Shader::upload(location_blendTex, 1);
}


SMAA::SMAA(int w, int h)
    : quadMesh(FullScreenQuad())
{
    screenSize = ivec2(w, h);
    stencilTex = std::make_shared<Texture>();


    // GL_STENCIL_INDEX may be used for format only if the GL version is 4.4 or higher.
    bool useStencilOnly = hasExtension("GL_ARB_texture_stencil8");
    if (useStencilOnly)
    {
        stencilTex->create(w, h, GL_STENCIL_INDEX, GL_STENCIL_INDEX8, GL_UNSIGNED_BYTE);
    }
    else
    {
        stencilTex->create(w, h, GL_DEPTH_STENCIL, GL_DEPTH24_STENCIL8, GL_UNSIGNED_INT_24_8);
        std::cerr << "Warning: OpenGL extension ARB_texture_stencil8 not found. Fallback to Depth Stencil Texture."
                  << std::endl;
    }

    edgesTex = std::make_shared<Texture>();
    edgesTex->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
    edgesFb.create();
    edgesFb.attachTexture(edgesTex);
    if (useStencilOnly)
        edgesFb.attachTextureStencil(stencilTex);
    else
        edgesFb.attachTextureDepthStencil(stencilTex);
    edgesFb.drawToAll();
    edgesFb.check();
    edgesFb.unbind();

    blendTex = std::make_shared<Texture>();
    blendTex->create(w, h, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
    blendFb.create();
    blendFb.attachTexture(blendTex);
    blendFb.attachTextureStencil(stencilTex);
    blendFb.drawToAll();
    blendFb.check();
    blendFb.unbind();

    areaTex = std::make_shared<Texture>();
    areaTex->create(AREATEX_WIDTH, AREATEX_HEIGHT, GL_RG, GL_RG8, GL_UNSIGNED_BYTE, areaTexBytes);

    searchTex = std::make_shared<Texture>();
    searchTex->create(SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, GL_RED, GL_R8, GL_UNSIGNED_BYTE, searchTexBytes);


}

void SMAA::loadShader(SMAA::Quality _quality)
{
    quality = _quality;
    // example:
    //#define SMAA_RT_METRICS float4(1.0 / 1280.0, 1.0 / 720.0, 1280.0, 720.0)
    vec4 rtMetrics(1.0f / screenSize[0], 1.0f / screenSize[1], screenSize[0], screenSize[1]);
    std::string rtMetricsStr = "#define SMAA_RT_METRICS float4(" + std::to_string(rtMetrics[0]) + "," +
                               std::to_string(rtMetrics[1]) + "," + std::to_string(rtMetrics[2]) + "," +
                               std::to_string(rtMetrics[3]) + ")";

    std::string qualityStr;
    switch (quality)
    {
        case Quality::SMAA_PRESET_LOW:
            qualityStr = "#define SMAA_PRESET_LOW";
            break;
        case Quality::SMAA_PRESET_MEDIUM:
            qualityStr = "#define SMAA_PRESET_MEDIUM";
            break;
        case Quality::SMAA_PRESET_HIGH:
            qualityStr = "#define SMAA_PRESET_HIGH";
            break;
        case Quality::SMAA_PRESET_ULTRA:
            qualityStr = "#define SMAA_PRESET_ULTRA";
            break;
    }

    ShaderPart::ShaderCodeInjections smaaInjection;
    smaaInjection.emplace_back(GL_VERTEX_SHADER, rtMetricsStr, 1);
    smaaInjection.emplace_back(GL_FRAGMENT_SHADER, rtMetricsStr, 1);
    smaaInjection.emplace_back(GL_VERTEX_SHADER, qualityStr, 2);
    smaaInjection.emplace_back(GL_FRAGMENT_SHADER, qualityStr, 2);

    smaaEdgeDetectionShader =
        shaderLoader.load<PostProcessingShader>("post_processing/smaa/SMAAEdgeDetection.glsl", smaaInjection);
    smaaBlendingWeightCalculationShader = shaderLoader.load<SMAABlendingWeightCalculationShader>(
        "post_processing/smaa/SMAABlendingWeightCalculation.glsl", smaaInjection);
    smaaNeighborhoodBlendingShader = shaderLoader.load<SMAANeighborhoodBlendingShader>(
        "post_processing/smaa/SMAANeighborhoodBlending.glsl", smaaInjection);

    shaderLoaded = true;
    assert_no_glerror();
}

void SMAA::resize(int w, int h)
{
    screenSize = make_ivec2((float)w, (float)h);
    edgesFb.resize(w, h);
    blendFb.resize(w, h);
    shaderLoaded = false;
}

void SMAA::render(std::shared_ptr<Texture> input, Framebuffer& output)
{
    if (!shaderLoaded) loadShader(quality);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_STENCIL_TEST);
    glViewport(0, 0, screenSize[0], screenSize[1]);

    // write 1 to stencil if the pixel is not discarded
    glStencilFunc(GL_ALWAYS, 0x1, ~0);
    glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);

    edgesFb.bind();
    glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    if(smaaEdgeDetectionShader->bind())
    {
        smaaEdgeDetectionShader->uploadTexture(input);
        quadMesh.BindAndDraw();
        smaaEdgeDetectionShader->unbind();
    }
    assert_no_glerror();


    // only work on pixels that are marked with 1
    glStencilFunc(GL_EQUAL, 0x1, ~0);
    glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);


    blendFb.bind();
    glClear(GL_COLOR_BUFFER_BIT);
    if(smaaBlendingWeightCalculationShader->bind())
    {
        smaaBlendingWeightCalculationShader->uploadTextures(edgesTex, areaTex, searchTex);
        quadMesh.BindAndDraw();
        smaaBlendingWeightCalculationShader->unbind();
    }
    assert_no_glerror();


    glDisable(GL_STENCIL_TEST);

    output.bind();
    glClear(GL_COLOR_BUFFER_BIT);
    if(smaaNeighborhoodBlendingShader->bind())
    {
        smaaNeighborhoodBlendingShader->uploadTextures(input, blendTex);
        quadMesh.BindAndDraw();
        smaaNeighborhoodBlendingShader->unbind();
    }
    assert_no_glerror();


    glEnable(GL_DEPTH_TEST);
}

void SMAA::renderImGui()
{
    ImGui::PushID("SMAA::renderImGui");
    static const char* items[4] = {"LOW", "MEDIUM", "HIGH", "ULTRA"};
    int currentItem             = (int)quality;

    if (ImGui::Combo("Quality", &currentItem, items, 4))
    {
        quality      = (Quality)currentItem;
        shaderLoaded = false;
    }
    ImGui::PopID();
}

}  // namespace Saiga
