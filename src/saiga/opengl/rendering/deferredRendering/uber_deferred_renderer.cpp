/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/camera/camera.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/uberDeferredRendering.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/OpenGLWindow.h"

namespace Saiga
{
UberDeferredRenderer::UberDeferredRenderer(OpenGLWindow& window, UberDeferredRenderingParameters _params)
    : OpenGLRenderer(window),
      lighting(gbuffer),
      params(_params),
      renderWidth(window.getWidth() * _params.renderScale),
      renderHeight(window.getHeight() * _params.renderScale),
      ddo(window.getWidth(), window.getHeight())
{
    {
        // create a 2x2 grayscale black dummy texture
        blackDummyTexture = std::make_shared<Texture>();
        std::vector<int> data(2 * 2, 0);
        blackDummyTexture->create(2, 2, GL_RED, GL_R8, GL_UNSIGNED_BYTE, (GLubyte*)data.data());
    }

    gbuffer.init(renderWidth, renderHeight, params.gbp);

    lighting.shadowSamples = params.shadowSamples;
    lighting.clearColor    = params.lightingClearColor;
    lighting.init(renderWidth, renderHeight, params.useGPUTimers);
    this->params.maximumNumberOfDirectionalLights = std::max(0, params.maximumNumberOfDirectionalLights);
    this->params.maximumNumberOfPointLights       = std::max(0, params.maximumNumberOfPointLights);
    this->params.maximumNumberOfSpotLights        = std::max(0, params.maximumNumberOfSpotLights);
    lighting.setLightMaxima(params.maximumNumberOfDirectionalLights, params.maximumNumberOfPointLights,
                            params.maximumNumberOfSpotLights);
    lighting.loadShaders();


    quadMesh.fromMesh(FullScreenQuad());

    int numTimers = UberDeferredTimingBlock::COUNT;
    if (!params.useGPUTimers) numTimers = 1;  // still use one rendering timer :)
    timers.resize(numTimers);
    for (auto& t : timers)
    {
        t.create();
    }


    blitDepthShader = shaderLoader.load<MVPTextureShader>("lighting/blitDepth.glsl");

    ddo.setDeferredFramebuffer(&gbuffer, blackDummyTexture);


    std::cout << "Uber Deferred Renderer initialized. Render resolution: " << renderWidth << "x" << renderHeight
              << std::endl;
}

UberDeferredRenderer::~UberDeferredRenderer() {}

void UberDeferredRenderer::resize(int windowWidth, int windowHeight)
{
    OpenGLRenderer::resize(windowWidth, windowHeight);
    this->renderWidth  = windowWidth * params.renderScale;
    this->renderHeight = windowHeight * params.renderScale;
    std::cout << "Resizing Window to : " << windowWidth << "," << windowHeight << std::endl;
    std::cout << "Framebuffer size: " << renderWidth << " " << renderHeight << std::endl;
    gbuffer.resize(renderWidth, renderHeight);
    lighting.resize(renderWidth, renderHeight);
}

void UberDeferredRenderer::render(const Saiga::RenderInfo& _renderInfo)
{
    if (!rendering) return;

    Saiga::RenderInfo renderInfo = _renderInfo;

    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(renderInfo);

    SAIGA_ASSERT(params.userViewPort || renderInfo.cameras.size() == 1);


    if (renderInfo.cameras.size() == 1)
    {
        renderInfo.cameras.front().second = ViewPort({0, 0}, {renderWidth, renderHeight});
    }


    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);


    startTimer(TOTAL);

    clearGBuffer();

    assert_no_glerror();
    lighting.initRender();
    assert_no_glerror();
    for (auto c : renderInfo.cameras)
    {
        auto camera = c.first;
        camera->recalculatePlanes();
        bindCamera(camera);

        setViewPort(c.second);
        renderGBuffer(c);

        if (cullLights) lighting.cullLights(camera);
        // renderDepthMaps();

        renderLighting(c);
    }
    assert_no_glerror();

    Framebuffer::bindDefaultFramebuffer();

    if (params.writeDepthToOverlayBuffer)
    {
        // writeGbufferDepthToCurrentFramebuffer();
    }
    else
    {
        glClear(GL_DEPTH_BUFFER_BIT);
    }

    startTimer(OVERLAY);

    for (auto c : renderInfo.cameras)
    {
        auto camera = c.first;
        bindCamera(camera);
        setViewPort(c.second);
        renderingInterface->render(camera, RenderPass::Forward);
    }
    stopTimer(OVERLAY);

    glViewport(0, 0, renderWidth, renderHeight);

    // write depth to default framebuffer
    if (params.writeDepthToDefaultFramebuffer)
    {
        writeGbufferDepthToCurrentFramebuffer();
    }

    startTimer(FINAL);
    glViewport(0, 0, renderWidth, renderHeight);
    if (renderDDO)
    {
        bindCamera(&ddo.layout.cam);
        ddo.render();
    }

    {
        Framebuffer::bindDefaultFramebuffer();
        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDisable(GL_CULL_FACE);
        glDisable(GL_DEPTH_TEST);
        // final render pass
        if (imgui)
        {
            SAIGA_ASSERT(ImGui::GetCurrentContext());
            imgui->beginFrame();
            lighting.renderImGui();
        }
        renderingInterface->render(nullptr, RenderPass::GUI);
        if (imgui)
        {
            imgui->endFrame();
            imgui->render();
        }
    }
    stopTimer(FINAL);

    glDisable(GL_BLEND);

    if (params.useGlFinish) glFinish();

    stopTimer(TOTAL);

    assert_no_glerror();
}

void UberDeferredRenderer::clearGBuffer()
{
    gbuffer.bind();

    glViewport(0, 0, renderWidth, renderHeight);

    glClearColor(params.clearColor[0], params.clearColor[1], params.clearColor[2], params.clearColor[3]);

    if (params.maskUsedPixels)
    {
        glClearStencil(0xFF);  // sets stencil buffer to 255
    }
    else
    {
        glClearStencil(0x00);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    gbuffer.unbind();
}



void UberDeferredRenderer::renderGBuffer(const std::pair<Saiga::Camera*, Saiga::ViewPort>& camera)
{
    startTimer(GEOMETRYPASS);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    if (params.maskUsedPixels)
    {
        // mark all written pixels with 0 in the stencil buffer
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    }
    else
    {
        glDisable(GL_STENCIL_TEST);
    }

    gbuffer.bind();

    //    setViewPort(camera.second);
    //    glViewport(0, 0, renderWidth, renderHeight);



    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);


    if (params.offsetGeometry)
    {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(params.offsetFactor, params.offsetUnits);
    }

    if (params.wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(params.wireframeLineSize);
    }
    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    renderingInterface->render(camera.first, RenderPass::Deferred);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    if (params.offsetGeometry)
    {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    glDisable(GL_STENCIL_TEST);

    gbuffer.unbind();


    stopTimer(GEOMETRYPASS);

    assert_no_glerror();
}

void UberDeferredRenderer::renderDepthMaps()
{
    startTimer(DEPTHMAPS);

    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    lighting.renderDepthMaps(renderingInterface);


    stopTimer(DEPTHMAPS);

    assert_no_glerror();
}

void UberDeferredRenderer::renderLighting(const std::pair<Saiga::Camera*, Saiga::ViewPort>& camera)
{
    startTimer(LIGHTING);


    Framebuffer::bindDefaultFramebuffer();
    writeGbufferDepthToCurrentFramebuffer();
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glClearColor(params.clearColor[0], params.clearColor[1], params.clearColor[2], params.clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT);

    assert_no_glerror();
    lighting.render(camera.first, camera.second);
    assert_no_glerror();

    stopTimer(LIGHTING);

}

void UberDeferredRenderer::writeGbufferDepthToCurrentFramebuffer()
{
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_ALWAYS);
    blitDepthShader->bind();
    blitDepthShader->uploadTexture(gbuffer.getTextureDepth().get());
    quadMesh.bindAndDraw();
    blitDepthShader->unbind();
    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    assert_no_glerror();
}

// void UberDeferredRenderer::printTimings()
// {
//     std::cout << "====================================" << std::endl;
//     std::cout << "Geometry pass: " << getTime(GEOMETRYPASS) << "ms" << std::endl;
//     std::cout << "Depthmaps: " << getTime(DEPTHMAPS) << "ms" << std::endl;
//     std::cout << "Lighting: " << getTime(LIGHTING) << "ms" << std::endl;
//     // lighting.printTimings();
//     //    std::cout<<"Light accumulation: "<<getTime(LIGHTACCUMULATION)<<"ms"<<endl;
//     std::cout << "Overlay pass: " << getTime(OVERLAY) << "ms" << std::endl;
//     std::cout << "Final pass: " << getTime(FINAL) << "ms" << std::endl;
//     float total = getTime(TOTAL);
//     std::cout << "Total: " << total << "ms (" << 1000 / total << " fps)" << std::endl;
//     std::cout << "====================================" << std::endl;
// }


void UberDeferredRenderer::renderImgui()
{
    int w = 340;
    int h = 240;
    ImGui::SetNextWindowPos(ImVec2(340, outputHeight - h), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
    ImGui::Begin("Deferred Renderer", &should_render_imgui);

    ImGui::Checkbox("renderDDO", &renderDDO);
    ImGui::Checkbox("wireframe", &params.wireframe);
    ImGui::Checkbox("offsetGeometry", &params.offsetGeometry);
    ImGui::Checkbox("Stencil Optimization", &params.maskUsedPixels);
    ImGui::Checkbox("Cull Lights", &cullLights);

    ImGui::Text("Render Time");
    ImGui::Text("%fms - Geometry pass", getTime(GEOMETRYPASS));
    ImGui::Text("%fms - Depthmaps", getTime(DEPTHMAPS));
    ImGui::Text("%fms - Lighting", getTime(LIGHTING));
    ImGui::Text("%fms - Overlay pass", getTime(OVERLAY));
    ImGui::Text("%fms - Final pass", getTime(FINAL));
    ImGui::Text("%fms - Total", getTime(TOTAL));

    ImGui::Separator();



    ImGui::Checkbox("Show Lighting UI", &showLightingImgui);

    ImGui::End();
}

}  // namespace Saiga
