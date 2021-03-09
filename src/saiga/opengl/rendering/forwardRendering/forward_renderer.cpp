/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/forwardRendering/forward_renderer.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/assets/asset.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/OpenGLWindow.h"


namespace Saiga
{
ForwardRenderer::ForwardRenderer(OpenGLWindow& window, const ParameterType& params)

    : OpenGLRenderer(window), params(params), lighting()

{
    int timerCount = ForwardTimingBlock::COUNT;
    timers.resize(timerCount);
    for (auto& t : timers)
    {
        t.create();
    }

    lighting.init(window.getWidth(), window.getHeight(), false);
    this->params.maximumNumberOfDirectionalLights = std::max(0, params.maximumNumberOfDirectionalLights);
    this->params.maximumNumberOfPointLights       = std::max(0, params.maximumNumberOfPointLights);
    this->params.maximumNumberOfSpotLights        = std::max(0, params.maximumNumberOfSpotLights);
    lighting.setLightMaxima(params.maximumNumberOfDirectionalLights, params.maximumNumberOfPointLights,
                            params.maximumNumberOfSpotLights);

    std::cout << " Forward Renderer initialized. Render resolution: " << window.getWidth() << "x" << window.getHeight()
              << std::endl;
}

void ForwardRenderer::render(const Saiga::RenderInfo& _renderInfo)

{
    if (!rendering) return;

    Saiga::RenderInfo renderInfo = _renderInfo;

    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(renderInfo);

    // if we have multiple cameras defined the user has to specify the viewports of each individual camera
    SAIGA_ASSERT(params.userViewPort || renderInfo.cameras.size() == 1);


    if (renderInfo.cameras.size() == 1)
    {
        renderInfo.cameras.front().second = ViewPort({0, 0}, {outputWidth, outputHeight});
    }

    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);


    startTimer(TOTAL);


    startTimer(TOTAL);

    if (params.wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(params.wireframeLineSize);
    }
    for (auto c : renderInfo.cameras)
    {
        startTimer(FORWARD);
        auto camera = c.first;
        camera->recalculatePlanes();
        bindCamera(camera);

        setViewPort(c.second);

        Framebuffer::bindDefaultFramebuffer();
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
        glDisable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glClearColor(params.clearColor[0], params.clearColor[1], params.clearColor[2], params.clearColor[3]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

        // forward pass with lighting
        lighting.initRender();
        if (cullLights) lighting.cullLights(camera);
        renderingInterface->render(camera, RenderPass::Forward);
        lighting.render(c.first, c.second);
        stopTimer(FORWARD);
    }
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    startTimer(FINAL);

    Framebuffer::bindDefaultFramebuffer();
    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    // gui render pass
    if (imgui)
    {
        SAIGA_ASSERT(ImGui::GetCurrentContext());
        imgui->beginFrame();
        renderingInterface->render(nullptr, RenderPass::GUI);
        imgui->endFrame();
        imgui->render();
    }

    stopTimer(FINAL);

    glDisable(GL_BLEND);

    if (params.useGlFinish) glFinish();

    stopTimer(TOTAL);

    assert_no_glerror();
}

void ForwardRenderer::resize(int windowWidth, int windowHeight)
{
    OpenGLRenderer::resize(windowWidth, windowHeight);
    lighting.resize(windowWidth, windowHeight);
}

void ForwardRenderer::renderImgui()
{
    ImGui::Begin("Forward Renderer", &should_render_imgui);
    ImGui::Checkbox("wireframe", &params.wireframe);
    ImGui::Checkbox("Cull Lights", &cullLights);

    ImGui::Text("Render Time");
    ImGui::Text("%fms - Forward pass", getBlockTime(FORWARD));
    ImGui::Text("%fms - Final pass", getBlockTime(FINAL));
    ImGui::Text("%fms - Total", getBlockTime(TOTAL));

    ImGui::Separator();

    ImGui::Checkbox("Show Lighting UI", &showLightingImgui);

    ImGui::End();
}


}  // namespace Saiga
