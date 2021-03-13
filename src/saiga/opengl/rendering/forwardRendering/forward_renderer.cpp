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

void ForwardRenderer::renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera)

{
    if (!rendering) return;

    Resize(viewport.size.x(), viewport.size.y());

    SAIGA_ASSERT(rendering);

    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);


    startTimer(TOTAL);

    if (params.wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(params.wireframeLineSize);
    }

    startTimer(FORWARD);
    camera->recalculatePlanes();
    bindCamera(camera);

    setViewPort(viewport);

    target_framebuffer->bind();
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClearColor(params.clearColor[0], params.clearColor[1], params.clearColor[2], params.clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // depth prepass
    if (depthPrepass)
    {
        renderingInterface->render(camera, RenderPass::DepthPrepass);
        glDepthFunc(GL_EQUAL);
    }
    // forward pass with lighting
    lighting.initRender();
    if (cullLights) lighting.cullLights(camera);
    lighting.cluster(camera, viewport);
    renderingInterface->render(camera, RenderPass::Forward);
    glDepthFunc(GL_LESS);
    lighting.render(camera, viewport);
    stopTimer(FORWARD);

    if (params.wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    stopTimer(TOTAL);

    assert_no_glerror();
}

void ForwardRenderer::Resize(int windowWidth, int windowHeight)
{
    if (windowWidth == renderWidth && windowHeight == renderHeight)
    {
        // Already at correct size
        // -> Skip resize
        return;
    }
    lighting.resize(windowWidth, windowHeight);
}

void ForwardRenderer::renderImgui()
{
    lighting.renderImGui();

    if (!should_render_imgui) return;

    int w = 340;
    int h = 240;
    if (!editor_gui.enabled)
    {
        ImGui::SetNextWindowPos(ImVec2(340, outputHeight - h), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
    }

    ImGui::Begin("Forward Renderer", &should_render_imgui);
    ImGui::Checkbox("wireframe", &params.wireframe);
    ImGui::Checkbox("Cull Lights", &cullLights);
    ImGui::Checkbox("Depth Prepass", &depthPrepass);

    ImGui::Text("Render Time");
    ImGui::Text("%fms - Forward pass", getBlockTime(FORWARD));
    ImGui::Text("%fms - Final pass", getBlockTime(FINAL));
    ImGui::Text("%fms - Total", getBlockTime(TOTAL));

    ImGui::Separator();

    ImGui::Checkbox("Show Lighting UI", &showLightingImgui);

    ImGui::End();
}


}  // namespace Saiga
