/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/standardForwardRendering/standard_forward_renderer.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/window/OpenGLWindow.h"
#include "saiga/opengl/shader/shaderLoader.h"


namespace Saiga
{

StandardForwardRenderer::StandardForwardRenderer(OpenGLWindow& window, const ParameterType& params)
    : OpenGLRenderer(window), params(params), lighting()
{
    lighting.init(window.getWidth(), window.getHeight());

    int timerCount = StandardForwardTimingBlock::COUNT;
    timers.resize(timerCount);
    for (auto& t : timers)
    {
        t.create();
    }

    std::cout << "Standard Forward Renderer initialized. Render resolution: " << window.getWidth() << "x" << window.getHeight() << std::endl;
}

void StandardForwardRenderer::render(const Saiga::RenderInfo& renderInfo)
{
    if (!rendering) return;

    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(renderInfo);

    auto camera = renderInfo.cameras.front().first;

    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    glViewport(0, 0, outputWidth, outputHeight);

    if (params.srgbWrites) glEnable(GL_FRAMEBUFFER_SRGB);


    startTimer(TOTAL);


    camera->recalculatePlanes();
    bindCamera(camera);


    startTimer(FORWARD);

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
    renderingInterface->render(camera, RenderPass::Forward);
    lighting.endRender();

    stopTimer(FORWARD);


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
    }
    renderingInterface->render(camera, RenderPass::GUI);
    if (imgui)
    {
        imgui->endFrame();
        imgui->render();
    }

    stopTimer(FORWARD);

    if (params.useGlFinish) glFinish();

    stopTimer(TOTAL);
}

void StandardForwardRenderer::resize(int width, int height)
{
    lighting.resize(width, height);
    OpenGLRenderer::resize(width, height);
}


}  // namespace Saiga
