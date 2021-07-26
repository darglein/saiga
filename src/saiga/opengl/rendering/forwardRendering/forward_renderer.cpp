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
    : OpenGLRenderer(window), params(params), lighting(timer.get())
{
    lighting.init(window.getWidth(), window.getHeight(), false);

    std::cout << " Forward Renderer initialized. Render resolution: " << window.getWidth() << "x" << window.getHeight()
              << std::endl;
}

void ForwardRenderer::renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera)

{
    if (!rendering) return;

    if (!lightAccumulationTexture)
    {
        lightAccumulationBuffer.create();

        // NOTE: Use the same depth-stencil buffer as the gbuffer. I hope this works on every hardware :).
        lightAccumulationBuffer.attachTextureDepthStencil(target_framebuffer->getTextureDepth());

        lightAccumulationTexture = std::make_shared<Texture>();
        lightAccumulationTexture->create(viewport.size.x(), viewport.size.y(), GL_RGBA, GL_RGBA16F, GL_HALF_FLOAT);
        lightAccumulationBuffer.attachTexture(lightAccumulationTexture);

        lightAccumulationBuffer.drawTo({0});
        lightAccumulationBuffer.check();
        lightAccumulationBuffer.unbind();
    }

    Resize(viewport.size.x(), viewport.size.y());

    SAIGA_ASSERT(rendering);

    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);


    if (params.wireframe)
    {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glLineWidth(params.wireframeLineSize);
    }

    camera->recalculatePlanes();
    bindCamera(camera);

    setViewPort(viewport);

    lightAccumulationBuffer.bind();
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClearColor(params.clearColor[0], params.clearColor[1], params.clearColor[2], params.clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    // forward pass with lighting
    lighting.ComputeCullingAndStatistics(camera);
    lighting.initRender();
    lighting.ComputeCullingAndStatistics(camera);

    lighting.cluster(camera, viewport);
    {
        auto tim = timer->Measure("Forward + Shade");
        renderingInterface->render({camera, RenderPass::Forward});
        glDepthFunc(GL_LESS);
    }
    lighting.render(camera, viewport);
    lightAccumulationBuffer.unbind();

    {
        auto tim = timer->Measure("Tone Mapping");
        tone_mapper.MapLinear(lightAccumulationTexture.get());
        tone_mapper.Map(lightAccumulationTexture.get(), target_framebuffer->getTextureColor(0).get());
    }


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

    this->renderWidth  = windowWidth;
    this->renderHeight = windowHeight;
    std::cout << "Resizing Window to : " << windowWidth << "," << windowHeight << std::endl;
    std::cout << "Framebuffer size: " << renderWidth << " " << renderHeight << std::endl;


    lighting.resize(windowWidth, windowHeight);
    lightAccumulationBuffer.resize(windowWidth, windowHeight);
}

void ForwardRenderer::renderImgui()
{
    lighting.renderImGui();

    if (!should_render_imgui) return;

    if (!editor_gui.enabled)
    {
        int w = 340;
        int h = 240;
        ImGui::SetNextWindowPos(ImVec2(340, outputHeight - h), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
    }

    ImGui::Begin("Forward Renderer", &should_render_imgui);

    ImGui::Checkbox("wireframe", &params.wireframe);
    ImGui::Checkbox("Cull Lights", &cullLights);
    ImGui::Checkbox("Depth Prepass", &depthPrepass);

    ImGui::Separator();

    tone_mapper.imgui();

    ImGui::End();
}


}  // namespace Saiga
