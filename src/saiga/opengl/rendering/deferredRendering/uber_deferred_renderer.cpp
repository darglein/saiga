/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
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
      lighting(gbuffer, timer.get()),
      params(_params),
      renderWidth(window.getWidth() * _params.renderScale),
      renderHeight(window.getHeight() * _params.renderScale),
      quadMesh(FullScreenQuad())
{
    {
        // create a 2x2 grayscale black dummy texture
        blackDummyTexture = std::make_shared<Texture>();
        std::vector<int> data(2 * 2, 0);
        blackDummyTexture->create(2, 2, GL_RED, GL_R8, GL_UNSIGNED_BYTE, (GLubyte*)data.data());
    }

    gbuffer.init(renderWidth, renderHeight, false);

    lighting.shadowSamples = params.shadowSamples;
    lighting.clearColor    = params.lightingClearColor;
    lighting.init(renderWidth, renderHeight, params.useGPUTimers);
    lighting.loadShaders();


    blitDepthShader = shaderLoader.load<MVPTextureShader>("lighting/blitDepth.glsl");



    std::cout << "Uber Deferred Renderer initialized. Render resolution: " << renderWidth << "x" << renderHeight
              << std::endl;
}

UberDeferredRenderer::~UberDeferredRenderer() {}

void UberDeferredRenderer::Resize(int windowWidth, int windowHeight)
{
    if (windowWidth == renderWidth && windowHeight == renderHeight)
    {
        // Already at correct size
        // -> Skip resize
        return;
    }

    this->renderWidth  = windowWidth * params.renderScale;
    this->renderHeight = windowHeight * params.renderScale;
    std::cout << "Resizing Window to : " << windowWidth << "," << windowHeight << std::endl;
    std::cout << "Framebuffer size: " << renderWidth << " " << renderHeight << std::endl;
    gbuffer.resize(renderWidth, renderHeight);
    lighting.resize(renderWidth, renderHeight);
}

void UberDeferredRenderer::renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera)
{
    if (!rendering) return;

    Resize(viewport.size.x(), viewport.size.y());

    SAIGA_ASSERT(rendering);



    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);



    {
        auto tim = timer->Measure("Clear");
        clearGBuffer();
    }

    {
        auto tim = timer->Measure("Geometry");
        camera->recalculatePlanes();
        bindCamera(camera);
        setViewPort(viewport);
        renderGBuffer({camera, viewport});
    }

    lighting.ComputeCullingAndStatistics(camera);
    lighting.initRender();
    assert_no_glerror();

    {
        auto tim = timer->Measure("Lighting");
        target_framebuffer->bind();
        renderLighting({camera, viewport});
    }

    assert_no_glerror();


    {
        auto tim = timer->Measure("Forward");
        bindCamera(camera);
        setViewPort(viewport);
        renderingInterface->render({camera, RenderPass::Forward});
    }

    {
        auto tim = timer->Measure("Tone Mapping");
        tone_mapper.MapLinear(lighting.lightAccumulationTexture.get());
        tone_mapper.Map(lighting.lightAccumulationTexture.get(), target_framebuffer->getTextureColor(0).get());
    }


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
    renderingInterface->render({camera.first, RenderPass::Deferred});
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);


    if (params.offsetGeometry)
    {
        glDisable(GL_POLYGON_OFFSET_FILL);
    }

    glDisable(GL_STENCIL_TEST);

    gbuffer.unbind();


    assert_no_glerror();
}

void UberDeferredRenderer::renderLighting(const std::pair<Saiga::Camera*, Saiga::ViewPort>& camera)
{
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
}

void UberDeferredRenderer::writeGbufferDepthToCurrentFramebuffer()
{
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_ALWAYS);
    if(blitDepthShader->bind())
    {
        blitDepthShader->uploadTexture(gbuffer.getTextureDepth().get());
        quadMesh.BindAndDraw();
        blitDepthShader->unbind();
    }
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
    lighting.renderImGui();

    if (!should_render_imgui) return;

    int w = 340;
    int h = 240;
    if (!editor_gui.enabled)
    {
        ImGui::SetNextWindowPos(ImVec2(340, outputHeight - h), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
    }

    ImGui::Begin("Uber Deferred Renderer", &should_render_imgui);

    ImGui::Checkbox("wireframe", &params.wireframe);
    ImGui::Checkbox("offsetGeometry", &params.offsetGeometry);
    ImGui::Checkbox("Stencil Optimization", &params.maskUsedPixels);
    ImGui::Checkbox("Cull Lights", &cullLights);

    ImGui::Separator();

    tone_mapper.imgui();

    ImGui::End();
}

}  // namespace Saiga
