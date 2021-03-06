/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "deferred_renderer.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/OpenGLWindow.h"


namespace Saiga
{
DeferredRenderer::DeferredRenderer(OpenGLWindow& window, DeferredRenderingParameters _params)
    : OpenGLRenderer(window),
      lighting(gbuffer),
      params(_params),
      renderWidth(window.getWidth() * _params.renderScale),
      renderHeight(window.getHeight() * _params.renderScale),
      ddo(window.getWidth(), window.getHeight())
{
    if (params.useSMAA)
    {
        smaa = std::make_shared<SMAA>(renderWidth, renderHeight);
        smaa->loadShader(params.smaaQuality);
    }

    {
        // create a 2x2 grayscale black dummy texture
        blackDummyTexture = std::make_shared<Texture>();
        std::vector<int> data(2 * 2, 0);
        blackDummyTexture->create(2, 2, GL_RED, GL_R8, GL_UNSIGNED_BYTE, (GLubyte*)data.data());
    }
    if (params.useSSAO)
    {
        ssao = std::make_shared<SSAO>(renderWidth, renderHeight);
    }
    lighting.ssaoTexture = ssao ? ssao->bluredTexture : blackDummyTexture;

    gbuffer.init(renderWidth, renderHeight, params.gbp);

    lighting.shadowSamples = params.shadowSamples;
    lighting.clearColor    = params.lightingClearColor;
    lighting.init(renderWidth, renderHeight, params.useGPUTimers);
    lighting.loadShaders();



    postProcessor.init(renderWidth, renderHeight, &gbuffer, params.ppp, lighting.lightAccumulationTexture,
                       params.useGPUTimers);


    quadMesh.fromMesh(FullScreenQuad());


    int numTimers = DeferredTimings::COUNT;
    if (!params.useGPUTimers) numTimers = 1;  // still use one rendering timer :)
    timers.resize(numTimers);
    for (auto& t : timers)
    {
        t.create();
    }



    blitDepthShader = shaderLoader.load<MVPTextureShader>("lighting/blitDepth.glsl");

    ddo.setDeferredFramebuffer(&gbuffer, lighting.volumetricLightTexture2);


    std::shared_ptr<PostProcessingShader> pps =
        shaderLoader.load<PostProcessingShader>("post_processing/post_processing.glsl");  // this shader does nothing
    std::vector<std::shared_ptr<PostProcessingShader> > defaultEffects;
    defaultEffects.push_back(pps);
    postProcessor.setPostProcessingEffects(defaultEffects);

    std::cout << "Deferred Renderer initialized. Render resolution: " << renderWidth << "x" << renderHeight
              << std::endl;
}

void DeferredRenderer::resize(int windowWidth, int windowHeight)
{
    if (windowWidth <= 0 || windowHeight <= 0)
    {
        std::cerr << "Warning: The window size must be greater than zero." << std::endl;
        windowWidth  = std::max(windowWidth, 1);
        windowHeight = std::max(windowHeight, 1);
    }
    this->outputWidth  = windowWidth;
    this->outputHeight = windowHeight;
    this->renderWidth  = windowWidth * params.renderScale;
    this->renderHeight = windowHeight * params.renderScale;
    std::cout << "Resizing Window to : " << windowWidth << "," << windowHeight << std::endl;
    std::cout << "Framebuffer size: " << renderWidth << " " << renderHeight << std::endl;
    postProcessor.resize(renderWidth, renderHeight);
    gbuffer.resize(renderWidth, renderHeight);
    lighting.resize(renderWidth, renderHeight);

    if (ssao) ssao->resize(renderWidth, renderHeight);

    if (smaa)
    {
        smaa->resize(renderWidth, renderHeight);
    }
}



void DeferredRenderer::render(const Saiga::RenderInfo& _renderInfo)
{
    if (!rendering) return;


    if (params.useSSAO && !ssao)
    {
        ssao                 = std::make_shared<SSAO>(renderWidth, renderHeight);
        lighting.ssaoTexture = ssao->bluredTexture;
    }



    Saiga::RenderInfo renderInfo = _renderInfo;

    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(renderInfo);

    // if we have multiple cameras defined the user has to specify the viewports of each individual camera
    SAIGA_ASSERT(params.userViewPort || renderInfo.cameras.size() == 1);


    if (renderInfo.cameras.size() == 1)
    {
        renderInfo.cameras.front().second = ViewPort({0, 0}, {renderWidth, renderHeight});
    }


    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);


    startTimer(TOTAL);


    params.maskUsedPixels = true;

    for (auto c : renderInfo.cameras)
    {
        auto camera = c.first;
        camera->recalculatePlanes();
        bindCamera(camera);

        setViewPort(c.second);

        glEnable(GL_SCISSOR_TEST);
        setScissor(c.second);
        clearGBuffer();
        lighting.initRender();
        glDisable(GL_SCISSOR_TEST);


        renderGBuffer(c);
        renderSSAO(c);

        lighting.cullLights(camera);

        renderDepthMaps();


        bindCamera(camera);
        setViewPort(c.second);
        renderLighting(c);
        renderingInterface->render(camera, RenderPass::Forward);
    }
    assert_no_glerror();


    //    return;


    if (params.writeDepthToOverlayBuffer)
    {
        writeGbufferDepthToCurrentFramebuffer();
    }
    else
    {
    }
#if 0

    startTimer(OVERLAY);

    for (auto c : renderInfo.cameras)
    {
        auto camera = c.first;
        bindCamera(camera);
        setViewPort(c.second);
    }
    stopTimer(OVERLAY);
#endif

    glViewport(0, 0, renderWidth, renderHeight);



    lighting.applyVolumetricLightBuffer();

    postProcessor.nextFrame();
    postProcessor.bindCurrentBuffer();
    //    postProcessor.switchBuffer();


    startTimer(POSTPROCESSING);
    // postprocessor's 'currentbuffer' will still be bound after render
    postProcessor.render();
    stopTimer(POSTPROCESSING);


    if (params.useSMAA)
    {
        startTimer(SMAATIME);
        smaa->render(postProcessor.getCurrentTexture(), postProcessor.getTargetBuffer());
        postProcessor.switchBuffer();
        postProcessor.bindCurrentBuffer();
        stopTimer(SMAATIME);
    }

    // write depth to default framebuffer
    if (params.writeDepthToDefaultFramebuffer)
    {
        //        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        postProcessor.bindCurrentBuffer();
        writeGbufferDepthToCurrentFramebuffer();
    }


    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //    glClear(GL_COLOR_BUFFER_BIT);
    startTimer(FINAL);
    glViewport(0, 0, renderWidth, renderHeight);
    if (renderDDO)
    {
        bindCamera(&ddo.layout.cam);
        ddo.render();
    }

    {
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
            renderImgui();
            lighting.renderImGui();
            renderingInterface->render(nullptr, RenderPass::GUI);
            imgui->endFrame();
            imgui->render();
        }
    }
    stopTimer(FINAL);

    glDisable(GL_BLEND);

    if (params.blitLastFramebuffer)
        postProcessor.blitLast(outputWidth, outputHeight);
    else
        postProcessor.renderLast(outputWidth, outputHeight);


    if (params.useGlFinish) glFinish();

    stopTimer(TOTAL);

    assert_no_glerror();
}

void DeferredRenderer::clearGBuffer()
{
    gbuffer.bind();

    // glViewport(0, 0, renderWidth, renderHeight);

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



void DeferredRenderer::renderGBuffer(const std::pair<Saiga::Camera*, Saiga::ViewPort>& camera)
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

void DeferredRenderer::renderDepthMaps()
{
    startTimer(DEPTHMAPS);

    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    lighting.renderDepthMaps(renderingInterface);


    stopTimer(DEPTHMAPS);

    assert_no_glerror();
}

void DeferredRenderer::renderLighting(const std::pair<Saiga::Camera*, Saiga::ViewPort>& camera)
{
    startTimer(LIGHTING);

    lighting.render(camera.first, camera.second);

    stopTimer(LIGHTING);

    assert_no_glerror();
}

void DeferredRenderer::renderSSAO(const std::pair<Saiga::Camera*, Saiga::ViewPort>& camera)
{
    startTimer(SSAOT);

    if (params.useSSAO) ssao->render(camera.first, camera.second, &gbuffer);


    stopTimer(SSAOT);

    assert_no_glerror();
}

void DeferredRenderer::writeGbufferDepthToCurrentFramebuffer()
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



void DeferredRenderer::printTimings()
{
    std::cout << "====================================" << std::endl;
    std::cout << "Geometry pass: " << getTime(GEOMETRYPASS) << "ms" << std::endl;
    std::cout << "SSAO: " << getTime(SSAOT) << "ms" << std::endl;
    std::cout << "Depthmaps: " << getTime(DEPTHMAPS) << "ms" << std::endl;
    std::cout << "Lighting: " << getTime(LIGHTING) << "ms" << std::endl;
    lighting.printTimings();
    //    std::cout<<"Light accumulation: "<<getTime(LIGHTACCUMULATION)<<"ms"<<endl;
    std::cout << "Overlay pass: " << getTime(OVERLAY) << "ms" << std::endl;
    std::cout << "Postprocessing: " << getTime(POSTPROCESSING) << "ms" << std::endl;
    postProcessor.printTimings();
    std::cout << "SMAA: " << getTime(SMAATIME) << "ms" << std::endl;
    std::cout << "Final pass: " << getTime(FINAL) << "ms" << std::endl;
    float total = getTime(TOTAL);
    std::cout << "Total: " << total << "ms (" << 1000 / total << " fps)" << std::endl;
    std::cout << "====================================" << std::endl;
}


void DeferredRenderer::renderImgui()
{
    if (!should_render_imgui) return;
    int w = 340;
    int h = 240;
    ImGui::SetNextWindowPos(ImVec2(340, outputHeight - h), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(w, h), ImGuiCond_FirstUseEver);
    ImGui::Begin("Deferred Renderer", &should_render_imgui);

    ImGui::Checkbox("renderDDO", &renderDDO);
    ImGui::Checkbox("wireframe", &params.wireframe);
    ImGui::Checkbox("offsetGeometry", &params.offsetGeometry);

    ImGui::Text("Render Time");
    ImGui::Text("%fms - Geometry pass", getTime(GEOMETRYPASS));
    ImGui::Text("%fms - SSAO", getTime(SSAOT));
    ImGui::Text("%fms - Depthmaps", getTime(DEPTHMAPS));
    ImGui::Text("%fms - Lighting", getTime(LIGHTING));
    ImGui::Text("%fms - Overlay pass", getTime(OVERLAY));
    ImGui::Text("%fms - Postprocessing", getTime(POSTPROCESSING));
    ImGui::Text("%fms - SMAA", getTime(SMAATIME));
    ImGui::Text("%fms - Final pass", getTime(FINAL));
    ImGui::Text("%fms - Total", getTime(TOTAL));

    ImGui::Separator();

    if (ImGui::Checkbox("SMAA", &params.useSMAA))
    {
        if (params.useSMAA)
        {
            smaa = std::make_shared<SMAA>(renderWidth, renderHeight);
            smaa->loadShader(params.smaaQuality);
        }
        else
        {
            smaa.reset();
        }
    }
    if (smaa)
    {
        smaa->renderImGui();
    }


    if (ImGui::Checkbox("SSAO", &params.useSSAO))
    {
        if (params.useSSAO)
        {
            ssao = std::make_shared<SSAO>(renderWidth, renderHeight);
        }
        else
        {
            ssao.reset();
        }
        lighting.ssaoTexture = ssao ? ssao->bluredTexture : blackDummyTexture;
        ddo.setDeferredFramebuffer(&gbuffer, ssao ? ssao->bluredTexture : blackDummyTexture);
    }
    if (ssao)
    {
        ssao->renderImGui();
    }


    ImGui::Checkbox("showLightingImgui", &showLightingImgui);

    ImGui::End();
}
TemplatedImage<ucvec4> DeferredRenderer::DownloadRender()
{
    auto texture = postProcessor.getCurrentTexture();

    TemplatedImage<ucvec4> result(renderHeight, renderWidth);
    texture->download(result.data());
    return result;
}
TemplatedImage<float> DeferredRenderer::DownloadDepth()
{
    auto texture = gbuffer.getTextureDepth();

    TemplatedImage<uint32_t> raw_depth(renderHeight, renderWidth);
    texture->download(raw_depth.data());

    TemplatedImage<float> result(raw_depth.dimensions());
    for (int i : raw_depth.rowRange())
    {
        for (int j : raw_depth.colRange())
        {
            uint32_t di = raw_depth(i, j);
            // stencil
            di = di >> 8;

            float df     = float(di) / (1 << 24);
            result(i, j) = df;
        }
    }



    return result;
}

}  // namespace Saiga
