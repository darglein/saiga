/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "deferred_renderer.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_main_menu.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/imgui/imgui_opengl.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/program.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/OpenGLWindow.h"


namespace Saiga
{
DeferredRenderer::DeferredRenderer(OpenGLWindow& window, DeferredRenderingParameters _params)
    : OpenGLRenderer(window),
      lighting(gbuffer, timer.get()),
      params(_params),
      renderWidth(window.getWidth()),
      renderHeight(window.getHeight()),
      quadMesh(FullScreenQuad())
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

    gbuffer.init(renderWidth, renderHeight, params.srgb);

    lighting.shadowSamples = params.shadowSamples;
    lighting.clearColor    = params.lightingClearColor;
    lighting.init(renderWidth, renderHeight, params.useGPUTimers);
    lighting.loadShaders();



    postProcessor.init(renderWidth, renderHeight, &gbuffer, params.ppp, lighting.lightAccumulationTexture);


    blitDepthShader = shaderLoader.load<MVPTextureShader>("lighting/blitDepth.glsl");

    std::shared_ptr<PostProcessingShader> pps =
        shaderLoader.load<PostProcessingShader>("post_processing/post_processing.glsl");  // this shader does nothing
    std::vector<std::shared_ptr<PostProcessingShader> > defaultEffects;
    defaultEffects.push_back(pps);
    postProcessor.setPostProcessingEffects(defaultEffects);

    console << "Deferred Renderer initialized. Render resolution: " << renderWidth << "x" << renderHeight << std::endl;
}

void DeferredRenderer::Resize(int windowWidth, int windowHeight)
{
    if (windowWidth == renderWidth && windowHeight == renderHeight)
    {
        // Already at correct size
        // -> Skip resize
        return;
    }

    this->renderWidth  = windowWidth;
    this->renderHeight = windowHeight;
    postProcessor.resize(renderWidth, renderHeight);
    gbuffer.resize(renderWidth, renderHeight);
    lighting.resize(renderWidth, renderHeight);

    if (ssao) ssao->resize(renderWidth, renderHeight);

    if (smaa)
    {
        smaa->resize(renderWidth, renderHeight);
    }
    std::cout << "[DeferredRenderer] Resize " << windowWidth << "x" << windowHeight << std::endl;
}



void DeferredRenderer::renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera)
{
    if (!rendering) return;


    Resize(viewport.size.x(), viewport.size.y());


    if (params.useSSAO && !ssao)
    {
        ssao                 = std::make_shared<SSAO>(renderWidth, renderHeight);
        lighting.ssaoTexture = ssao->bluredTexture;
    }

    RenderingInterface* renderingInterface = (RenderingInterface*)rendering;

    params.maskUsedPixels = true;


    //    for (auto c : renderInfo.cameras)
    {
        //        auto camera = c.first;
        auto tim = timer->Measure("Geometry");
        bindCamera(camera);

        setViewPort(viewport);

        glEnable(GL_SCISSOR_TEST);
        setScissor(viewport);
        clearGBuffer();
        lighting.initRender();
        glDisable(GL_SCISSOR_TEST);
        renderGBuffer({camera, viewport});
    }



    if (params.useSSAO) ssao->render(camera, viewport, &gbuffer);



    lighting.ComputeCullingAndStatistics(camera);

    {
        auto tim = timer->Measure("Shadow");
        lighting.renderDepthMaps(camera, renderingInterface);
    }

    {
        auto tim = timer->Measure("Lighting");
        bindCamera(camera);
        setViewPort(viewport);
        lighting.render(camera, viewport);
    }

    {
        auto tim = timer->Measure("Forward");
        renderingInterface->render({camera, RenderPass::Forward});
    }

    {
        auto tim = timer->Measure("Write depth");
        // writeGbufferDepthToCurrentFramebuffer();
    }



    lighting.applyVolumetricLightBuffer();
    setViewPort(viewport);

    if (params.hdr)
    {
        auto tim = timer->Measure("Tone Mapping Linear");
        tone_mapper.MapLinear(lighting.lightAccumulationTexture.get());
    }

    if (params.bloom && params.hdr)
    {
        auto tim = timer->Measure("Bloom");
        bloom.Render(lighting.lightAccumulationTexture.get());
    }

    postProcessor.nextFrame();
    postProcessor.bindCurrentBuffer();



    if (params.hdr)
    {
        auto tim = timer->Measure("Tone Mapping");
        tone_mapper.Map(lighting.lightAccumulationTexture.get(),
                        postProcessor.getTargetBuffer().getTextureColor(0).get());
        postProcessor.switchBuffer();
        postProcessor.bindCurrentBuffer();
    }


    {
        auto tim = timer->Measure("Post Processing");
        postProcessor.render(!params.hdr);
    }


    if (params.useSMAA)
    {
        auto tim = timer->Measure("SMAA");
        smaa->render(postProcessor.getCurrentTexture(), postProcessor.getTargetBuffer());
        postProcessor.switchBuffer();
        postProcessor.bindCurrentBuffer();
    }

    // write depth to default framebuffer
    if (params.writeDepthToDefaultFramebuffer)
    {
        auto tim = timer->Measure("copy depth");
        //        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        postProcessor.bindCurrentBuffer();
        writeGbufferDepthToCurrentFramebuffer();
    }



    {
        auto tim = timer->Measure("Final");
        postProcessor.renderLast(target_framebuffer, viewport);

        target_framebuffer->bind();
        renderingInterface->render({camera, RenderPass::Final});
        target_framebuffer->unbind();

        if (params.useGlFinish)
        {
            glFinish();
        }
    }

    assert_no_glerror();
}

void DeferredRenderer::clearGBuffer()
{
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

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
    glDisable(GL_BLEND);

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


void DeferredRenderer::writeGbufferDepthToCurrentFramebuffer()
{
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_ALWAYS);
    if (blitDepthShader->bind())
    {
        blitDepthShader->uploadTexture(gbuffer.getTextureDepth().get());
        quadMesh.BindAndDraw();
        blitDepthShader->unbind();
    }
    glDepthFunc(GL_LESS);
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

    assert_no_glerror();
}

void DeferredRenderer::renderImgui()
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

    if (ImGui::Begin("Deferred Renderer", &should_render_imgui))
    {
        ImGui::Checkbox("wireframe", &params.wireframe);
        ImGui::Checkbox("offsetGeometry", &params.offsetGeometry);

        ImGui::Separator();


        if (ImGui::Checkbox("srgb", &params.srgb))
        {
            gbuffer.init(renderWidth, renderHeight, params.srgb);
            lighting.lightAccumulationBuffer.attachTextureDepthStencil(gbuffer.getTextureDepth());
        }

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


        ImGui::Separator();
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
        }
        if (ssao)
        {
            ssao->renderImGui();
        }


        ImGui::Checkbox("hdr", &params.hdr);
        if (params.hdr)
        {
            ImGui::Separator();
            tone_mapper.imgui();
        }

        ImGui::Checkbox("bloom", &params.bloom);
        if (params.bloom)
        {
            ImGui::Separator();
            bloom.imgui();
        }
    }
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
