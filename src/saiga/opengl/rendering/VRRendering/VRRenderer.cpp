/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/config.h"

#ifdef SAIGA_VR
#    include "saiga/core/camera/camera.h"
#    include "saiga/core/geometry/triangle_mesh_generator.h"
#    include "saiga/core/imgui/imgui.h"
#    include "saiga/opengl/window/OpenGLWindow.h"

#    include "VRRenderer.h"

namespace Saiga
{
VRRenderer::VRRenderer(OpenGLWindow& window, const VRRenderingParameters& params)
    : OpenGLRenderer(window), params(params)
{
    timer.create();


    if (!vr.init())
    {
        throw std::runtime_error("Could not initialize OpenVR!");
    }


    int width  = vr.renderWidth();
    int height = vr.renderHeight();


    std::shared_ptr<Texture> depth = std::make_shared<Texture>();
    depth->create(width, height, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT32, GL_UNSIGNED_SHORT);

    auto tex = framebuffer_texture_t(depth);

    for (int i = 0; i < 2; ++i)
    {
        framebuffers[i].create();
        framebuffers[i].attachTextureDepth(tex);
        textures[i] = std::make_shared<Texture>();
        textures[i]->create(width, height, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
        framebuffers[i].attachTexture(framebuffer_texture_t(textures[i]));
        framebuffers[i].drawToAll();
        framebuffers[i].check();
        framebuffers[i].unbind();
    }


    auto qb = TriangleMeshGenerator::createFullScreenQuadMesh();
    quadMesh.fromMesh(*qb);


    framebufferToDebugWindowShader =
        shaderLoader.load<PostProcessingShader>("post_processing/VRToDebugWindow.glsl");
    assert_no_glerror();
}

void VRRenderer::render(const RenderInfo& renderInfo)
{
    if (!rendering) return;


    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(renderInfo);

    auto camera = dynamic_cast<PerspectiveCamera*>(renderInfo.cameras.front().first);
    SAIGA_ASSERT(camera);

    auto [cameraLeft, cameraRight] = vr.getEyeCameras(*camera);

    ForwardRenderingInterface* renderingInterface = dynamic_cast<ForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    glViewport(0, 0, vr.renderWidth(), vr.renderHeight());

    timer.startTimer();

    if (params.srgbWrites) glEnable(GL_FRAMEBUFFER_SRGB);

    renderEye(&cameraLeft, vr::Hmd_Eye::Eye_Left, framebuffers[0]);
    renderEye(&cameraRight, vr::Hmd_Eye::Eye_Right, framebuffers[1]);


    vr.handleInput();
    vr.UpdateHMDMatrixPose();

    vr.submitImage(vr::Hmd_Eye::Eye_Left, textures[0].get());
    vr.submitImage(vr::Hmd_Eye::Eye_Right, textures[1].get());


    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glViewport(0, 0, outputWidth, outputHeight);
    Framebuffer::bindDefaultFramebuffer();

    framebufferToDebugWindowShader->bind();
    //    vec4 screenSize(width, height, 1.0 / width, 1.0 / height);
    framebufferToDebugWindowShader->upload(0, textures[0], 0);
    framebufferToDebugWindowShader->upload(1, textures[1], 1);
    framebufferToDebugWindowShader->upload(2, vec2(outputWidth, outputHeight));

    quadMesh.bindAndDraw();
    framebufferToDebugWindowShader->unbind();



    glEnable(GL_BLEND);


    // final render pass
    if (imgui)
    {
        SAIGA_ASSERT(ImGui::GetCurrentContext());
        imgui->beginFrame();
    }
    renderingInterface->renderFinal(camera);
    if (imgui)
    {
        imgui->endFrame();
        imgui->render();
    }

    if (params.useGlFinish) glFinish();

    timer.stopTimer();
}

void VRRenderer::renderEye(Camera* camera, vr::Hmd_Eye eye, Framebuffer& target)
{
    ForwardRenderingInterface* renderingInterface = dynamic_cast<ForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    camera->recalculatePlanes();
    bindCamera(camera);

    target.bind();

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClearColor(params.clearColor[0], params.clearColor[1], params.clearColor[2], params.clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    renderingInterface->renderOverlay(camera);

    target.unbind();
}


}  // namespace Saiga
#endif
