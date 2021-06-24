/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/config.h"

#ifdef SAIGA_VR
#    include "saiga/core/camera/camera.h"
#    include "saiga/core/imgui/imgui.h"
#    include "saiga/core/model/model_from_shape.h"
#    include "saiga/opengl/window/OpenGLWindow.h"

#    include "VRRenderer.h"

namespace Saiga
{
VRRenderer::VRRenderer(OpenGLWindow& window, const VRRenderingParameters& params)
    : DeferredRenderer(window, params), params(params), quadMesh(FullScreenQuad())
{
    vr = std::make_shared<OpenVRWrapper>();

    int width  = vr->renderWidth();
    int height = vr->renderHeight();

    for (int i = 0; i < 2; ++i)
    {
        framebuffers[i].create();
        textures[i] = std::make_shared<Texture>();
        textures[i]->create(width, height, GL_RGBA, GL_RGBA8, GL_UNSIGNED_BYTE);
        framebuffers[i].attachTexture(textures[i]);
        framebuffers[i].drawToAll();
        framebuffers[i].check();
        framebuffers[i].unbind();
    }



    framebufferToDebugWindowShader = shaderLoader.load<PostProcessingShader>("post_processing/VRToDebugWindow.glsl");
    assert_no_glerror();
}

void VRRenderer::render(const RenderInfo& renderInfo)
{
    if (!rendering) return;

    vr->update();


    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(renderInfo);

    auto camera = dynamic_cast<PerspectiveCamera*>(renderInfo.cameras.front().first);
    SAIGA_ASSERT(camera);

    auto [cameraLeft, cameraRight] = vr->getEyeCameras(*camera);

    RenderingInterface* renderingInterface = dynamic_cast<RenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    viewport_offset = ivec2(0, 0);
    viewport_size   = ivec2(vr->renderWidth(), vr->renderHeight());

    ViewPort vp(viewport_offset, viewport_size);
    renderGL(&framebuffers[0], vp, &cameraLeft);
    renderGL(&framebuffers[1], vp, &cameraRight);

    vr->submitImage(vr::Hmd_Eye::Eye_Left, textures[0].get());
    vr->submitImage(vr::Hmd_Eye::Eye_Right, textures[1].get());


    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    glViewport(0, 0, outputWidth, outputHeight);
    Framebuffer::bindDefaultFramebuffer();

    if (framebufferToDebugWindowShader->bind())
    {
        //    vec4 screenSize(width, height, 1.0 / width, 1.0 / height);
        framebufferToDebugWindowShader->upload(0, textures[0], 0);
        framebufferToDebugWindowShader->upload(1, textures[1], 1);
        framebufferToDebugWindowShader->upload(2, vec2(outputWidth, outputHeight));

        quadMesh.BindAndDraw();
        framebufferToDebugWindowShader->unbind();
    }


    PrepareImgui();
    FinalizeImgui();

    // glEnable(GL_BLEND);


    // final render pass
    //if (imgui)
    //{
    //    SAIGA_ASSERT(ImGui::GetCurrentContext());
    //    imgui->beginFrame();
    //    renderingInterface->render(camera, RenderPass::GUI);
    //    imgui->endFrame();
    //    imgui->render();
    //}

    if (params.useGlFinish) glFinish();
}

}  // namespace Saiga
#endif
