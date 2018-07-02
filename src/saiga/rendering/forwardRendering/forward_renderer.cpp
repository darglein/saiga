/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/forwardRendering/forward_renderer.h"
#include "saiga/imgui/imgui.h"
#include "saiga/window/window.h"
#include "saiga/camera/camera.h"

namespace Saiga {

Forward_Renderer::Forward_Renderer(OpenGLWindow &window, const ForwardRenderingParameters &params)
    : Renderer(window), params(params)
{
    timer.create();
}

void Forward_Renderer::render(Camera *cam)
{
    if(!rendering)
        return;

    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(cam);

    ForwardRenderingInterface* renderingInterface = dynamic_cast<ForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    glViewport(0,0,outputWidth,outputHeight);

    timer.startTimer();

    if(params.srgbWrites)
    glEnable(GL_FRAMEBUFFER_SRGB);


    cam->recalculatePlanes();
    bindCamera(cam);


    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClearColor(params.clearColor.x,params.clearColor.y,params.clearColor.z,params.clearColor.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    renderingInterface->renderOverlay(cam);


    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    //final render pass
    if(imgui)
    {
        imgui->beginFrame();
    }
    renderingInterface->renderFinal(cam);
    if(imgui)
    {
        imgui->endFrame();
        imgui->render();
    }

    if(params.useGlFinish)
        glFinish();

    timer.stopTimer();

}


}
