/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/forward_renderer.h"
#include "saiga/imgui/imgui.h"
#include "saiga/window/window.h"
#include "saiga/camera/camera.h"

namespace Saiga {

Forward_Renderer::Forward_Renderer(OpenGLWindow &window)
    : Renderer(window)
{
    timer.create();
}

void Forward_Renderer::render_intern(Camera *cam)
{
    if(!rendering)
        return;

    SAIGA_ASSERT(rendering);
    SAIGA_ASSERT(cam);


    timer.startTimer();

    glEnable(GL_FRAMEBUFFER_SRGB);


    cam->recalculatePlanes();
    bindCamera(cam);



    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    rendering->renderOverlay(cam);


    glEnable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    //final render pass
    if(imgui)
    {
        imgui->beginFrame();
    }
    rendering->renderFinal(cam);
    if(imgui)
    {
        imgui->endFrame();
    }

    timer.stopTimer();

}


}
