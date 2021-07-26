/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RendererSampleWindow.h"

#include "saiga/config.h"

namespace Saiga
{
RendererSampleWindow::RendererSampleWindow() : StandaloneWindow("config.ini")
{
    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 140.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);

    // Set the camera from which view the scene is rendered
    window->setCamera(&camera);
}

void RendererSampleWindow::update(float dt)
{
    // Update the camera position
    if (renderer->use_keyboard_input_in_3dview) camera.update(dt);
}

void RendererSampleWindow::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    if (renderer->use_mouse_input_in_3dview) camera.interpolate(dt, interpolation);
}

void RendererSampleWindow::render(RenderInfo render_info)
{
    if (render_info.render_pass == RenderPass::Forward)
    {
        if (showSkybox) skybox.render(render_info.camera);
    }
    else if (render_info.render_pass == RenderPass::GUI)
    {
    }
}

void RendererSampleWindow::keyPressed(int key, int scancode, int mods)
{
    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            window->close();
            break;
        default:
            break;
    }
}

void RendererSampleWindow::keyReleased(int key, int scancode, int mods) {}

}  // namespace Saiga
