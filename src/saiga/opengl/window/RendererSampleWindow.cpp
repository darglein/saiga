/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_SDL

#    include "RendererSampleWindow.h"

namespace Saiga
{
RendererSampleWindow::RendererSampleWindow() : StandaloneWindow("config.ini")
{
    // create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 80.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);

    // Set the camera from which view the scene is rendered
    window->setCamera(&camera);
}

void RendererSampleWindow::update(float dt)
{
    // Update the camera position
    if (!ImGui::captureKeyboard()) camera.update(dt);
}

void RendererSampleWindow::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    if (!ImGui::captureMouse()) camera.interpolate(dt, interpolation);
}

void RendererSampleWindow::render(Camera* cam, RenderPass render_pass)
{
    if (render_pass == RenderPass::Forward)
    {
        if (showSkybox) skybox.render(cam);
    }
    else if (render_pass == RenderPass::GUI)
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");
        ImGui::Checkbox("showSkybox", &showSkybox);
        camera.imgui();
        ImGui::End();
    }
}

void RendererSampleWindow::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            window->close();
            break;
        default:
            break;
    }
}

void RendererSampleWindow::keyReleased(SDL_Keysym key) {}

}  // namespace Saiga

#endif
