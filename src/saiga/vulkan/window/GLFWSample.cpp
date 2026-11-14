/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_GLFW
#    include "saiga/core/imgui/imgui.h"

#    include "GLFWSample.h"

#    if defined(SAIGA_OPENGL_INCLUDED)
#        error OpenGL was included somewhere.
#    endif

namespace Saiga
{
VulkanExampleBase::VulkanExampleBase() : StandaloneWindow("config.ini")
{
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 1, 3), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);
    window->setCamera(&camera);

    // The Vulkan renderer draws ImGui only if this is set (toggled with G).
    renderer->should_render_imgui = true;
}

VulkanExampleBase::~VulkanExampleBase() {}

void VulkanExampleBase::update(float dt)
{
    if (ImGui::captureMouse())
        camera.resetMouse();
    else
        camera.interpolate(dt, 0);
    if (!ImGui::captureKeyboard()) camera.update(dt);
}

void VulkanExampleBase::renderGUI()
{
    window->renderImGui();
}

void VulkanExampleBase::keyPressed(int key, int scancode, int mods)
{
    if (ImGui::captureKeyboard()) return;

    switch (key)
    {
        case GLFW_KEY_ESCAPE:
            window->close();
            break;
        case GLFW_KEY_G:
            renderer->should_render_imgui = !renderer->should_render_imgui;
            break;
        case GLFW_KEY_F12:
            renderer->requestScreenshot("screenshot.png");
            break;
        default:
            break;
    }
}

}  // namespace Saiga
#endif
