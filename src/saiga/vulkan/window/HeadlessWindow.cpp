/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "HeadlessWindow.h"

#include "saiga/core/framework/framework.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

namespace Saiga
{
namespace Vulkan
{
HeadlessWindow::HeadlessWindow(WindowParameters _windowParameters) : VulkanWindow(_windowParameters)
{
    Saiga::initSaiga(windowParameters.saigaParameters);
}

HeadlessWindow::~HeadlessWindow() {}

void HeadlessWindow::createSurface(VkInstance instance, VkSurfaceKHR* surface)
{
    *surface = VK_NULL_HANDLE;
}

void HeadlessWindow::update(float dt)
{
    if (updating) updating->update(dt);
}

void HeadlessWindow::renderFrames(int count, float dt)
{
    for (int i = 0; i < count; ++i)
    {
        update(dt);
        render();
    }
}

}  // namespace Vulkan
}  // namespace Saiga
