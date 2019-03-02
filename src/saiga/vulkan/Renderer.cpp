/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "Renderer.h"

#include "saiga/vulkan/window/Window.h"

namespace Saiga
{
namespace Vulkan
{
VulkanRenderer::VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters)
    : window(window), vulkanParameters(vulkanParameters)
{
    window.setRenderer(this);


    std::vector<const char*> instanceExtensions = window.getRequiredInstanceExtensions();
    instance.create(instanceExtensions, vulkanParameters.enableValidationLayer);


    window.createSurface(instance, &surface);

    base().setPhysicalDevice(instance.pickPhysicalDevice());

    vulkanParameters.physicalDeviceFeatures.fillModeNonSolid = VK_TRUE;
    vulkanParameters.physicalDeviceFeatures.wideLines        = VK_TRUE;
    base().createLogicalDevice(surface, vulkanParameters, true);

    createSwapChain();
    syncObjects.resize(swapChain.imageCount);
    for (auto& sync : syncObjects)
    {
        sync.create(base().device);
    }
}

VulkanRenderer::~VulkanRenderer()
{
    // Only wait until the queue is done.
    // All vulkan objects are destroyed in their destructor
    waitIdle();
}

void VulkanRenderer::render(Camera*)
{
    if (!valid)
    {
        validCounter--;

        if (validCounter == 0)
        {
            this->resizeSwapChain();
            createDepthBuffer(surfaceWidth, SurfaceHeight);
            createFrameBuffers(swapChain.imageCount, surfaceWidth, SurfaceHeight);
            valid = true;
        }
        else
        {
            return;
        }
    }


    FrameSync& sync = syncObjects[nextSyncObject];
    sync.wait();


    VkResult err = swapChain.acquireNextImage(sync.imageAvailable, &currentBuffer);
    VK_CHECK_RESULT(err);

    if (err == VK_ERROR_OUT_OF_DATE_KHR)
    {
        waitIdle();
        valid        = false;
        validCounter = 3;
        return;
    }


    render2(sync, currentBuffer);


    err = swapChain.queuePresent(base().mainQueue, currentBuffer, sync.renderComplete);

    if (err == VK_ERROR_OUT_OF_DATE_KHR)
    {
        waitIdle();
        valid        = false;
        validCounter = 3;
        return;
    }

    //    VK_CHECK_RESULT(swapChain.queuePresent(graphicsQueue, currentBuffer));
    //    VK_CHECK_RESULT(vkQueueWaitIdle(presentQueue));
    //    presentQueue.waitIdle();

    nextSyncObject = (nextSyncObject + 1) % syncObjects.size();
}

float VulkanRenderer::getTotalRenderTime()
{
    return window.mainLoop.renderCPUTimer.getTimeMS();
}

void VulkanRenderer::initInstanceDevice() {}

void VulkanRenderer::renderImGui(bool* p_open)
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Renderer Info", p_open, ImGuiWindowFlags_NoCollapse);

    base().renderGUI();
    base().memory.renderGUI();

    ImGui::End();
}

void VulkanRenderer::createSwapChain()
{
    swapChain.connect(instance, base().physicalDevice, base().device);
    swapChain.initSurface(surface);
    swapChain.create(&surfaceWidth, &SurfaceHeight, false);
}

void VulkanRenderer::resizeSwapChain()
{
    swapChain.create(&surfaceWidth, &SurfaceHeight, false);
}

void VulkanRenderer::waitIdle()
{
    base().mainQueue.waitIdle();
    //    presentQueue.waitIdle();
    //    transferQueue.waitIdle();
}


}  // namespace Vulkan
}  // namespace Saiga
