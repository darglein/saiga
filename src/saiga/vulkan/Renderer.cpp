/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Renderer.h"

#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/window/Window.h"

namespace Saiga
{
namespace Vulkan
{
VulkanRenderer::VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters)
    : window(window), vulkanParameters(vulkanParameters)
{
    window.setRenderer(this);

    auto instanceExtensions = window.getRequiredInstanceExtensions();
    instance.create(instanceExtensions, vulkanParameters.enableValidationLayer);

    window.createSurface(instance, &surface);

    base().setPhysicalDevice(instance, instance.pickPhysicalDevice(surface, vulkanParameters.preferredDevice));

    vulkanParameters.physicalDeviceFeatures.fillModeNonSolid = VK_TRUE;
    vulkanParameters.physicalDeviceFeatures.wideLines        = VK_TRUE;
    vulkanParameters.physicalDeviceFeatures.largePoints      = VK_TRUE;
    base().createLogicalDevice(vulkanParameters, surface);

    swapChain.connect(instance, base().physicalDevice, base().device);
    swapChain.initSurface(surface);
    SAIGA_ASSERT(swapChain.queueNodeIndex == base().mainQueue.getQueueFamilyIndex(),
                 "The main queue must be able to present");

    for (auto& f : frames) f.create(base().device);

    state = State::INITIALIZED;
    base().initMemory(framesInFlight);

    timings.init(base().device, base().physicalDevice, &base().mainQueue, &(base().memory));
    timings.registerFrameSection("TRANSFER", 0);
    timings.registerFrameSection("MAIN", 1);
    timings.registerFrameSection("IMGUI", 2);
    // One more entry than frames in flight: up to framesInFlight entries are unread, and a full ring must be
    // distinguishable from an empty one.
    timings.create(framesInFlight + 1);

    if (vulkanParameters.enableImgui)
    {
        imGui = window.createImGui(framesInFlight);
    }
}

VulkanRenderer::~VulkanRenderer()
{
    VLOG(3) << "Destroying VulkanRenderer";
    waitIdle();
    destroySwapChainObjects();
    for (auto& f : frames) f.destroy();
    // The remaining objects are destroyed in their destructors (members in reverse order, the device last).
}

void VulkanRenderer::destroySwapChainObjects()
{
    for (auto s : renderComplete) base().device.destroySemaphore(s);
    renderComplete.clear();
}

void VulkanRenderer::init()
{
    SAIGA_ASSERT(state == State::INITIALIZED);

    // Used by the swap chain if the surface does not define its size (e.g. Wayland).
    ivec2 framebuffer_size   = window.getFramebufferSize();
    surfaceWidth             = framebuffer_size.x();
    SurfaceHeight            = framebuffer_size.y();
    swapChainFramebufferSize = framebuffer_size;
    swapChain.create(&surfaceWidth, &SurfaceHeight, vulkanParameters.vsync);

    destroySwapChainObjects();
    for (uint32_t i = 0; i < swapChain.imageCount; ++i)
    {
        renderComplete.push_back(base().device.createSemaphore(vk::SemaphoreCreateInfo{}));
    }

    createBuffers(swapChainSize(), surfaceWidth, SurfaceHeight);

    state = State::RENDERABLE;
}

void VulkanRenderer::render(const RenderInfo& renderInfo)
{
    // A minimized window has a 0x0 framebuffer, for which no swap chain can be created.
    ivec2 framebuffer_size = window.getFramebufferSize();
    if (framebuffer_size.x() <= 0 || framebuffer_size.y() <= 0)
    {
        if (state == State::RENDERABLE) reset();
        return;
    }

    // Surfaces without a fixed extent (Wayland) may not report OUT_OF_DATE/SUBOPTIMAL when the framebuffer size
    // changes (e.g. a scale change). Compared with the framebuffer size at swap chain creation (the swap chain extent
    // can differ from it by clamping).
    if (state == State::RENDERABLE && framebuffer_size != swapChainFramebufferSize)
    {
        reset();
    }

    if (state == State::RESET)
    {
        state = State::INITIALIZED;
    }

    if (state == State::INITIALIZED)
    {
        init();
    }

    FrameSync& sync = frames[frameIndex];

    // The frame that last used this slot (its command buffer, imageAvailable semaphore, ...) must be complete.
    base().waitFrameValue(sync.timelineValue);
    timings.update(base().frameTimeline);

    uint32_t imageIndex = 0;
    VkResult err        = swapChain.acquireNextImage(sync.imageAvailable, &imageIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR)
    {
        reset();
        return;
    }
    // VK_SUBOPTIMAL_KHR is a success code: the image was acquired, so render and present it and recreate the swap
    // chain afterwards.
    bool suboptimal = err == VK_SUBOPTIMAL_KHR;
    if (!suboptimal) VK_CHECK_RESULT(err);

    FrameInfo frame;
    frame.frameIndex     = frameIndex;
    frame.imageIndex     = imageIndex;
    frame.imageAvailable = sync.imageAvailable;
    frame.renderComplete = renderComplete[imageIndex];
    frame.timelineValue  = base().currentFrameSignalValue();

    timings.beginFrame(frame.timelineValue);
    render(frame);
    sync.timelineValue = frame.timelineValue;

    // Defragmentation copies may run after this frame (they wait for its timeline value).
    timings.finishFrame(base().frameTimeline, frame.timelineValue);

    err = swapChain.queuePresent(base().mainQueue, imageIndex, frame.renderComplete);

    base().finish_frame();
    frameIndex = (frameIndex + 1) % framesInFlight;

    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR || suboptimal)
    {
        reset();
        return;
    }
    VK_CHECK_RESULT(err);
}

void VulkanRenderer::reset()
{
    SAIGA_ASSERT(state == State::RESET || state == State::RENDERABLE);
    waitIdle();
    state = State::RESET;
}

void VulkanRenderer::renderImgui()
{
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("Renderer Info", &should_render_imgui, ImGuiWindowFlags_NoCollapse);

    base().renderGUI();
    base().memory.renderGUI();

    ImGui::End();
}

void VulkanRenderer::waitIdle()
{
    // All queues, not only the main queue: the dedicated compute/transfer queues may still use resources.
    if (base().device) base().device.waitIdle();
}

}  // namespace Vulkan
}  // namespace Saiga
