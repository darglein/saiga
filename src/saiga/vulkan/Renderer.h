/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/SwapChain.h"
#include "saiga/vulkan/svulkan.h"

#include "FrameTimings.h"

#include <array>

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;
class ImGuiVulkanRenderer;

/**
 * Base class for all Vulkan renderers.
 * This already includes basic functionality that every renderer needs:
 * - Window management
 * - Swap Chain
 * - Instance and Device
 * - Frame synchronization: framesInFlight frames are recorded/executed concurrently. The completion of a frame is
 *   tracked with VulkanBase::frameTimeline.
 */
class SAIGA_VULKAN_API VulkanRenderer : public RendererBase
{
   public:
    // Number of frames that can be in flight (recorded by the CPU while the GPU executes the previous ones).
    static constexpr uint32_t framesInFlight = 2;

    /**
     * - Create Instance
     * - Create Device
     * - Create Surface
     * - Create ImGui (if enabled)
     */
    VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters);
    virtual ~VulkanRenderer() override;

    /**
     * - Create Swapchain
     * - Create the derived render objects (createBuffers)
     *
     * Called every time the swapchain has to be recreated (e.g. window size changes).
     */
    void init();

    /**
     * Must be overriden by the actual renderers.
     * Creates the objects that depend on the swap chain (e.g. the depth buffer).
     * Called every time the swapchain is recreated, so make sure there are no leaks.
     */
    virtual void createBuffers(int numImages, int w, int h) = 0;

    /**
     * Records and submits the frame. The submit must wait for frame.imageAvailable and signal frame.renderComplete and
     * VulkanBase::frameTimeline with frame.timelineValue (see FrameInfo).
     */
    virtual void render(const FrameInfo& frame) = 0;

    virtual void render(const RenderInfo& renderInfo) override;

    void renderImgui() override;

    // Waits until the device is idle (all queues).
    void waitIdle();

    int swapChainSize() { return swapChain.imageCount; }
    inline VulkanBase& base() { return vulkanBase; }

    vk::Format getColorFormat() const { return vk::Format(swapChain.colorFormat); }

   protected:
    /**
     * Size of the render surface (the swap chain images) in pixels.
     */
    int surfaceWidth  = 1280;
    int SurfaceHeight = 720;

    enum class State
    {
        // Initial State
        UNINITIALIZED,
        // After Constructor
        INITIALIZED,
        // After init()
        RENDERABLE,
        // Happens at resize window or change swapchain settings
        RESET
    };
    State state = State::UNINITIALIZED;

    Instance instance;
    VulkanBase vulkanBase;
    FrameTimings<> timings;
    VulkanWindow& window;
    VkSurfaceKHR surface = VK_NULL_HANDLE;

    VulkanSwapChain swapChain;
    VulkanParameters vulkanParameters;

    std::unique_ptr<ImGuiVulkanRenderer> imGui;

   private:
    uint32_t frameIndex = 0;
    std::array<FrameSync, framesInFlight> frames;

    // One per swap chain image: a binary semaphore may only be signaled again after the presentation engine has
    // consumed it, which is guaranteed once the same image is acquired again.
    std::vector<vk::Semaphore> renderComplete;

    void destroySwapChainObjects();
    void reset();
};

}  // namespace Vulkan
}  // namespace Saiga
