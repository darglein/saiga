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
#include "saiga/vulkan/texture/Texture.h"

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
 * - Swap Chain, or offscreen color images if the window has no surface (HeadlessWindow)
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

    // Number of color images: swap chain images, or the offscreen images of a headless renderer.
    int swapChainSize() { return isHeadless() ? (int)offscreenImages.size() : (int)swapChain.imageCount; }
    inline VulkanBase& base() { return vulkanBase; }

    vk::Format getColorFormat() const { return isHeadless() ? offscreenFormat : vk::Format(swapChain.colorFormat); }

    /**
     * True if the window did not create a surface (HeadlessWindow). The frames are then rendered into offscreen images
     * (one per frame in flight) and nothing is presented. No swap chain or surface extension is used.
     */
    bool isHeadless() const { return surface == VK_NULL_HANDLE; }

   protected:
    // The color image a frame renders into (FrameInfo::imageIndex).
    vk::Image colorImage(uint32_t imageIndex) const;
    vk::ImageView colorImageView(uint32_t imageIndex) const;
    // The layout the color image must have at the end of the frame: present src, or transfer src if headless.
    vk::ImageLayout finalColorLayout() const;
    // If the color images can be the source of a copy (screenshots).
    bool colorImagesSupportTransferSrc() const;

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
    // The window's framebuffer size when the swap chain was created
    ivec2 swapChainFramebufferSize = ivec2(0, 0);
    std::array<FrameSync, framesInFlight> frames;

    // One per swap chain image: a binary semaphore may only be signaled again after the presentation engine has
    // consumed it, which is guaranteed once the same image is acquired again.
    std::vector<vk::Semaphore> renderComplete;

    // Headless: the color images, indexed by the frame index. A frame index is only reused after its previous frame
    // has completed.
    static constexpr vk::Format offscreenFormat = vk::Format::eR8G8B8A8Unorm;
    std::vector<Texture2D> offscreenImages;

    void destroySwapChainObjects();
    void reset();
};

}  // namespace Vulkan
}  // namespace Saiga
