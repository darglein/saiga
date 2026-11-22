/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/templatedImage.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/Instance.h"
#include "saiga/vulkan/Parameters.h"
#include "saiga/vulkan/SwapChain.h"
#include "saiga/vulkan/VulkanDevice.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/texture/Texture.h"

#include "FrameTimings.h"

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
class VulkanWindow;
class ImGuiVulkanRenderer;
class StagingBuffer;

/**
 * Projection matrix for Vulkan's clip space (y down, depth in [0, 1]) with reversed depth and an infinite far plane:
 * depth = zNear / (distance along the view direction). View space as in OpenGL (the camera looks along -z, y is up).
 * fovy is the vertical field of view in radians. Used with VulkanForwardRenderer
 * (ForwardRenderingParameters::reversedZ) and RenderScene.
 */
SAIGA_VULKAN_API mat4 reversedZInfinitePerspective(float fovy, float aspect, float zNear);


/**
 * Base class for all Vulkan renderers.
 * This already includes basic functionality that every renderer needs:
 * - Window management
 * - Swap Chain, or offscreen color images if the window has no surface (HeadlessWindow)
 * - Instance and Device
 * - Frame synchronization: framesInFlight frames are recorded/executed concurrently. The completion of a frame is
 *   tracked with VulkanBase::frameTimeline.
 * - For the derived renderers (VulkanForwardRenderer, RenderScene's frame): the per-frame command buffers and their
 *   submit, the ImGui frame, screenshots, the GPU sections (FrameTimings) and the MSAA sample count selection.
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
     * gpuSections: the names of the GPU sections (FrameTimings) in the order of the frame; the derived renderer writes
     * each of them in every frame (enterSection / leaveSection).
     */
    VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters,
                   const std::vector<std::string>& gpuSections);
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
    inline VulkanBase& base() { return device->base; }
    VulkanDevice& vulkanDevice() { return *device; }

    vk::Format getColorFormat() const
    {
        return isHeadless() ? vulkanParameters.offscreenFormat : vk::Format(swapChain.colorFormat);
    }
    // True for an sRGB color format: the hardware encodes the linear values written into it
    static bool isSrgbFormat(vk::Format format);
    // Size of the render surface (the swap chain or offscreen images) in pixels
    ivec2 surfaceSize() const { return ivec2(surfaceWidth, SurfaceHeight); }

    /**
     * True if the window did not create a surface (HeadlessWindow). The frames are then rendered into offscreen images
     * (one per frame in flight) and nothing is presented. No swap chain or surface extension is used.
     */
    bool isHeadless() const { return device->surface == VK_NULL_HANDLE; }
    // Frames submitted so far (render() skips a frame for a minimized window or an out-of-date swap chain)
    uint64_t submittedFrames() const { return submitted; }
    // CPU time of the last frame's command recording and submit (without the wait for the frame slot and the image)
    double lastRecordingMs() const { return recordingMs; }
    /**
     * CPU time the last render call blocked without working: the wait for the frame slot (the GPU still runs the frame
     * that last used it), the swap chain image acquire and the present. With vsync this is mostly waiting for the
     * display; without it, waiting for the GPU. 0 for a frame that was not rendered.
     */
    double lastWaitMs() const { return waitMs; }
    // Its parts: the frame slot (the GPU still runs the frame that last used it), the image acquire, the present
    double lastSlotWaitMs() const { return slotWaitMs; }
    double lastAcquireWaitMs() const { return acquireWaitMs; }
    double lastPresentWaitMs() const { return presentWaitMs; }
    // The last render call presented its frame (headless: rendered it); false for a frame that was not submitted or
    // that the presentation engine rejected (an out-of-date swap chain)
    bool lastPresented() const { return presented; }
    // True if the frames are presented with vsync (FIFO): the frame rate follows the display; false when headless
    bool vsync() const { return !isHeadless() && vulkanParameters.vsync; }
    // The GPU section times of the oldest completed frame not yet taken (FrameTimings::popCompleted)
    bool takeCompletedFrame(FrameTimings<>::CompletedFrame& out) { return timings.popCompleted(out); }

    /**
     * Saves the next rendered frame (including the GUI) to the image file (e.g. png). The file is written at the end
     * of that frame's render call (which then waits for the GPU).
     */
    void requestScreenshot(const std::string& file);

    /**
     * Passes the next rendered frame (including the GUI) to the callback: RGBA with alpha = 255, row 0 is the top
     * row. The callback is called at the end of that frame's render call (which then waits for the GPU). Every
     * requested callback is called exactly once; all requests made before a frame get the same image. If the color
     * images can not be read back (see the error output), the callbacks get an empty image (width 0).
     */
    using ScreenshotCallback = std::function<void(const TemplatedImage<ucvec4>&)>;
    void requestScreenshot(ScreenshotCallback callback) { screenshotCallbacks.push_back(std::move(callback)); }
    // The callbacks not yet taken by a frame, in request order (e.g. withdrawn after a frame that was not submitted)
    std::vector<ScreenshotCallback> takeScreenshotRequests()
    {
        std::vector<ScreenshotCallback> requests;
        requests.swap(screenshotCallbacks);
        return requests;
    }
    // True if the next rendered frame is read back
    bool screenshotRequested() const { return !screenshotCallbacks.empty(); }

    /**
     * The largest sample count <= requested that the device supports for framebuffer color and depth attachments
     * (at least 1). With a color format (not eUndefined): also for sampled images and for that format as a sampled
     * color attachment (getImageFormatProperties), e.g. an MSAA image read by a custom resolve.
     */
    static vk::SampleCountFlagBits selectSampleCount(vk::PhysicalDevice physicalDevice, int requested,
                                                     vk::Format sampledColorFormat = vk::Format::eUndefined);

   protected:
    // The color image a frame renders into (FrameInfo::imageIndex).
    vk::Image colorImage(uint32_t imageIndex) const;
    vk::ImageView colorImageView(uint32_t imageIndex) const;
    // The layout the color image must have at the end of the frame: present src, or transfer src if headless.
    vk::ImageLayout finalColorLayout() const;
    // If the color images can be the source of a copy (screenshots).
    bool colorImagesSupportTransferSrc() const;

    // The command buffer of the frame slot, reset and begun (one time submit)
    vk::CommandBuffer beginCommands(const FrameInfo& frame);
    // Ends and submits the frame's command buffer: waits for imageAvailable, signals renderComplete and the frame
    // timeline (see render(const FrameInfo&))
    void submitCommands(vk::CommandBuffer cmd, const FrameInfo& frame);
    // The ImGui frame (begin, gui, end). It runs even if the GUI is hidden, so that the input state (WantCaptureMouse,
    // ...) stays current and the backend's event queue is consumed. Returns true if ImGui is drawn
    // (should_render_imgui).
    bool imguiFrame(const std::function<void()>& gui);

    /**
     * Screenshots of a frame: begin takes the requested callbacks (empty: none requested) and records the copy of the
     * color image (in eColorAttachmentOptimal, after the last pass) into a host visible buffer, leaving it in
     * eTransferSrcOptimal (returns the buffer; null if not possible). recordPresentTransition hands the color image to
     * the presentation engine (or leaves it readable if headless); finishScreenshots (after the submit) waits for the
     * frame and calls the callbacks (an empty image if the frame could not be read back).
     */
    std::unique_ptr<StagingBuffer> beginScreenshots(vk::CommandBuffer cmd, vk::Image colorImage,
                                                    std::vector<ScreenshotCallback>& callbacks);
    void recordPresentTransition(vk::CommandBuffer cmd, vk::Image colorImage, bool afterScreenshot);
    void finishScreenshots(const FrameInfo& frame, StagingBuffer* screenshot,
                           std::vector<ScreenshotCallback>& callbacks);
    // Records the copy of the color image into a host visible buffer. Returns the buffer.
    std::unique_ptr<StagingBuffer> recordScreenshot(vk::CommandBuffer cmd, vk::Image colorImage);
    TemplatedImage<ucvec4> readScreenshot(StagingBuffer& buffer);
    std::vector<ScreenshotCallback> screenshotCallbacks;

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

    // First: destroyed last (the swap chain and every resource of the derived renderers use it)
    std::unique_ptr<VulkanDevice> device;
    FrameTimings<> timings;
    VulkanWindow& window;

    VulkanSwapChain swapChain;
    VulkanParameters vulkanParameters;

    std::unique_ptr<ImGuiVulkanRenderer> imGui;

    // One per frame in flight (FrameInfo::frameIndex)
    CommandPool renderCommandPool;
    std::vector<vk::CommandBuffer> drawCmdBuffers;

   private:
    uint32_t frameIndex = 0;
    uint64_t submitted  = 0;
    double recordingMs  = 0;
    double waitMs       = 0;
    double slotWaitMs = 0, acquireWaitMs = 0, presentWaitMs = 0;
    bool presented = false;
    // The window's framebuffer size when the swap chain was created
    ivec2 swapChainFramebufferSize = ivec2(0, 0);
    std::array<FrameSync, framesInFlight> frames;

    // One per swap chain image: a binary semaphore may only be signaled again after the presentation engine has
    // consumed it, which is guaranteed once the same image is acquired again.
    std::vector<vk::Semaphore> renderComplete;

    // Headless: the color images (VulkanParameters::offscreenFormat), indexed by the frame index. A frame index is only
    // reused after its previous frame has completed.
    std::vector<Texture2D> offscreenImages;

    void destroySwapChainObjects();
    void reset();
};

}  // namespace Vulkan
}  // namespace Saiga
