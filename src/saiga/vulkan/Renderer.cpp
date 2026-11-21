/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Renderer.h"

#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"
#include "saiga/vulkan/window/Window.h"

#include <chrono>
#include <cmath>
#include <iostream>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

mat4 reversedZInfinitePerspective(float fovy, float aspect, float zNear)
{
    float f = 1.0f / std::tan(fovy * 0.5f);
    mat4 m  = mat4::Zero();
    m(0, 0) = f / aspect;
    // Vulkan's y axis points down
    m(1, 1) = -f;
    // z_clip = zNear, w_clip = -z_view  =>  depth = zNear / -z_view: 1 at the near plane, 0 at infinity
    m(2, 3) = zNear;
    m(3, 2) = -1.0f;
    return m;
}


VulkanRenderer::VulkanRenderer(VulkanWindow& window, VulkanParameters vulkanParameters,
                               const std::vector<std::string>& gpuSections)
    : window(window), vulkanParameters(vulkanParameters)
{
    SAIGA_ASSERT(vulkanParameters.offscreenFormat == vk::Format::eR8G8B8A8Unorm ||
                     vulkanParameters.offscreenFormat == vk::Format::eR8G8B8A8Srgb,
                 "VulkanParameters::offscreenFormat: eR8G8B8A8Unorm or eR8G8B8A8Srgb");
    window.setRenderer(this);
    device = createVulkanDevice(window, vulkanParameters, framesInFlight);

    if (!isHeadless())
    {
        swapChain.connect(device->instance, base().physicalDevice, base().device);
        // The device owns the surface (the swap chain does not destroy it)
        swapChain.initSurface(device->surface);
        SAIGA_ASSERT(swapChain.queueNodeIndex == base().mainQueue.getQueueFamilyIndex(),
                     "The main queue must be able to present");
    }

    for (auto& f : frames) f.create(base().device);

    state = State::INITIALIZED;

    timings.init(base().device, base().physicalDevice, &base().mainQueue);
    for (size_t i = 0; i < gpuSections.size(); ++i) timings.registerFrameSection(gpuSections[i], uint32_t(i));
    // One more entry than frames in flight: up to framesInFlight entries are unread, and a full ring must be
    // distinguishable from an empty one.
    timings.create(framesInFlight + 1);

    if (vulkanParameters.enableImgui)
    {
        imGui = window.createImGui(framesInFlight);
    }

    renderCommandPool = base().mainQueue.createCommandPool(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    drawCmdBuffers    = renderCommandPool.allocateCommandBuffers(framesInFlight, vk::CommandBufferLevel::ePrimary);
}

VulkanRenderer::~VulkanRenderer()
{
    VLOG(3) << "Destroying VulkanRenderer";
    waitIdle();
    renderCommandPool.freeCommandBuffers(drawCmdBuffers);
    drawCmdBuffers.clear();
    destroySwapChainObjects();
    for (auto& f : frames) f.destroy();
    // The remaining objects are destroyed in their destructors (members in reverse order, the device last).
}

void VulkanRenderer::destroySwapChainObjects()
{
    for (auto s : renderComplete) base().device.destroySemaphore(s);
    renderComplete.clear();
    // Deferred, see Texture::destroy.
    offscreenImages.clear();
}

vk::Image VulkanRenderer::colorImage(uint32_t imageIndex) const
{
    if (isHeadless()) return offscreenImages[imageIndex].memoryLocation->data.image;
    return swapChain.images[imageIndex];
}

vk::ImageView VulkanRenderer::colorImageView(uint32_t imageIndex) const
{
    if (isHeadless()) return offscreenImages[imageIndex].memoryLocation->data.view;
    return swapChain.buffers[imageIndex].view;
}

vk::ImageLayout VulkanRenderer::finalColorLayout() const
{
    return isHeadless() ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::ePresentSrcKHR;
}

bool VulkanRenderer::colorImagesSupportTransferSrc() const
{
    if (isHeadless())
    {
        return bool(offscreenImages.front().type.usageFlags & vk::ImageUsageFlagBits::eTransferSrc);
    }
    return swapChain.imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
}

void VulkanRenderer::init()
{
    SAIGA_ASSERT(state == State::INITIALIZED);

    // Used by the swap chain if the surface does not define its size (e.g. Wayland).
    ivec2 framebuffer_size   = window.getFramebufferSize();
    surfaceWidth             = framebuffer_size.x();
    SurfaceHeight            = framebuffer_size.y();
    swapChainFramebufferSize = framebuffer_size;

    destroySwapChainObjects();
    if (isHeadless())
    {
        offscreenImages.resize(framesInFlight);
        for (auto& image : offscreenImages)
        {
            image.create(base(), surfaceWidth, SurfaceHeight, vulkanParameters.offscreenFormat,
                         vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled);
        }
    }
    else
    {
        swapChain.create(&surfaceWidth, &SurfaceHeight, vulkanParameters.vsync);
        for (uint32_t i = 0; i < swapChain.imageCount; ++i)
        {
            renderComplete.push_back(base().device.createSemaphore(vk::SemaphoreCreateInfo{}));
        }
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

    FrameInfo frame;
    frame.frameIndex    = frameIndex;
    frame.timelineValue = base().currentFrameSignalValue();

    VkResult err    = VK_SUCCESS;
    bool suboptimal = false;
    if (isHeadless())
    {
        // The frame that last rendered into this image has completed (see the wait above).
        frame.imageIndex = frameIndex;
    }
    else
    {
        uint32_t imageIndex = 0;
        err                 = swapChain.acquireNextImage(sync.imageAvailable, &imageIndex);
        if (err == VK_ERROR_OUT_OF_DATE_KHR)
        {
            reset();
            return;
        }
        // VK_SUBOPTIMAL_KHR is a success code: the image was acquired, so render and present it and recreate the swap
        // chain afterwards.
        suboptimal = err == VK_SUBOPTIMAL_KHR;
        if (!suboptimal) VK_CHECK_RESULT(err);

        frame.imageIndex     = imageIndex;
        frame.imageAvailable = sync.imageAvailable;
        frame.renderComplete = renderComplete[imageIndex];
    }

    timings.beginFrame(frame.timelineValue);
    const auto recordStart = std::chrono::steady_clock::now();
    render(frame);
    recordingMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - recordStart).count();
    submitted++;
    sync.timelineValue = frame.timelineValue;

    if (!isHeadless())
    {
        err = swapChain.queuePresent(base().mainQueue, frame.imageIndex, frame.renderComplete);
    }

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

bool VulkanRenderer::isSrgbFormat(vk::Format format)
{
    return format == vk::Format::eB8G8R8A8Srgb || format == vk::Format::eR8G8B8A8Srgb ||
           format == vk::Format::eA8B8G8R8SrgbPack32;
}

vk::SampleCountFlagBits VulkanRenderer::selectSampleCount(vk::PhysicalDevice physicalDevice, int requested,
                                                          vk::Format sampledColorFormat)
{
    auto limits        = physicalDevice.getProperties().limits;
    uint32_t supported = uint32_t(limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts);
    if (sampledColorFormat != vk::Format::eUndefined)
    {
        supported &= uint32_t(limits.sampledImageColorSampleCounts);
        vk::ImageFormatProperties props;
        auto result = physicalDevice.getImageFormatProperties(
            sampledColorFormat, vk::ImageType::e2D, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, {}, &props);
        supported &= result == vk::Result::eSuccess ? uint32_t(props.sampleCounts) : 1u;
    }
    uint32_t result = 1;
    for (uint32_t c = 1; c <= 64; c *= 2)
    {
        if (int(c) <= requested && (supported & c)) result = c;
    }
    return vk::SampleCountFlagBits(result);
}

vk::CommandBuffer VulkanRenderer::beginCommands(const FrameInfo& frame)
{
    vk::CommandBuffer cmd = drawCmdBuffers[frame.frameIndex];
    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
    return cmd;
}

void VulkanRenderer::submitCommands(vk::CommandBuffer cmd, const FrameInfo& frame)
{
    cmd.end();

    // A headless renderer has no swap chain semaphores (FrameInfo).
    std::vector<vk::SemaphoreSubmitInfo> waitInfos, signalInfos;
    if (frame.imageAvailable) waitInfos.emplace_back(frame.imageAvailable, 0, Stage::eColorAttachmentOutput);
    if (frame.renderComplete) signalInfos.emplace_back(frame.renderComplete, 0, Stage::eAllCommands);
    signalInfos.emplace_back(base().frameTimeline, frame.timelineValue, Stage::eAllCommands);
    vk::CommandBufferSubmitInfo cmdInfo(cmd);

    vk::SubmitInfo2 submitInfo;
    submitInfo.waitSemaphoreInfoCount   = (uint32_t)waitInfos.size();
    submitInfo.pWaitSemaphoreInfos      = waitInfos.data();
    submitInfo.commandBufferInfoCount   = 1;
    submitInfo.pCommandBufferInfos      = &cmdInfo;
    submitInfo.signalSemaphoreInfoCount = (uint32_t)signalInfos.size();
    submitInfo.pSignalSemaphoreInfos    = signalInfos.data();
    base().mainQueue.submit2(submitInfo);
}

bool VulkanRenderer::imguiFrame(const std::function<void()>& gui)
{
    if (!imGui) return false;
    imGui->beginFrame();
    if (gui) gui();
    imGui->endFrame();
    return should_render_imgui;
}

void VulkanRenderer::requestScreenshot(const std::string& file)
{
    requestScreenshot(
        [file](const TemplatedImage<ucvec4>& img)
        {
            if (img.width > 0 && img.save(file))
                std::cout << "Saved screenshot " << file << std::endl;
            else
                std::cerr << "Could not save screenshot " << file << std::endl;
        });
}

std::unique_ptr<StagingBuffer> VulkanRenderer::beginScreenshots(vk::CommandBuffer cmd, vk::Image colorImage,
                                                                std::vector<ScreenshotCallback>& callbacks)
{
    callbacks.clear();
    if (screenshotCallbacks.empty()) return nullptr;
    callbacks.swap(screenshotCallbacks);
    return recordScreenshot(cmd, colorImage);
}

void VulkanRenderer::recordPresentTransition(vk::CommandBuffer cmd, vk::Image colorImage, bool afterScreenshot)
{
    // Hand the image to the presentation engine (the present waits for renderComplete), headless: leave it readable.
    // After a screenshot the image is in TRANSFER_SRC.
    auto layout = afterScreenshot ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eColorAttachmentOptimal;
    auto stage  = afterScreenshot ? vk::PipelineStageFlags2(Stage::eAllTransfer)
                                  : vk::PipelineStageFlags2(Stage::eColorAttachmentOutput);
    auto access = afterScreenshot ? vk::AccessFlags2() : vk::AccessFlags2(Access::eColorAttachmentWrite);
    vk::ImageMemoryBarrier2 present(stage, access, Stage::eNone, {}, layout, finalColorLayout(),
                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, colorImage,
                                    vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, present));
}

void VulkanRenderer::finishScreenshots(const FrameInfo& frame, StagingBuffer* screenshot,
                                       std::vector<ScreenshotCallback>& callbacks)
{
    if (callbacks.empty()) return;
    // Every callback is answered, with an empty image if the frame could not be read back
    TemplatedImage<ucvec4> image;
    if (screenshot)
    {
        base().waitFrameValue(frame.timelineValue);
        image = readScreenshot(*screenshot);
    }
    for (auto& callback : callbacks) callback(image);
    callbacks.clear();
}

std::unique_ptr<StagingBuffer> VulkanRenderer::recordScreenshot(vk::CommandBuffer cmd, vk::Image colorImage)
{
    if (!colorImagesSupportTransferSrc())
    {
        std::cerr << "Screenshot not possible: the color images do not support TRANSFER_SRC" << std::endl;
        return nullptr;
    }
    auto format = getColorFormat();
    if (format != vk::Format::eB8G8R8A8Unorm && format != vk::Format::eB8G8R8A8Srgb &&
        format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eR8G8B8A8Srgb)
    {
        std::cerr << "Screenshot not possible: unsupported color format " << vk::to_string(format) << std::endl;
        return nullptr;
    }

    auto buffer = std::make_unique<StagingBuffer>();
    buffer->init(base(), size_t(surfaceWidth) * SurfaceHeight * 4);

    vk::ImageMemoryBarrier2 toTransfer(
        Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite, Stage::eAllTransfer, Access::eTransferRead,
        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, colorImage, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toTransfer));

    vk::BufferImageCopy region = buffer->getBufferImageCopy(0);
    region.imageSubresource    = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
    region.imageExtent         = vk::Extent3D(surfaceWidth, SurfaceHeight, 1);
    cmd.copyImageToBuffer(colorImage, vk::ImageLayout::eTransferSrcOptimal, buffer->createInfo().buffer, region);

    // Make the copy visible to the host read after the wait.
    vk::MemoryBarrier2 toHost(Stage::eAllTransfer, Access::eTransferWrite, Stage::eHost, Access::eHostRead);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, toHost, nullptr, nullptr));
    return buffer;
}

TemplatedImage<ucvec4> VulkanRenderer::readScreenshot(StagingBuffer& buffer)
{
    TemplatedImage<ucvec4> img(SurfaceHeight, surfaceWidth);
    buffer.download(img.data(), size_t(surfaceWidth) * SurfaceHeight * 4);
    auto format = getColorFormat();
    if (format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eB8G8R8A8Srgb)
    {
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x) std::swap(img(y, x)(0), img(y, x)(2));
    }
    // The swap chain content is not meant to be blended: store it opaque.
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x)(3) = 255;

    return img;
}

void VulkanRenderer::waitIdle()
{
    // All queues, not only the main queue: the dedicated compute/transfer queues may still use resources.
    if (base().device) base().device.waitIdle();
}

}  // namespace Vulkan
}  // namespace Saiga
