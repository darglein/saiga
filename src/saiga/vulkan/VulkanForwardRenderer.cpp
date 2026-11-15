/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanForwardRenderer.h"

#include "saiga/core/image/managedImage.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/imgui/ImGuiVulkanRenderer.h"

#include <cmath>
#include <cstdlib>

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

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

// The largest supported sample count <= requested
static vk::SampleCountFlagBits selectSampleCount(vk::PhysicalDevice physicalDevice, int requested)
{
    auto limits     = physicalDevice.getProperties().limits;
    auto supported  = uint32_t(limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts);
    uint32_t result = 1;
    for (uint32_t c = 1; c <= 64; c *= 2)
    {
        if (int(c) <= requested && (supported & c)) result = c;
    }
    return vk::SampleCountFlagBits(result);
}

VulkanForwardRenderer::VulkanForwardRenderer(VulkanWindow& window, VulkanParameters vulkanParameters,
                                             const ForwardRenderingParameters& parameters)
    : VulkanRenderer(window, vulkanParameters), params(parameters)
{
    samples     = selectSampleCount(base().physicalDevice, params.msaaSamples);
    depthFormat = params.depthFormat;
    if (depthFormat == vk::Format::eUndefined)
    {
        depthFormat = params.reversedZ ? vk::Format::eD32Sfloat : vk::Format::eD16Unorm;
    }
    if (int(samples) != params.msaaSamples)
    {
        std::cout << "VulkanForwardRenderer: " << params.msaaSamples << "x MSAA is not supported, using "
                  << int(samples) << "x" << std::endl;
    }

    renderCommandPool = base().mainQueue.createCommandPool(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    drawCmdBuffers    = renderCommandPool.allocateCommandBuffers(framesInFlight, vk::CommandBufferLevel::ePrimary);

    if (imGui) imGui->initResources(base(), renderingFormats());

    if (const char* file = std::getenv("SAIGA_SCREENSHOT"))
    {
        autoScreenshotFile = file;
    }
    std::cout << "VulkanForwardRenderer init done." << std::endl;
}

VulkanForwardRenderer::~VulkanForwardRenderer()
{
    waitIdle();
    renderCommandPool.freeCommandBuffers(drawCmdBuffers);
}

RenderingFormats VulkanForwardRenderer::renderingFormats() const
{
    RenderingFormats formats;
    formats.color   = {getColorFormat()};
    formats.depth   = depthFormat;
    formats.samples = samples;
    return formats;
}

void VulkanForwardRenderer::createBuffers(int numImages, int w, int h)
{
    // Only called after waitIdle (see VulkanRenderer::reset), so the old images are not in use anymore.
    depthImage.destroy();
    msaaColor.destroy();
    depthImage.create(base(), w, h, depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment, 1, samples);
    if (samples != vk::SampleCountFlagBits::e1)
    {
        msaaColor.create(base(), w, h, getColorFormat(), vk::ImageUsageFlagBits::eColorAttachment, 1, samples);
    }
}

void VulkanForwardRenderer::render(const FrameInfo& frame)
{
    VulkanForwardRenderingInterface* renderingInterface = dynamic_cast<VulkanForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    // The ImGui frame runs even if the GUI is hidden, so that the input state (WantCaptureMouse, ...) stays current and
    // the backend's event queue is consumed. should_render_imgui only decides whether it is drawn.
    if (imGui)
    {
        imGui->beginFrame();
        renderingInterface->renderGUI();
        imGui->endFrame();
    }
    bool draw_imgui = imGui && should_render_imgui;

    renderedFrames++;
    bool close_after_frame = false;
    if (!autoScreenshotFile.empty() && renderedFrames == 60)
    {
        requestScreenshot(autoScreenshotFile);
        close_after_frame = true;
    }

    vk::CommandBuffer cmd = drawCmdBuffers[frame.frameIndex];
    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    timings.resetFrame(cmd);
    timings.enterSection("TRANSFER", cmd);
    {
        // The buffers written in transfer() (uniform buffers, vertex data) are shared by all frames in flight:
        // wait until earlier frames have read them (write-after-read) ...
        vk::MemoryBarrier2 war(Stage::eVertexInput | Stage::eVertexShader | Stage::eFragmentShader, {},
                               Stage::eAllTransfer, {});
        cmd.pipelineBarrier2(vk::DependencyInfo({}, war, nullptr, nullptr));

        renderingInterface->transfer(cmd);

        // ... and make the writes visible to this frame's draws.
        vk::MemoryBarrier2 raw(
            Stage::eAllTransfer, Access::eTransferWrite,
            Stage::eVertexInput | Stage::eVertexShader | Stage::eFragmentShader,
            Access::eUniformRead | Access::eShaderRead | Access::eVertexAttributeRead | Access::eIndexRead);
        cmd.pipelineBarrier2(vk::DependencyInfo({}, raw, nullptr, nullptr));
    }
    timings.leaveSection("TRANSFER", cmd);

    // ImGui's vertex data is in host visible memory, one buffer per frame in flight.
    if (draw_imgui) imGui->updateBuffers(cmd, frame.frameIndex);

    vk::Image colorImage   = this->colorImage(frame.imageIndex);
    vk::Image depthVkImage = depthImage.image();
    bool multisampled      = samples != vk::SampleCountFlagBits::e1;

    {
        // The swap chain image is written after the acquire (the submit waits for imageAvailable at
        // COLOR_ATTACHMENT_OUTPUT). The previous content is discarded.
        // Headless: the previous frame that used this image has completed (VulkanRenderer::render).
        vk::ImageMemoryBarrier2 color(Stage::eColorAttachmentOutput, {}, Stage::eColorAttachmentOutput,
                                      Access::eColorAttachmentWrite | Access::eColorAttachmentRead,
                                      vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, colorImage,
                                      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

        // The depth buffer is shared by all frames: the clear must wait for the previous frame's depth tests.
        // Combined depth/stencil formats need both aspects (separateDepthStencilLayouts is not enabled).
        vk::ImageAspectFlags depthAspect = vk::ImageAspectFlagBits::eDepth;
        if (depthFormat == vk::Format::eD16UnormS8Uint || depthFormat == vk::Format::eD24UnormS8Uint ||
            depthFormat == vk::Format::eD32SfloatS8Uint)
            depthAspect |= vk::ImageAspectFlagBits::eStencil;
        vk::ImageMemoryBarrier2 depth(
            Stage::eEarlyFragmentTests | Stage::eLateFragmentTests, Access::eDepthStencilAttachmentWrite,
            Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
            Access::eDepthStencilAttachmentWrite | Access::eDepthStencilAttachmentRead, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            depthVkImage, vk::ImageSubresourceRange(depthAspect, 0, 1, 0, 1));

        std::vector<vk::ImageMemoryBarrier2> barriers = {color, depth};
        if (multisampled)
        {
            // Shared by all frames as the depth buffer, the previous content is discarded.
            barriers.emplace_back(
                Stage::eColorAttachmentOutput, Access::eColorAttachmentWrite, Stage::eColorAttachmentOutput,
                Access::eColorAttachmentWrite | Access::eColorAttachmentRead, vk::ImageLayout::eUndefined,
                vk::ImageLayout::eColorAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                msaaColor.image(), vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        }
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barriers));
        depthImage.setImageLayout(vk::ImageLayout::eDepthStencilAttachmentOptimal);
        if (multisampled) msaaColor.setImageLayout(vk::ImageLayout::eColorAttachmentOptimal);
    }

    {
        vec4 clearColor = params.clearColor;

        vk::RenderingAttachmentInfo colorAttachment;
        colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        colorAttachment.loadOp      = vk::AttachmentLoadOp::eClear;
        if (multisampled)
        {
            // Only the resolved image is kept.
            colorAttachment.imageView          = msaaColor.view();
            colorAttachment.storeOp            = vk::AttachmentStoreOp::eDontCare;
            colorAttachment.resolveMode        = vk::ResolveModeFlagBits::eAverage;
            colorAttachment.resolveImageView   = colorImageView(frame.imageIndex);
            colorAttachment.resolveImageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }
        else
        {
            colorAttachment.imageView = colorImageView(frame.imageIndex);
            colorAttachment.storeOp   = vk::AttachmentStoreOp::eStore;
        }
        colorAttachment.clearValue.color =
            vk::ClearColorValue(std::array<float, 4>{clearColor[0], clearColor[1], clearColor[2], clearColor[3]});

        vk::RenderingAttachmentInfo depthAttachment;
        depthAttachment.imageView                     = depthImage.view();
        depthAttachment.imageLayout                   = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depthAttachment.loadOp                        = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp                       = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.clearValue.depthStencil.depth = clearDepth();

        vk::RenderingInfo renderingInfo;
        renderingInfo.renderArea           = vk::Rect2D({0, 0}, {(uint32_t)surfaceWidth, (uint32_t)SurfaceHeight});
        renderingInfo.layerCount           = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments    = &colorAttachment;
        renderingInfo.pDepthAttachment     = &depthAttachment;

        cmd.beginRendering(renderingInfo);
    }

    cmd.setViewport(0, vk::Viewport(0, 0, (float)surfaceWidth, (float)SurfaceHeight, 0.0f, 1.0f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {(uint32_t)surfaceWidth, (uint32_t)SurfaceHeight}));

    timings.enterSection("MAIN", cmd);
    renderingInterface->render(cmd);
    timings.leaveSection("MAIN", cmd);

    timings.enterSection("IMGUI", cmd);
    if (draw_imgui) imGui->render(cmd, frame.frameIndex);
    timings.leaveSection("IMGUI", cmd);

    cmd.endRendering();

    std::unique_ptr<StagingBuffer> screenshot;
    std::vector<ScreenshotCallback> screenshot_callbacks;
    if (!screenshotCallbacks.empty())
    {
        screenshot_callbacks.swap(screenshotCallbacks);
        screenshot = recordScreenshot(cmd, colorImage);
    }

    {
        // Hand the image to the presentation engine (the present waits for renderComplete), headless: leave it
        // readable. After a screenshot the image is in TRANSFER_SRC.
        auto layout = screenshot ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eColorAttachmentOptimal;
        auto stage  = screenshot ? vk::PipelineStageFlags2(Stage::eAllTransfer)
                                 : vk::PipelineStageFlags2(Stage::eColorAttachmentOutput);
        auto access = screenshot ? vk::AccessFlags2() : vk::AccessFlags2(Access::eColorAttachmentWrite);
        vk::ImageMemoryBarrier2 present(stage, access, Stage::eNone, {}, layout, finalColorLayout(),
                                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, colorImage,
                                        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, present));
    }

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

    if (!screenshot_callbacks.empty())
    {
        // Every callback is answered, with an empty image if the frame could not be read back
        TemplatedImage<ucvec4> image;
        if (screenshot)
        {
            base().waitFrameValue(frame.timelineValue);
            image = readScreenshot(*screenshot);
        }
        for (auto& callback : screenshot_callbacks) callback(image);
    }
    if (close_after_frame) window.close();
}

void VulkanForwardRenderer::requestScreenshot(const std::string& file)
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

std::unique_ptr<StagingBuffer> VulkanForwardRenderer::recordScreenshot(vk::CommandBuffer cmd, vk::Image colorImage)
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

TemplatedImage<ucvec4> VulkanForwardRenderer::readScreenshot(StagingBuffer& buffer)
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

}  // namespace Vulkan
}  // namespace Saiga
