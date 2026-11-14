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

VulkanForwardRenderer::VulkanForwardRenderer(VulkanWindow& window, VulkanParameters vulkanParameters)
    : VulkanRenderer(window, vulkanParameters)
{
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
    formats.color = {getColorFormat()};
    formats.depth = depthBuffer.format;
    return formats;
}

void VulkanForwardRenderer::createBuffers(int numImages, int w, int h)
{
    // Only called after waitIdle (see VulkanRenderer::reset), so the old depth buffer is not in use anymore.
    depthBuffer.init(base(), w, h);
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

    vk::Image colorImage = swapChain.images[frame.imageIndex];
    vk::Image depthImage = depthBuffer.location->data.image;

    {
        // The swap chain image is written after the acquire (the submit waits for imageAvailable at
        // COLOR_ATTACHMENT_OUTPUT). The previous content is discarded.
        vk::ImageMemoryBarrier2 color(Stage::eColorAttachmentOutput, {}, Stage::eColorAttachmentOutput,
                                      Access::eColorAttachmentWrite | Access::eColorAttachmentRead,
                                      vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                                      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, colorImage,
                                      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

        // The depth buffer is shared by all frames: the clear must wait for the previous frame's depth tests.
        vk::ImageMemoryBarrier2 depth(
            Stage::eEarlyFragmentTests | Stage::eLateFragmentTests, Access::eDepthStencilAttachmentWrite,
            Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
            Access::eDepthStencilAttachmentWrite | Access::eDepthStencilAttachmentRead, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eDepthStencilAttachmentOptimal, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            depthImage, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1));

        std::array<vk::ImageMemoryBarrier2, 2> barriers = {color, depth};
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barriers));
    }

    {
        // This is blender's default viewport background color :)
        vec4 clearColor = vec4(57, 57, 57, 255) / 255.0f;

        vk::RenderingAttachmentInfo colorAttachment;
        colorAttachment.imageView   = swapChain.buffers[frame.imageIndex].view;
        colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        colorAttachment.loadOp      = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp     = vk::AttachmentStoreOp::eStore;
        colorAttachment.clearValue.color =
            vk::ClearColorValue(std::array<float, 4>{clearColor[0], clearColor[1], clearColor[2], clearColor[3]});

        vk::RenderingAttachmentInfo depthAttachment;
        depthAttachment.imageView                     = depthBuffer.location->data.view;
        depthAttachment.imageLayout                   = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depthAttachment.loadOp                        = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp                       = vk::AttachmentStoreOp::eDontCare;
        depthAttachment.clearValue.depthStencil.depth = 1.0f;

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
    std::string screenshot_file;
    if (!screenshotFile.empty())
    {
        screenshot_file = screenshotFile;
        screenshotFile.clear();
        screenshot = recordScreenshot(cmd, colorImage);
    }

    {
        // Hand the image to the presentation engine (the present waits for renderComplete). After a screenshot the
        // image is in TRANSFER_SRC.
        auto layout = screenshot ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eColorAttachmentOptimal;
        auto stage  = screenshot ? vk::PipelineStageFlags2(Stage::eAllTransfer)
                                 : vk::PipelineStageFlags2(Stage::eColorAttachmentOutput);
        auto access = screenshot ? vk::AccessFlags2() : vk::AccessFlags2(Access::eColorAttachmentWrite);
        vk::ImageMemoryBarrier2 present(stage, access, Stage::eNone, {}, layout, vk::ImageLayout::ePresentSrcKHR,
                                        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, colorImage,
                                        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, present));
    }

    cmd.end();

    vk::SemaphoreSubmitInfo waitInfo(frame.imageAvailable, 0, Stage::eColorAttachmentOutput);
    std::array<vk::SemaphoreSubmitInfo, 2> signalInfos = {
        vk::SemaphoreSubmitInfo(frame.renderComplete, 0, Stage::eAllCommands),
        vk::SemaphoreSubmitInfo(base().frameTimeline, frame.timelineValue, Stage::eAllCommands)};
    vk::CommandBufferSubmitInfo cmdInfo(cmd);

    vk::SubmitInfo2 submitInfo;
    submitInfo.waitSemaphoreInfoCount   = 1;
    submitInfo.pWaitSemaphoreInfos      = &waitInfo;
    submitInfo.commandBufferInfoCount   = 1;
    submitInfo.pCommandBufferInfos      = &cmdInfo;
    submitInfo.signalSemaphoreInfoCount = signalInfos.size();
    submitInfo.pSignalSemaphoreInfos    = signalInfos.data();
    base().mainQueue.submit2(submitInfo);

    if (screenshot)
    {
        base().waitFrameValue(frame.timelineValue);
        saveScreenshot(*screenshot, screenshot_file);
    }
    if (close_after_frame) window.close();
}

std::unique_ptr<StagingBuffer> VulkanForwardRenderer::recordScreenshot(vk::CommandBuffer cmd, vk::Image colorImage)
{
    if (!(swapChain.imageUsage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
    {
        std::cerr << "Screenshot not possible: the swap chain images do not support TRANSFER_SRC" << std::endl;
        return nullptr;
    }
    auto format = getColorFormat();
    if (format != vk::Format::eB8G8R8A8Unorm && format != vk::Format::eB8G8R8A8Srgb &&
        format != vk::Format::eR8G8B8A8Unorm && format != vk::Format::eR8G8B8A8Srgb)
    {
        std::cerr << "Screenshot not possible: unsupported swap chain format " << vk::to_string(format) << std::endl;
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

void VulkanForwardRenderer::saveScreenshot(StagingBuffer& buffer, const std::string& file)
{
    TemplatedImage<ucvec4> img(SurfaceHeight, surfaceWidth);
    buffer.download(img.data());
    auto format = getColorFormat();
    if (format == vk::Format::eB8G8R8A8Unorm || format == vk::Format::eB8G8R8A8Srgb)
    {
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x) std::swap(img(y, x)(0), img(y, x)(2));
    }
    // The swap chain content is not meant to be blended: store it opaque.
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x)(3) = 255;

    if (img.save(file))
        std::cout << "Saved screenshot " << file << std::endl;
    else
        std::cerr << "Could not save screenshot " << file << std::endl;
}

}  // namespace Vulkan
}  // namespace Saiga
