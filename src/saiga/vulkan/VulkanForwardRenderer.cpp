/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanForwardRenderer.h"

#include "saiga/core/image/managedImage.h"
#include "saiga/core/util/env.h"
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

VulkanForwardRenderer::VulkanForwardRenderer(VulkanWindow& window, VulkanParameters vulkanParameters,
                                             const ForwardRenderingParameters& parameters)
    : VulkanRenderer(window, vulkanParameters, {"TRANSFER", "MAIN", "IMGUI"}), params(parameters)
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

    if (imGui) imGui->initResources(base(), renderingFormats());

    if (auto file = GetEnvIfSet("SAIGA_SCREENSHOT"))
    {
        autoScreenshotFile = *file;
    }
    std::cout << "VulkanForwardRenderer init done." << std::endl;
}

VulkanForwardRenderer::~VulkanForwardRenderer()
{
    waitIdle();
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

    bool draw_imgui = imguiFrame([renderingInterface]() { renderingInterface->renderGUI(); });

    renderedFrames++;
    bool close_after_frame = false;
    if (!autoScreenshotFile.empty() && renderedFrames == 60)
    {
        requestScreenshot(autoScreenshotFile);
        close_after_frame = true;
    }

    vk::CommandBuffer cmd = beginCommands(frame);

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

    std::vector<ScreenshotCallback> screenshot_callbacks;
    std::unique_ptr<StagingBuffer> screenshot = beginScreenshots(cmd, colorImage, screenshot_callbacks);
    recordPresentTransition(cmd, colorImage, screenshot != nullptr);
    submitCommands(cmd, frame);
    finishScreenshots(frame, screenshot.get(), screenshot_callbacks);
    if (close_after_frame) window.close();
}

}  // namespace Vulkan
}  // namespace Saiga
