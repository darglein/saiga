/*
 * Vulkan Example base class
 *
 * Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */


#include "VulkanForwardRenderer.h"

#include "saiga/vulkan/Shader/all.h"

#include "VulkanInitializers.hpp"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif


namespace Saiga
{
namespace Vulkan
{
VulkanForwardRenderer::VulkanForwardRenderer(VulkanWindow& window, VulkanParameters vulkanParameters)
    : VulkanRenderer(window, vulkanParameters)
{
    setupRenderPass();
    renderCommandPool = base().mainQueue.createCommandPool(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
    std::cout << "VulkanForwardRenderer init done." << std::endl;
}

VulkanForwardRenderer::~VulkanForwardRenderer()
{
    base().device.destroyRenderPass(renderPass);
}


void VulkanForwardRenderer::createBuffers(int numImages, int w, int h)
{
    depthBuffer.init(base(), w, h);

    frameBuffers.clear();
    frameBuffers.resize(numImages);
    for (int i = 0; i < numImages; i++)
    {
        frameBuffers[i].createColorDepthStencil(w, h, swapChain.buffers[i].view, depthBuffer.location->data.view,
                                                renderPass, base().device);
    }


    renderCommandPool.freeCommandBuffers(drawCmdBuffers);
    drawCmdBuffers.clear();
    drawCmdBuffers = renderCommandPool.allocateCommandBuffers(numImages, vk::CommandBufferLevel::ePrimary);

    if (imGui) imGui->initResources(base(), renderPass);
}


void VulkanForwardRenderer::setupRenderPass()
{
    std::array<VkAttachmentDescription, 2> attachments = {};
    // Color attachment
    attachments[0].format         = swapChain.colorFormat;
    attachments[0].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // Depth attachment
    attachments[1].format         = (VkFormat)depthBuffer.format;
    attachments[1].samples        = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference = {};
    colorReference.attachment            = 0;
    colorReference.layout                = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthReference = {};
    depthReference.attachment            = 1;
    depthReference.layout                = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription    = {};
    subpassDescription.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount    = 1;
    subpassDescription.pColorAttachments       = &colorReference;
    subpassDescription.pDepthStencilAttachment = &depthReference;
    subpassDescription.inputAttachmentCount    = 0;
    subpassDescription.pInputAttachments       = nullptr;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments    = nullptr;
    subpassDescription.pResolveAttachments     = nullptr;

    // Subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;

    dependencies[0].srcSubpass      = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass      = 0;
    dependencies[0].srcStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass      = 0;
    dependencies[1].dstSubpass      = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask    = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask    = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask   = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask   = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType                  = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount        = 2;  // static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments           = attachments.data();
    renderPassInfo.subpassCount           = 1;
    renderPassInfo.pSubpasses             = &subpassDescription;
    renderPassInfo.dependencyCount        = 1;  // static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies          = dependencies.data();

    VK_CHECK_RESULT(vkCreateRenderPass(base().device, &renderPassInfo, nullptr, &renderPass));
}



void VulkanForwardRenderer::render(FrameSync& sync, int currentImage)
{
    VulkanForwardRenderingInterface* renderingInterface = dynamic_cast<VulkanForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    //    std::cout << "VulkanForwardRenderer::render" << std::endl;
    if (imGui && should_render_imgui)
    {
        //        std::thread t([&](){
        imGui->beginFrame();
        renderingInterface->renderGUI();
        imGui->endFrame();
        //        });
        //        t.join();
    }



    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
    //    cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkClearValue clearValues[2];

    // This is blender's default viewport background color :)
    vec4 clearColor             = vec4(57, 57, 57, 255) / 255.0f;
    clearValues[0].color        = {{clearColor[0], clearColor[1], clearColor[2], clearColor[3]}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo    = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass               = renderPass;
    renderPassBeginInfo.renderArea.offset.x      = 0;
    renderPassBeginInfo.renderArea.offset.y      = 0;
    renderPassBeginInfo.renderArea.extent.width  = surfaceWidth;
    renderPassBeginInfo.renderArea.extent.height = SurfaceHeight;
    renderPassBeginInfo.clearValueCount          = 2;
    renderPassBeginInfo.pClearValues             = clearValues;


    vk::CommandBuffer& cmd = drawCmdBuffers[currentImage];
    // cmd.reset(vk::CommandBufferResetFlagBits::eReleaseResources);
    // Set target frame buffer
    renderPassBeginInfo.framebuffer = frameBuffers[currentImage].framebuffer;

    cmd.begin(cmdBufInfo);
    timings.resetFrame(cmd);
    timings.enterSection("TRANSFER", cmd);

    // VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));
    renderingInterface->transfer(cmd);

    timings.leaveSection("TRANSFER", cmd);


    if (imGui && should_render_imgui) imGui->updateBuffers(cmd, currentImage);

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = vks::initializers::viewport((float)surfaceWidth, (float)SurfaceHeight, 0.0f, 1.0f);
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(surfaceWidth, SurfaceHeight, 0, 0);
    vkCmdSetScissor(cmd, 0, 1, &scissor);


    {
        // Actual rendering
        timings.enterSection("MAIN", cmd);
        renderingInterface->render(cmd);
        timings.leaveSection("MAIN", cmd);
        timings.enterSection("IMGUI", cmd);
        if (imGui && should_render_imgui) imGui->render(cmd, currentImage);
        timings.leaveSection("IMGUI", cmd);
    }

    vkCmdEndRenderPass(cmd);


    VK_CHECK_RESULT(vkEndCommandBuffer(cmd));

    vk::PipelineStageFlags submitPipelineStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    std::array<vk::Semaphore, 2> signalSemaphores{sync.renderComplete, sync.defragMayStart};

    vk::SubmitInfo submitInfo;
    //    submitInfo = vks::initializers::submitInfo();
    submitInfo.pWaitDstStageMask    = &submitPipelineStages;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = &sync.imageAvailable;
    submitInfo.signalSemaphoreCount = 2;
    submitInfo.pSignalSemaphores    = signalSemaphores.data();

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;
    //    VK_CHECK_RESULT(vkQueueSubmit(graphicsQueue, 1, &submitInfo, sync.frameFence));
    base().mainQueue.submit(submitInfo, sync.frameFence);

    timings.finishFrame(sync.defragMayStart);
    //    graphicsQueue.queue.submit(submitInfo,vk::Fence());

    //    VK_CHECK_RESULT(swapChain.queuePresent(presentQueue, currentBuffer,  sync.renderComplete));
    base().finish_frame();
}



}  // namespace Vulkan
}  // namespace Saiga
