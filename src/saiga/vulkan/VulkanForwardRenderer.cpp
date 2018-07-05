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
#error OpenGL was included somewhere.
#endif


namespace Saiga {
namespace Vulkan {





VulkanForwardRenderer::VulkanForwardRenderer(VulkanWindow &window, VulkanParameters vulkanParameters)
    : VulkanRenderer(window,vulkanParameters)
{

    //    vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.graphics, 0, &graphicsQueue);
    //    vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.present, 0, &presentQueue);
    graphicsQueue.create(device,vulkanDevice->queueFamilyIndices.graphics);
    presentQueue.create(device,vulkanDevice->queueFamilyIndices.present);
    transferQueue.create(device,vulkanDevice->queueFamilyIndices.transfer);


    depthBuffer.init(vulkanDevice,width,height);


    //    VkCommandPoolCreateInfo cmdPoolInfo = {};
    //    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    //    cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
    //    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    //    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
    //    cmdPool.create(device,);

    syncObjects.resize(swapChain.imageCount);
    for (auto& sync : syncObjects)
    {
        sync.create(device);
    }


    drawCmdBuffers = graphicsQueue.commandPool.allocateCommandBuffers(swapChain.imageCount,vk::CommandBufferLevel::ePrimary);

    //    drawCmdBuffers.resize(swapChain.imageCount);
    //    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
    //            vks::initializers::commandBufferAllocateInfo(
    //                cmdPool,
    //                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    //                static_cast<uint32_t>(drawCmdBuffers.size()));
    //    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, drawCmdBuffers.data()));

    setupRenderPass();


    frameBuffers.resize(swapChain.imageCount);
    for (uint32_t i = 0; i < frameBuffers.size(); i++)
    {
        frameBuffers[i].createColorDepthStencil(width,height,swapChain.buffers[i].view,depthBuffer.depthview,renderPass,device);
        //        frameBuffers[i].createColor(width,height,swapChain.buffers[i].view,renderPass,device);
    }


    cout << "VulkanForwardRenderer init done." << endl;


}

VulkanForwardRenderer::~VulkanForwardRenderer()
{
    graphicsQueue.destroy();
    presentQueue.destroy();
    transferQueue.destroy();

    waitIdle();

    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    vkDestroyRenderPass(device, renderPass, nullptr);
    for (uint32_t i = 0; i < frameBuffers.size(); i++)
    {
        //        vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
        frameBuffers[i].destroy(device);
    }

    depthBuffer.destroy();
    vkDestroyPipelineCache(device, pipelineCache, nullptr);

    //    vkDestroyCommandPool(device, cmdPool, nullptr);

    for(auto& s : syncObjects)
    {
        s.destroy(device);
    }

}

void VulkanForwardRenderer::initChildren()
{
    imGui = window.createImGui();




    VulkanForwardRenderingInterface* renderingInterface = dynamic_cast<VulkanForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    auto cmd = transferQueue.commandPool.allocateCommandBuffer();
    renderingInterface->init(transferQueue,cmd);

    if(imGui)
        imGui->initResources(vulkanDevice,pipelineCache,renderPass, transferQueue,cmd);

    //    cmd.reset(vk::CommandBufferResetFlags());



    transferQueue.waitIdle();
}






void VulkanForwardRenderer::setupRenderPass()
{
    std::array<VkAttachmentDescription, 2> attachments = {};
    // Color attachment
    attachments[0].format = swapChain.colorFormat;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    // Depth attachment
    attachments[1].format = (VkFormat)depthBuffer.depthFormat;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorReference = {};
    colorReference.attachment = 0;
    colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthReference = {};
    depthReference.attachment = 1;
    depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpassDescription = {};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorReference;
    subpassDescription.pDepthStencilAttachment = &depthReference;
    subpassDescription.inputAttachmentCount = 0;
    subpassDescription.pInputAttachments = nullptr;
    subpassDescription.preserveAttachmentCount = 0;
    subpassDescription.pPreserveAttachments = nullptr;
    subpassDescription.pResolveAttachments = nullptr;

    // Subpass dependencies for layout transitions
    std::array<VkSubpassDependency, 2> dependencies;

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 2;//static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = 1;//static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}




void VulkanForwardRenderer::render(Camera *cam)
{
    VulkanForwardRenderingInterface* renderingInterface = dynamic_cast<VulkanForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    //    cout << "VulkanForwardRenderer::render" << endl;
    if(imGui)
    {
//        std::thread t([&](){
            imGui->beginFrame();
            renderingInterface->renderGUI();
            imGui->endFrame();
//        });
//        t.join();
    }



    FrameSync& sync = syncObjects[nextSyncObject];

    sync.wait(device);


    VkResult err = swapChain.acquireNextImage(sync.imageVailable, &currentBuffer);
    VK_CHECK_RESULT(err);




    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
    cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    VkClearValue clearValues[2];
    vec4 clearColor(0.4,0.8,1.0,1.0);
    clearValues[0].color = { { clearColor.x,clearColor.y,clearColor.z,clearColor.w} };
    //    clearValues[0].depthStencil = { 1.0f, 0 };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;


    vk::CommandBuffer& cmd = drawCmdBuffers[currentBuffer];

    // Set target frame buffer
    renderPassBeginInfo.framebuffer = frameBuffers[currentBuffer].framebuffer;

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));

    renderingInterface->transfer(cmd);

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
    vkCmdSetScissor(cmd, 0, 1, &scissor);


    {
        // Actual rendering
                renderingInterface->render(cmd);
        if(imGui) imGui->render(cmd);
    }

    vkCmdEndRenderPass(cmd);


    VK_CHECK_RESULT(vkEndCommandBuffer(cmd));

    vk::PipelineStageFlags submitPipelineStages =  vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo;
    //    submitInfo = vks::initializers::submitInfo();
    submitInfo.pWaitDstStageMask = &submitPipelineStages;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &sync.imageVailable;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &sync.renderComplete;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    //    VK_CHECK_RESULT(vkQueueSubmit(graphicsQueue, 1, &submitInfo, sync.frameFence));
    graphicsQueue.queue.submit(submitInfo,sync.frameFence);

    VK_CHECK_RESULT(swapChain.queuePresent(presentQueue, currentBuffer,  sync.renderComplete));
    //    VK_CHECK_RESULT(vkQueueWaitIdle(presentQueue));
    //    presentQueue.waitIdle();

    nextSyncObject = (nextSyncObject+1) % syncObjects.size();
}

void VulkanForwardRenderer::waitIdle()
{
    graphicsQueue.waitIdle();
    presentQueue.waitIdle();
    transferQueue.waitIdle();
}





}
}
