/*
* Vulkan Example base class
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/


#include "VulkanForwardRenderer.h"
#include "saiga/vulkan/Shader/all.h"
#include "VulkanDebug.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif


namespace Saiga {
namespace Vulkan {


bool VulkanForwardRenderer::checkCommandBuffers()
{
    for (auto& cmdBuffer : drawCmdBuffers)
    {
        if (cmdBuffer == VK_NULL_HANDLE)
        {
            return false;
        }
    }
    return true;
}

void VulkanForwardRenderer::createCommandBuffers()
{
    // Create one command buffer for each swap chain image and reuse for rendering
    drawCmdBuffers.resize(swapChain.imageCount);

    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
            vks::initializers::commandBufferAllocateInfo(
                cmdPool,
                VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                static_cast<uint32_t>(drawCmdBuffers.size()));

    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, drawCmdBuffers.data()));
}

void VulkanForwardRenderer::destroyCommandBuffers()
{
    vkFreeCommandBuffers(device, cmdPool, static_cast<uint32_t>(drawCmdBuffers.size()), drawCmdBuffers.data());
}

VkCommandBuffer VulkanForwardRenderer::createCommandBuffer(VkCommandBufferLevel level, bool begin)
{
    VkCommandBuffer cmdBuffer;

    VkCommandBufferAllocateInfo cmdBufAllocateInfo =
            vks::initializers::commandBufferAllocateInfo(
                cmdPool,
                level,
                1);

    VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &cmdBuffer));

    // If requested, also start the new command buffer
    if (begin)
    {
        VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
        VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
    }

    return cmdBuffer;
}

void VulkanForwardRenderer::flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free)
{
    if (commandBuffer == VK_NULL_HANDLE)
    {
        return;
    }

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK_RESULT(vkQueueWaitIdle(queue));

    if (free)
    {
        vkFreeCommandBuffers(device, cmdPool, 1, &commandBuffer);
    }
}

void VulkanForwardRenderer::createPipelineCache()
{
    VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
    pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));
}





void VulkanForwardRenderer::render(Camera *cam)
{
    VulkanForwardRenderingInterface* renderingInterface = dynamic_cast<VulkanForwardRenderingInterface*>(rendering);
    SAIGA_ASSERT(renderingInterface);

    //    cout << "VulkanForwardRenderer::render" << endl;
    imGui->beginFrame();
    renderingInterface->renderGUI();
    imGui->endFrame();



    FrameSync& sync = syncObjects[nextSyncObject];

    sync.wait(device);


    VkResult err = swapChain.acquireNextImage(sync.presentComplete, &currentBuffer);
    VK_CHECK_RESULT(err);

    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &sync.presentComplete;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &sync.renderComplete;


    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    vec4 clearColor(0.4,0.8,1.0,1.0);
    clearValues[0].color = { { clearColor.x,clearColor.y,clearColor.z,clearColor.w} };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;


    VkCommandBuffer& cmd = drawCmdBuffers[currentBuffer];

    // Set target frame buffer
    renderPassBeginInfo.framebuffer = frameBuffers[currentBuffer];

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &cmdBufInfo));

    vkCmdBeginRenderPass(cmd, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
    vkCmdSetScissor(cmd, 0, 1, &scissor);


    {
        // Actual rendering
        renderingInterface->render(cmd);
        imGui->render(cmd);
    }

    vkCmdEndRenderPass(cmd);

    VK_CHECK_RESULT(vkEndCommandBuffer(cmd));


    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, sync.frameFence));

    VK_CHECK_RESULT(swapChain.queuePresent(queue, currentBuffer,  sync.renderComplete));
    VK_CHECK_RESULT(vkQueueWaitIdle(queue));

    nextSyncObject = (nextSyncObject+1) % syncObjects.size();
}


void VulkanForwardRenderer::prepareFrame()
{

}

void VulkanForwardRenderer::submitFrame()
{



}

VulkanForwardRenderer::VulkanForwardRenderer(VulkanWindow &window, bool enableValidation)
    : VulkanRenderer(window), window(window)
{
    settings.validation = enableValidation;

    width = window.getWidth();
    height = window.getHeight();

    initVulkan();

    if (vulkanDevice->enableDebugMarkers) {
        vks::debugmarker::setup(device);
    }
    initSwapchain();
    createCommandPool();
    setupSwapChain();
    createCommandBuffers();
    createSynchronizationPrimitives();
    setupDepthStencil();
    setupRenderPass();
    createPipelineCache();
    setupFrameBuffer();
    cout << "VulkanForwardRenderer init done." << endl;

    imGui = window.createImGui();
    imGui->initResources(vulkanDevice,renderPass, queue);
}

VulkanForwardRenderer::~VulkanForwardRenderer()
{
    // Clean up Vulkan resources
    swapChain.cleanup();
    if (descriptorPool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }
    destroyCommandBuffers();
    vkDestroyRenderPass(device, renderPass, nullptr);
    for (uint32_t i = 0; i < frameBuffers.size(); i++)
    {
        vkDestroyFramebuffer(device, frameBuffers[i], nullptr);
    }
    vkDestroyImageView(device, depthStencil.view, nullptr);
    vkDestroyImage(device, depthStencil.image, nullptr);
    vkFreeMemory(device, depthStencil.mem, nullptr);

    vkDestroyPipelineCache(device, pipelineCache, nullptr);

    vkDestroyCommandPool(device, cmdPool, nullptr);

    for(auto& s : syncObjects)
    {
        s.destroy(device);
    }
    //    vkDestroySemaphore(device, semaphores.presentComplete, nullptr);
    //    vkDestroySemaphore(device, semaphores.renderComplete, nullptr);
    //    for (auto& fence : waitFences) {
    //        vkDestroyFence(device, fence, nullptr);
    //    }

    delete vulkanDevice;

    if (settings.validation)
    {
        vks::debug::freeDebugCallback(instance);
    }

    vkDestroyInstance(instance, nullptr);

}

bool VulkanForwardRenderer::initVulkan()
{
    VkResult err;

    std::vector<const char*> instanceExtensions = window.getRequiredInstanceExtensions();
    instanceExtensions.push_back( VK_KHR_SURFACE_EXTENSION_NAME );
    instance.create(instanceExtensions,true);



    // Physical device
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr));
    assert(gpuCount > 0);
    // Enumerate devices
    std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
    err = vkEnumeratePhysicalDevices(instance, &gpuCount, physicalDevices.data());
    if (err) {
        vks::tools::exitFatal("Could not enumerate physical devices : \n" + vks::tools::errorString(err), err);
        return false;
    }

    // GPU selection

    // Select physical device to be used for the Vulkan example
    // Defaults to the first device unless specified by command line
    uint32_t selectedDevice = 0;



    physicalDevice = physicalDevices[selectedDevice];

    {
        VkPhysicalDeviceProperties deviceProperties;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        std::cout << "Device [" << selectedDevice << "] : " << deviceProperties.deviceName << std::endl;
        std::cout << " Type: " << vks::tools::physicalDeviceTypeString(deviceProperties.deviceType) << std::endl;
        std::cout << " API: " << (deviceProperties.apiVersion >> 22) << "." << ((deviceProperties.apiVersion >> 12) & 0x3ff) << "." << (deviceProperties.apiVersion & 0xfff) << std::endl;
    }

    // Store properties (including limits), features and memory properties of the phyiscal device (so that examples can check against them)
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);



    // Vulkan device creation
    // This is handled by a separate class that gets a logical device representation
    // and encapsulates functions related to a device
    vulkanDevice = new vks::VulkanDevice(physicalDevice);
    VkResult res = vulkanDevice->createLogicalDevice(enabledFeatures, enabledDeviceExtensions);
    if (res != VK_SUCCESS) {
        vks::tools::exitFatal("Could not create Vulkan device: \n" + vks::tools::errorString(res), res);
        return false;
    }
    device = vulkanDevice->logicalDevice;

    // Get a graphics queue from the device
    vkGetDeviceQueue(device, vulkanDevice->queueFamilyIndices.graphics, 0, &queue);

    // Find a suitable depth format
    VkBool32 validDepthFormat = vks::tools::getSupportedDepthFormat(physicalDevice, &depthFormat);
    assert(validDepthFormat);

    swapChain.connect(instance, physicalDevice, device);

    // Set up submit info structure
    // Semaphores will stay the same during application lifetime
    // Command buffer submission info is set by each example
    submitInfo = vks::initializers::submitInfo();
    submitInfo.pWaitDstStageMask = &submitPipelineStages;
    //    submitInfo.waitSemaphoreCount = 1;
    //    submitInfo.pWaitSemaphores = &semaphores.presentComplete;
    //    submitInfo.signalSemaphoreCount = 1;
    //    submitInfo.pSignalSemaphores = &semaphores.renderComplete;


    return true;
}




void VulkanForwardRenderer::createSynchronizationPrimitives()
{


    int numFrames = drawCmdBuffers.size();

    syncObjects.resize(numFrames);
    for (auto& sync : syncObjects)
    {
        sync.create(device);
    }


}

void VulkanForwardRenderer::createCommandPool()
{
    VkCommandPoolCreateInfo cmdPoolInfo = {};
    cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmdPoolInfo.queueFamilyIndex = swapChain.queueNodeIndex;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool));
}

void VulkanForwardRenderer::setupDepthStencil()
{
    VkImageCreateInfo image = {};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.pNext = NULL;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = depthFormat;
    image.extent = { width, height, 1 };
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    image.flags = 0;

    VkMemoryAllocateInfo mem_alloc = {};
    mem_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mem_alloc.pNext = NULL;
    mem_alloc.allocationSize = 0;
    mem_alloc.memoryTypeIndex = 0;

    VkImageViewCreateInfo depthStencilView = {};
    depthStencilView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depthStencilView.pNext = NULL;
    depthStencilView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format = depthFormat;
    depthStencilView.flags = 0;
    depthStencilView.subresourceRange = {};
    depthStencilView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    depthStencilView.subresourceRange.baseMipLevel = 0;
    depthStencilView.subresourceRange.levelCount = 1;
    depthStencilView.subresourceRange.baseArrayLayer = 0;
    depthStencilView.subresourceRange.layerCount = 1;

    VkMemoryRequirements memReqs;

    VK_CHECK_RESULT(vkCreateImage(device, &image, nullptr, &depthStencil.image));
    vkGetImageMemoryRequirements(device, depthStencil.image, &memReqs);
    mem_alloc.allocationSize = memReqs.size;
    mem_alloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(device, &mem_alloc, nullptr, &depthStencil.mem));
    VK_CHECK_RESULT(vkBindImageMemory(device, depthStencil.image, depthStencil.mem, 0));

    depthStencilView.image = depthStencil.image;
    VK_CHECK_RESULT(vkCreateImageView(device, &depthStencilView, nullptr, &depthStencil.view));
}

void VulkanForwardRenderer::setupFrameBuffer()
{
    VkImageView attachments[2];

    // Depth/Stencil attachment is the same for all frame buffers
    attachments[1] = depthStencil.view;

    VkFramebufferCreateInfo frameBufferCreateInfo = {};
    frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.pNext = NULL;
    frameBufferCreateInfo.renderPass = renderPass;
    frameBufferCreateInfo.attachmentCount = 2;
    frameBufferCreateInfo.pAttachments = attachments;
    frameBufferCreateInfo.width = width;
    frameBufferCreateInfo.height = height;
    frameBufferCreateInfo.layers = 1;

    // Create frame buffers for every swap chain image
    frameBuffers.resize(swapChain.imageCount);
    for (uint32_t i = 0; i < frameBuffers.size(); i++)
    {
        attachments[0] = swapChain.buffers[i].view;
        VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &frameBuffers[i]));
    }
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
    attachments[1].format = depthFormat;
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
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpassDescription;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));
}



void VulkanForwardRenderer::initSwapchain()
{

    //    swapChain.initSurface(sdl_window);
    VkSurfaceKHR surface;
    window.createSurface(instance,&surface);
    swapChain.initSurface(surface);

}

void VulkanForwardRenderer::setupSwapChain()
{
    swapChain.create(&width, &height, settings.vsync);
}


}
}
