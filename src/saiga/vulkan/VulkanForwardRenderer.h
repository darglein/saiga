/*
* Vulkan Example base class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "saiga/util/assert.h"
#include "saiga/util/glm.h"


#include "VulkanTools.h"

#include "VulkanInitializers.hpp"

#include "saiga/vulkan/window/Window.h"
#include "saiga/vulkan/Renderer.h"

#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/buffer/DepthBuffer.h"



namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL VulkanForwardRenderer : public Saiga::Vulkan::VulkanRenderer
{

public:


    // Handle to the device graphics queue that command buffers are submitted to
    VkQueue queue;
    // Depth buffer format (selected during Vulkan initialization)
//    VkFormat depthFormat;

    DepthBuffer depthBuffer;

    // Command buffer pool
    VkCommandPool cmdPool;
    /** @brief Pipeline stages used to wait at for graphics queue submissions */
    VkPipelineStageFlags submitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Contains command buffers and semaphores to be presented to the queue
    VkSubmitInfo submitInfo;
    // Command buffers used for rendering
    std::vector<VkCommandBuffer> drawCmdBuffers;
    // Global render pass for frame buffer writes
    VkRenderPass renderPass;
    // List of available frame buffers (same as number of swap chain images)
    std::vector<VkFramebuffer>frameBuffers;
    // Active frame buffer index
    uint32_t currentBuffer = 0;
    // Descriptor set pool
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    // List of shader modules created (stored for cleanup)
    //	std::vector<VkShaderModule> shaderModules;
    // Pipeline cache object
    VkPipelineCache pipelineCache;
    // Wraps the swap chain to present images (framebuffers) to the windowing system

    // Synchronization semaphores

    std::vector<FrameSync> syncObjects;
    unsigned int nextSyncObject = 0;

    std::shared_ptr<Saiga::Vulkan::ImGuiVulkanRenderer> imGui;
public:


    VulkanForwardRenderer(Saiga::Vulkan::VulkanWindow& window, bool enableValidation = true);

    // dtor
    virtual ~VulkanForwardRenderer();



    virtual void render(Camera* cam);



    void createSynchronizationPrimitives();

    // Creates a new (graphics) command pool object storing command buffers
    void createCommandPool();

    // Create framebuffers for all requested swap chain images
    // Can be overriden in derived class to setup a custom framebuffer (e.g. for MSAA)
    virtual void setupFrameBuffer();
    // Setup a default render pass
    // Can be overriden in derived class to setup a custom render pass (e.g. for MSAA)
    virtual void setupRenderPass();


    // Create command buffers for drawing commands
    void createCommandBuffers();
    // Destroy all command buffers and set their handles to VK_NULL_HANDLE
    // May be necessary during runtime if options are toggled
    void destroyCommandBuffers();

    // Command buffer creation
    // Creates and returns a new command buffer
    VkCommandBuffer createCommandBuffer(VkCommandBufferLevel level, bool begin);
    // End the command buffer, submit it to the queue and free (if requested)
    // Note : Waits for the queue to become idle
    void flushCommandBuffer(VkCommandBuffer commandBuffer, VkQueue queue, bool free);

    // Create a cache pool for rendering pipelines
    void createPipelineCache();



};

}
}
