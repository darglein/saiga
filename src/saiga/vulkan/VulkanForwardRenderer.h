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

#include "saiga/vulkan/window/Window.h"
#include "saiga/vulkan/Renderer.h"

#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/buffer/DepthBuffer.h"
#include "saiga/vulkan/buffer/Framebuffer.h"



namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL VulkanForwardRenderer : public Saiga::Vulkan::VulkanRenderer
{
public:
    VkQueue queue;
    VkRenderPass renderPass;

    VulkanForwardRenderer(Saiga::Vulkan::VulkanWindow& window, bool enableValidation = true);
    virtual ~VulkanForwardRenderer();

    virtual void render(Camera* cam);
protected:


    DepthBuffer depthBuffer;

    VkCommandPool cmdPool;

    std::vector<VkCommandBuffer> drawCmdBuffers;
    std::vector<Framebuffer>frameBuffers;
    uint32_t currentBuffer = 0;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;


    std::vector<FrameSync> syncObjects;
    unsigned int nextSyncObject = 0;

    std::shared_ptr<Saiga::Vulkan::ImGuiVulkanRenderer> imGui;

    void setupRenderPass();
};

}
}
