/*
 * Vulkan Example base class
 *
 * Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/FrameSync.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/Renderer.h"
#include "saiga/vulkan/buffer/DepthBuffer.h"
#include "saiga/vulkan/buffer/Framebuffer.h"
#include "saiga/vulkan/window/Window.h"


namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API VulkanForwardRenderingInterface : public RenderingInterface
{
   public:
    virtual void transfer(vk::CommandBuffer cmd) {}
    virtual void render(vk::CommandBuffer cmd) {}
    virtual void renderGUI() {}
};

struct SAIGA_OPENGL_API ForwardRenderingParameters
{
    void fromConfigFile(const std::string& file) {}
};

class SAIGA_VULKAN_API VulkanForwardRenderer : public VulkanRenderer
{
   public:
    using InterfaceType = VulkanForwardRenderingInterface;
    using ParameterType = ForwardRenderingParameters;


    CommandPool renderCommandPool;
    VkRenderPass renderPass;

    VulkanForwardRenderer(VulkanWindow& window, VulkanParameters vulkanParameters);
    virtual ~VulkanForwardRenderer() override;

    virtual void render(FrameSync& sync, int currentImage) override;

    virtual void createBuffers(int numImages, int w, int h) override;
    void setupRenderPass();

   protected:
    DepthBuffer depthBuffer;
    std::vector<vk::CommandBuffer> drawCmdBuffers;
    std::vector<Framebuffer> frameBuffers;
};

}  // namespace Vulkan
}  // namespace Saiga
