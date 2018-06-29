/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "saiga/vulkan/window.h"
#include "saiga/vulkan/swapChain.h"
#include "saiga/vulkan/CommandPool.h"

#include "saiga/vulkan/buffer/depthBuffer.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL ForwardRenderer
{
public:

    VulkanBase& base;

    DepthBuffer depthBuffer;

    vk::RenderPass render_pass;

    std::vector<vk::Framebuffer> framebuffers;

    ForwardRenderer(VulkanBase& base);
    ~ForwardRenderer();

    void create(SwapChain& swapChain, int width, int height);

    void begin(vk::CommandBuffer& cmd);
    void end();
};

}
}
