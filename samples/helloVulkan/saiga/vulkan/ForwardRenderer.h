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
    SwapChain& swapChain;

    int width, height;
        uint32_t current_buffer;

    DepthBuffer depthBuffer;

    vk::RenderPass render_pass;

    vk::Semaphore imageAcquiredSemaphore;
    std::vector<vk::Framebuffer> framebuffers;

    ForwardRenderer(VulkanBase& base, SwapChain& swapChain);
    ~ForwardRenderer();

    void create(int width, int height);

    void begin(vk::CommandBuffer& cmd);
    void end(vk::CommandBuffer &cmd);
};

}
}
