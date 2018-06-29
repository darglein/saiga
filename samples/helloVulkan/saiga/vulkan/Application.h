/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "saiga/vulkan/window.h"
#include "saiga/vulkan/swapChain.h"
#include "saiga/vulkan/CommandPool.h"


#include "saiga/vulkan/ForwardRenderer.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Application : public VulkanBase
{
public:

    Window window;
    SwapChain* swapChain;

    ForwardRenderer forwardRenderer;

    CommandPool mainCommandPool;
    vk::CommandBuffer cmd;

    Application(int width, int height);
    ~Application();

    void run();

    virtual void update() = 0;
    virtual void render(vk::CommandBuffer& cmd) = 0;
};

}
}
