/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkan.h"
#include "saiga/vulkan/vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SwapChain
{
private:
    vk::Instance instance;
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    vk::SurfaceKHR surface;
public:
    struct SwapChainBuffer {
        vk::Image image;
        vk::ImageView view;
    };
    vk::Format colorFormat;
    vk::Format depthFormat;
    vk::ColorSpaceKHR colorSpace;
    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> images;
    std::vector<SwapChainBuffer> buffers;
    uint32_t queueNodeIndex = UINT32_MAX;

    SwapChain();

    ~SwapChain();


    void create(vk::Instance instance, vk::PhysicalDevice physicalDevice, vk::Device device);

    void setSurface(vk::SurfaceKHR _surface);


    void create(uint32_t *width, uint32_t *height, bool vsync = false);

    void acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t &imageIndex);

    void queuePresent(vk::Queue queue, uint32_t imageIndex, vk::Semaphore waitSemaphore = vk::Semaphore());


};
}
}
