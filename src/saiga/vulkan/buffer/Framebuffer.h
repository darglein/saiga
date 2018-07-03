/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Device.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL Framebuffer
{
public:
    VkFramebuffer framebuffer;

    void destroy(vk::Device device);

    /**
     * Creates a framebuffer with one color attachment and a depth-stencil attachment.
     * This is usefull as a default framebuffer with the color attachment being the swap-chain image.
     */
    void createColorDepthStencil(int width, int height, vk::ImageView color, vk::ImageView depthStencil, vk::RenderPass renderPass, vk::Device device);
};

}
}
