/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/Base.h"

namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API Framebuffer
{
   public:
    VkFramebuffer framebuffer = nullptr;

    ~Framebuffer() { destroy(); }
    void destroy();

    /**
     * Creates a framebuffer with one color attachment and a depth-stencil attachment.
     * This is usefull as a default framebuffer with the color attachment being the swap-chain image.
     */
    void createColorDepthStencil(int width, int height, vk::ImageView color, vk::ImageView depthStencil,
                                 vk::RenderPass renderPass, vk::Device device);
    void createColor(int width, int height, vk::ImageView color, vk::RenderPass renderPass, vk::Device device);
    void create(int width, int height, vk::RenderPass renderPass, vk::Device device);

   private:
    vk::Device device;
};


}  // namespace Vulkan
}  // namespace Saiga
