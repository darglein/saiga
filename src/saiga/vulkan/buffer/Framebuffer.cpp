/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Framebuffer.h"


namespace Saiga {
namespace Vulkan {

void Framebuffer::destroy(vk::Device device)
{
        vkDestroyFramebuffer(device, framebuffer, nullptr);
}

void Framebuffer::createColorDepthStencil(
        int width, int height,
        vk::ImageView color, vk::ImageView depthStencil,
        vk::RenderPass renderPass, vk::Device device
        )
{
    VkImageView attachments[2];

    // Depth/Stencil attachment is the same for all frame buffers
    //    attachments[1] = depthStencil.view;
    attachments[0] = color;
    attachments[1] = depthStencil;


    VkFramebufferCreateInfo frameBufferCreateInfo = {};
    frameBufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    frameBufferCreateInfo.pNext = NULL;
    frameBufferCreateInfo.renderPass = renderPass;
    frameBufferCreateInfo.attachmentCount = 2;
    frameBufferCreateInfo.pAttachments = attachments;
    frameBufferCreateInfo.width = width;
    frameBufferCreateInfo.height = height;
    frameBufferCreateInfo.layers = 1;


        VK_CHECK_RESULT(vkCreateFramebuffer(device, &frameBufferCreateInfo, nullptr, &framebuffer));

}

}
}
