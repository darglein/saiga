/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "depthBuffer.h"
#include "vulkanHelper.h"

namespace Saiga {
namespace Vulkan {

DepthBuffer::~DepthBuffer()
{
    device.destroyImage(depthimage);
    device.destroyImageView(depthview);
}

void DepthBuffer::init(VulkanBase &base, int width, int height)
{
    vk::Result res;
    {
        // depth buffer
        vk::ImageCreateInfo image_info = {};
        const vk::Format depth_format = vk::Format::eD16Unorm;
        vk::FormatProperties props;
//        vkGetPhysicalDeviceFormatProperties(info.gpus[0], depth_format, &props);
        props = base.physicalDevice.getFormatProperties(depth_format);

        if (props.linearTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            image_info.tiling = vk::ImageTiling::eLinear;
        } else if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            image_info.tiling = vk::ImageTiling::eOptimal;
        } else {
            /* Try other depth formats? */
            std::cout << "VK_FORMAT_D16_UNORM Unsupported.\n";
            exit(-1);
        }
        image_info.imageType = vk::ImageType::e2D;
        image_info.format = depth_format;
        image_info.extent.width =  width;
        image_info.extent.height = height;
        image_info.extent.depth = 1;
        image_info.mipLevels = 1;
        image_info.arrayLayers = 1;
        image_info.samples = vk::SampleCountFlagBits::e1;
        image_info.initialLayout = vk::ImageLayout::eUndefined;
        image_info.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
        image_info.queueFamilyIndexCount = 0;
        image_info.pQueueFamilyIndices = NULL;
        image_info.sharingMode = vk::SharingMode::eExclusive;
//        image_info.flags = 0;



        vk::ImageViewCreateInfo viewInfo = {};

        viewInfo.image = nullptr;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = depth_format;
        viewInfo.components.r = vk::ComponentSwizzle::eR;
        viewInfo.components.g = vk::ComponentSwizzle::eG;
        viewInfo.components.b = vk::ComponentSwizzle::eB;
        viewInfo.components.a = vk::ComponentSwizzle::eA;
        viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        vk::MemoryRequirements mem_reqs;

        depthFormat = depth_format;

        /* Create image */
//        res = vkCreateImage(info.device, &image_info, NULL, &info.depth.image);
        res = base.device.createImage(&image_info,nullptr,&depthimage);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
//        assert(res == VK_SUCCESS);


//        vkGetImageMemoryRequirements(info.device, info.depth.image, &mem_reqs);
        mem_reqs = base.device.getImageMemoryRequirements(depthimage);



        allocateMemory(base,mem_reqs,vk::MemoryPropertyFlagBits::eDeviceLocal);


        base.device.bindImageMemory(depthimage,memory,0);


        viewInfo.image = depthimage;
        res = base.device.createImageView(&viewInfo,nullptr,&depthview);
        SAIGA_ASSERT(res == vk::Result::eSuccess);
    }

}




}
}
