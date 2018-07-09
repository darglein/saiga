/*
* Vulkan device class
*
* Encapsulates a physical Vulkan device and it's logical representation
*
* Copyright (C) 2016-2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include <exception>
#include <assert.h>
#include <algorithm>

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/DeviceMemory.h"
#include "saiga/image/image.h"


namespace Saiga{
namespace Vulkan{

struct SAIGA_GLOBAL Texture : public DeviceMemory
{
    VulkanBase *base;
    vk::Image image;
    vk::ImageLayout imageLayout;
    vk::ImageView imageView;
    uint32_t width, height;
    uint32_t mipLevels;
    uint32_t layerCount;
    vk::Sampler sampler;

    void destroy();

    void transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout)
    {


        vk::ImageMemoryBarrier barrier = {};
        barrier.oldLayout = imageLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
//        barrier.srcAccessMask = 0; // TODO
//        barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite; // TODO


//        VkPipelineStageFlags sourceStage;
//        VkPipelineStageFlags destinationStage;

        if (imageLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

//            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
//            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (imageLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

//            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
//            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }



        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,vk::PipelineStageFlagBits::eAllCommands,vk::DependencyFlags(),0,nullptr,0,nullptr,1,&barrier);


        imageLayout = newLayout;
    }

    vk::DescriptorImageInfo getDescriptorInfo()
    {
        vk::DescriptorImageInfo descriptorInfo;
        descriptorInfo.imageLayout = imageLayout;
        descriptorInfo.imageView = imageView;
        descriptorInfo.sampler = sampler;
        return descriptorInfo;

    }


};

struct SAIGA_GLOBAL Texture2D : public Texture
{

    void fromImage(VulkanBase& base, Image &img);
};

}
}
