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
#include <saiga/vulkan/buffer/StagingBuffer.h>

#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/DeviceMemory.h"
#include "saiga/image/image.h"
#include "saiga/vulkan/AsyncCommand.h"

namespace Saiga{
namespace Vulkan{

struct SAIGA_GLOBAL Texture
{

//    VulkanBase *base;
    MemoryLocation memoryLocation;
    vk::Image image;
    vk::ImageLayout imageLayout;
    vk::ImageView imageView;
    uint32_t width, height;
    uint32_t mipLevels;
    uint32_t layerCount;
    vk::Sampler sampler;

//    ~Texture();
    void destroy(VulkanBase& base);

    void transitionImageLayout(vk::CommandBuffer cmd, vk::ImageLayout newLayout);

    vk::DescriptorImageInfo getDescriptorInfo();


};

struct SAIGA_GLOBAL Texture2D : public Texture
{
    AsyncCommand fromStagingBuffer(VulkanBase &base, uint32_t width, uint32_t height, vk::Format format,
                                    Saiga::Vulkan::StagingBuffer &stagingBuffer, Queue &queue, CommandPool &pool,
                                    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled);
    void fromImage(VulkanBase& base, Image &img, vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled);
    void fromImage(VulkanBase& base, Image &img, Queue& queue, CommandPool& pool, vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled);

    void uploadImage(VulkanBase& base, Image &img);
};

}
}
