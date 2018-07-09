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
    vk::DescriptorImageInfo descriptor;

    /** @brief Optional sampler to use with this texture */
    vk::Sampler sampler;

};

struct SAIGA_GLOBAL Texture2D : public Texture
{

    void fromImage(VulkanBase& base, Image &img);
};

}
}
