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


class SAIGA_GLOBAL DepthBuffer
{
public:
    //depth image
    vk::Format depthFormat;
    vk::Image depthimage;
    vk::DeviceMemory depthmem;
    vk::ImageView depthview;
    void init(VulkanBase& base, int width, int height);
};

}
}
