/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkan.h"
#include "saiga/vulkan/vulkanBase.h"
#include "saiga/vulkan/buffer/DeviceMemory.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL DepthBuffer : public DeviceMemory
{
public:
//    vk::DeviceMemory depthmem;

    //depth image
    vk::Format depthFormat;
    vk::Image depthimage;
    vk::ImageView depthview;

    ~DepthBuffer();
    void init(VulkanBase& base, int width, int height);
};

}
}
