/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL UniformBuffer
{
public:


    glm::mat4 Projection;
    glm::mat4 View;
    glm::mat4 Model;
    glm::mat4 Clip;
    glm::mat4 MVP;
    vk::Buffer uniformbuf;
    vk::DeviceMemory uniformmem;
    vk::DescriptorBufferInfo uniformbuffer_info;

    void init(VulkanBase& base);
};

}
}
