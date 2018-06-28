/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL VertexBuffer : public Buffer
{
public:

//    vk::Buffer buffer;
//    vk::DeviceMemory memory;
//    vk::DescriptorBufferInfo info;

    vk::VertexInputBindingDescription vi_binding;
    std::vector<vk::VertexInputAttributeDescription> vi_attribs;

    void init(VulkanBase& base);
};

}
}
