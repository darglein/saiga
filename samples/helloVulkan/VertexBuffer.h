/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL VertexBuffer
{
public:

    vk::Buffer vertexbuf;
    vk::DeviceMemory vertexmem;
    vk::DescriptorBufferInfo vertexbuffer_info;

    vk::VertexInputBindingDescription vi_binding;
    vk::VertexInputAttributeDescription vi_attribs[2];

    void init(VulkanBase& base);
};

}
}
