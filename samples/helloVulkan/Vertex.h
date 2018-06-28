/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "vulkanBase.h"

namespace Saiga {
namespace Vulkan {

struct Vertex
{
    float position[3];
    float color[3];
};



template<typename VertexType>
class VKVertexAttribBinder
{
public:
    void getVKAttribs(vk::VertexInputBindingDescription& vi_binding, std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);
};




template<>
void VKVertexAttribBinder<Vertex>::getVKAttribs(vk::VertexInputBindingDescription& vi_binding, std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);



}
}
