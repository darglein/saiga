/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/vulkan/vulkan.h"
#include "saiga/opengl/vertex.h"

namespace Saiga {
namespace Vulkan {



template<typename VertexType>
class VKVertexAttribBinder
{
public:
    void getVKAttribs(vk::VertexInputBindingDescription& vi_binding, std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);
};




template<>
void VKVertexAttribBinder<Vertex>::getVKAttribs(vk::VertexInputBindingDescription& vi_binding, std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);


template<>
void VKVertexAttribBinder<VertexNC>::getVKAttribs(vk::VertexInputBindingDescription& vi_binding, std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);

}
}
