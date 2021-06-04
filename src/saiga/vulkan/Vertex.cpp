/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Vertex.h"

namespace Saiga
{
namespace Vulkan
{
template <>
void VKVertexAttribBinder<VertexNC>::getVKAttribs(
    vk::VertexInputBindingDescription& vi_binding,
    std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors)
{
    vi_binding.binding   = 0;
    vi_binding.inputRate = vk::VertexInputRate::eVertex;
    vi_binding.stride    = sizeof(VertexNC);

    attributeDescriptors.resize(4);

    attributeDescriptors[0].binding  = 0;
    attributeDescriptors[0].location = 0;
    attributeDescriptors[0].format   = vk::Format::eR32G32B32A32Sfloat;
    attributeDescriptors[0].offset   = 0;

    attributeDescriptors[1].binding  = 0;
    attributeDescriptors[1].location = 1;
    attributeDescriptors[1].format   = vk::Format::eR32G32B32A32Sfloat;
    attributeDescriptors[1].offset   = 1 * sizeof(vec4);

    attributeDescriptors[2].binding  = 0;
    attributeDescriptors[2].location = 2;
    attributeDescriptors[2].format   = vk::Format::eR32G32B32A32Sfloat;
    attributeDescriptors[2].offset   = 2 * sizeof(vec4);

    attributeDescriptors[3].binding  = 0;
    attributeDescriptors[3].location = 3;
    attributeDescriptors[3].format   = vk::Format::eR32G32B32A32Sfloat;
    attributeDescriptors[3].offset   = 3 * sizeof(vec4);
}

template <>
void VKVertexAttribBinder<VertexNT>::getVKAttribs(
    vk::VertexInputBindingDescription& vi_binding,
    std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors)
{
    vi_binding.binding   = 0;
    vi_binding.inputRate = vk::VertexInputRate::eVertex;
    vi_binding.stride    = sizeof(VertexNT);

    attributeDescriptors.resize(3);

    attributeDescriptors[0].binding  = 0;
    attributeDescriptors[0].location = 0;
    attributeDescriptors[0].format   = vk::Format::eR32G32B32A32Sfloat;
    attributeDescriptors[0].offset   = 0;

    attributeDescriptors[1].binding  = 0;
    attributeDescriptors[1].location = 1;
    attributeDescriptors[1].format   = vk::Format::eR32G32B32A32Sfloat;
    attributeDescriptors[1].offset   = 1 * sizeof(vec4);

    attributeDescriptors[2].binding  = 0;
    attributeDescriptors[2].location = 2;
    attributeDescriptors[2].format   = vk::Format::eR32G32B32A32Sfloat;
    attributeDescriptors[2].offset   = 2 * sizeof(vec4);
}





}  // namespace Vulkan
}  // namespace Saiga
