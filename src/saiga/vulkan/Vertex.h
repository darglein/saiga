/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/core/geometry/vertex.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
template <typename VertexType>
class SAIGA_TEMPLATE VKVertexAttribBinder
{
   public:
    void getVKAttribs(vk::VertexInputBindingDescription& vi_binding,
                      std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);

    vk::PipelineVertexInputStateCreateInfo createPipelineVertexInputInfo();
};


template <typename VertexType>
vk::PipelineVertexInputStateCreateInfo VKVertexAttribBinder<VertexType>::createPipelineVertexInputInfo()
{
    vk::VertexInputBindingDescription vi_binding;
    std::vector<vk::VertexInputAttributeDescription> vi_attribs;

    getVKAttribs(vi_binding, vi_attribs);

    vk::PipelineVertexInputStateCreateInfo vi;

    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &vi_binding;
    vi.vertexAttributeDescriptionCount = vi_attribs.size();
    vi.pVertexAttributeDescriptions    = vi_attribs.data();
    return vi;
}



template <>
SAIGA_VULKAN_API void VKVertexAttribBinder<VertexNC>::getVKAttribs(
    vk::VertexInputBindingDescription& vi_binding,
    std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);

template <>
SAIGA_VULKAN_API void VKVertexAttribBinder<VertexNT>::getVKAttribs(
    vk::VertexInputBindingDescription& vi_binding,
    std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors);


}  // namespace Vulkan
}  // namespace Saiga
