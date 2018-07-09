/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/assets/model/Models.h"
#include "saiga/geometry/PointCloud.h"
#include "saiga/vulkan/svulkan.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/VulkanBuffer.hpp"
#include "saiga/vulkan/buffer/VertexBuffer.h"

namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL VulkanVertexColoredAsset : public VertexColoredModel
{
public:
    vk::Device device;
    vks::Buffer vertices;
    vks::Buffer indices;
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;

    void render(vk::CommandBuffer cmd);
    void updateBuffer(Saiga::Vulkan::VulkanBase& base);
    void destroy();
};


class SAIGA_GLOBAL VulkanLineVertexColoredAsset : public LineModelColored
{
public:
    using VertexType = VertexNC;
    VertexBuffer<VertexType> vertexBuffer;
    void render(vk::CommandBuffer cmd);
    void updateBuffer(Saiga::Vulkan::VulkanBase& base);
    void destroy();
};



class SAIGA_GLOBAL VulkanPointCloudAsset
{
public:
    using VertexType = VertexNC;
    PointCloud<VertexType> mesh;
    VertexBuffer<VertexType> vertexBuffer;

    void render(vk::CommandBuffer cmd);
    void updateBuffer(Saiga::Vulkan::VulkanBase& base);
    void destroy();
};



}
}
