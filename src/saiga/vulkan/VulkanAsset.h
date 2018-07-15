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
#include "saiga/vulkan/buffer/IndexBuffer.h"
#include "saiga/vulkan/texture/Texture.h"

namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL VulkanVertexColoredAsset : public VertexColoredModel
{
public:
    VertexBuffer<VertexType> vertexBuffer;
    IndexBuffer<uint32_t> indexBuffer;

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


class SAIGA_GLOBAL VulkanTexturedAsset : public TexturedModel
{
public:
    VertexBuffer<VertexType> vertexBuffer;
    IndexBuffer<uint32_t> indexBuffer;
    std::vector<std::shared_ptr<Texture2D>> textures;
    vk::DescriptorSet descriptor;

    void render(vk::CommandBuffer cmd);
    void updateBuffer(Saiga::Vulkan::VulkanBase& base);
    void destroy();
};




}
}
