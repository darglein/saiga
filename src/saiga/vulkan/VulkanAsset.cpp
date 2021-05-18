/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanAsset.h"

#include "saiga/core/image/imageTransformations.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"

namespace Saiga
{
namespace Vulkan
{
void VulkanVertexColoredAsset::init(Saiga::Vulkan::VulkanBase& base)
{
    auto indices = getIndexList();

    vertexBuffer.init(base, vertices.size(), vk::MemoryPropertyFlagBits::eDeviceLocal);
    indexBuffer.init(base, indices.size(), vk::MemoryPropertyFlagBits::eDeviceLocal);

    vertexBuffer.stagedUpload(base, vertices.size() * sizeof(VertexType), vertices.data());
    indexBuffer.stagedUpload(base, indices.size() * sizeof(uint32_t), indices.data());
}


void VulkanVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    //    if(!vertexBuffer.m_memoryLocation.buffer) return;
    vertexBuffer.bind(cmd);
    indexBuffer.bind(cmd);
    indexBuffer.draw(cmd);
}

void VulkanLineVertexColoredAsset::init(VulkanBase& base)
{
    auto lines   = mesh.toLineList();
    auto newSize = lines.size();
    auto size    = newSize * sizeof(VertexType);
    vertexBuffer.init(base, newSize, vk::MemoryPropertyFlagBits::eDeviceLocal);


    vertexBuffer.stagedUpload(base, size, lines.data());
}

void VulkanLineVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd);
}



void VulkanPointCloudAsset::init(VulkanBase& base, int _capacity)
{
    capacity = _capacity;
    vertexBuffer.init(base, capacity, vk::MemoryPropertyFlagBits::eDeviceLocal);
    stagingBuffer.init(base, capacity * sizeof(VertexType));
    pointCloud = ArrayView<VertexType>((VertexType*)stagingBuffer.getMappedPointer(), capacity);
}

void VulkanPointCloudAsset::render(vk::CommandBuffer cmd, int start, int count)
{
    vertexBuffer.bind(cmd);
    vertexBuffer.draw(cmd, count < 0 ? size : count, start);
}

void VulkanPointCloudAsset::updateBuffer(vk::CommandBuffer cmd, int start, int count)
{
    stagingBuffer.copyTo(cmd, vertexBuffer, start * sizeof(VertexType), start * sizeof(VertexType),
                         (count < 0 ? size : count) * sizeof(VertexType));
}



}  // namespace Vulkan
}  // namespace Saiga
