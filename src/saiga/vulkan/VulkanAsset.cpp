/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanAsset.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/animation/objLoader2.h"



namespace Saiga {
namespace Vulkan {



void VulkanVertexColoredAsset::render(VkCommandBuffer cmd)
{

    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, offsets);
    vkCmdBindIndexBuffer(cmd, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

}

void VulkanVertexColoredAsset::updateBuffer(vks::VulkanDevice *device, VkQueue copyQueue)
{
    vertexCount = mesh.vertices.size();
    indexCount = mesh.faces.size() * 3;

    uint32_t vBufferSize = vertexCount * sizeof(VertexNC);
    uint32_t iBufferSize = indexCount * sizeof(uint32_t);

    // Use staging buffer to move vertex and index buffer to device local memory
    // Create staging buffers
    vks::Buffer vertexStaging, indexStaging;

    // Vertex buffer
    VK_CHECK_RESULT(device->createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &vertexStaging,
                        vBufferSize,
                        mesh.vertices.data()));

    // Index buffer
    VK_CHECK_RESULT(device->createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &indexStaging,
                        iBufferSize,
                        mesh.faces.data()));

    // Create device local target buffers
    // Vertex buffer
    VK_CHECK_RESULT(device->createBuffer(
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &vertices,
                        vBufferSize));

    // Index buffer
    VK_CHECK_RESULT(device->createBuffer(
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &indices,
                        iBufferSize));

    // Copy from staging buffers
    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkBufferCopy copyRegion{};

    copyRegion.size = vertices.size;
    vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertices.buffer, 1, &copyRegion);

    copyRegion.size = indices.size;
    vkCmdCopyBuffer(copyCmd, indexStaging.buffer, indices.buffer, 1, &copyRegion);

    device->flushCommandBuffer(copyCmd, copyQueue);

    // Destroy staging resources
    vkDestroyBuffer(device->logicalDevice, vertexStaging.buffer, nullptr);
    vkFreeMemory(device->logicalDevice, vertexStaging.memory, nullptr);
    vkDestroyBuffer(device->logicalDevice, indexStaging.buffer, nullptr);
    vkFreeMemory(device->logicalDevice, indexStaging.memory, nullptr);
}



}
}
