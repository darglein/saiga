/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "VulkanAsset.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"

namespace Saiga {
namespace Vulkan {



void VulkanVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    if(!vertices.buffer) return;
    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertices.buffer, offsets);
    vkCmdBindIndexBuffer(cmd, indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
}

void VulkanVertexColoredAsset::updateBuffer(VulkanBase &base)
{
    vertexCount = mesh.vertices.size();
    indexCount = mesh.faces.size() * 3;

    uint32_t vBufferSize = vertexCount * sizeof(VertexNC);
    uint32_t iBufferSize = indexCount * sizeof(uint32_t);

    // Use staging buffer to move vertex and index buffer to device local memory
    // Create staging buffers
    vks::Buffer vertexStaging, indexStaging;

    // Vertex buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &vertexStaging,
                        vBufferSize,
                        mesh.vertices.data()));

    // Index buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &indexStaging,
                        iBufferSize,
                        mesh.faces.data()));

    // Create device local target buffers
    // Vertex buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &vertices,
                        vBufferSize));

    // Index buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &indices,
                        iBufferSize));

    // Copy from staging buffers
//    vk::CommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    auto cmd = base.createAndBeginTransferCommand();


    VkBufferCopy copyRegion{};

    copyRegion.size = vertices.size;
    vkCmdCopyBuffer(cmd, vertexStaging.buffer, vertices.buffer, 1, &copyRegion);

    copyRegion.size = indices.size;
    vkCmdCopyBuffer(cmd, indexStaging.buffer, indices.buffer, 1, &copyRegion);


//    cmd.end();
    base.endTransferWait(cmd);

    // Destroy staging resources
    vkDestroyBuffer(base.device, vertexStaging.buffer, nullptr);
    vkFreeMemory(base.device, vertexStaging.memory, nullptr);
    vkDestroyBuffer(base.device, indexStaging.buffer, nullptr);
    vkFreeMemory(base.device, indexStaging.memory, nullptr);
}

void VulkanVertexColoredAsset::destroy()
{
    vertices.destroy();
    indices.destroy();
    vertexCount = 0;
    indexCount = 0;
}

void VulkanLineVertexColoredAsset::render(vk::CommandBuffer cmd)
{
    if(!vertexBuffer.buffer) return;
    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.buffer, offsets);
    vkCmdDraw(cmd, vertexCount, 1, 0, 0);
}

void VulkanLineVertexColoredAsset::updateBuffer(VulkanBase &base)
{
    vertexBuffer.destroy();

    auto vertices = mesh.toLineList();

    vertexCount = vertices.size();

    uint32_t vBufferSize = vertexCount * sizeof(VertexType);


    // Use staging buffer to move vertex and index buffer to device local memory
    // Create staging buffers
    vks::Buffer vertexStaging;

    // Vertex buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &vertexStaging,
                        vBufferSize,
                        vertices.data()));


    // Create device local target buffers
    // Vertex buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &vertexBuffer,
                        vBufferSize));


    // Copy from staging buffers
    VkCommandBuffer copyCmd = base.createAndBeginTransferCommand();//createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);

    VkBufferCopy copyRegion{};

    copyRegion.size = vertexBuffer.size;
    vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertexBuffer.buffer, 1, &copyRegion);


//    device->flushCommandBuffer(copyCmd, copyQueue);
    base.endTransferWait(copyCmd);

    // Destroy staging resources
    vkDestroyBuffer(base.device, vertexStaging.buffer, nullptr);
    vkFreeMemory(base.device, vertexStaging.memory, nullptr);
}

void VulkanLineVertexColoredAsset::destroy()
{
    vertexBuffer.destroy();
    vertexCount = 0;
}



void VulkanPointCloudAsset::render(vk::CommandBuffer cmd)
{
    if(!vertexBuffer.buffer) return;
    VkDeviceSize offsets[1] = { 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer.buffer, offsets);
    vkCmdDraw(cmd, vertexCount, 1, 0, 0);
}

void VulkanPointCloudAsset::updateBuffer(VulkanBase &base)
{
    vertexBuffer.destroy();
    auto vertices = mesh.points;

    vertexCount = vertices.size();

    uint32_t vBufferSize = vertexCount * sizeof(VertexType);


    // Use staging buffer to move vertex and index buffer to device local memory
    // Create staging buffers
    vks::Buffer vertexStaging;

    // Vertex buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &vertexStaging,
                        vBufferSize,
                        vertices.data()));


    // Create device local target buffers
    // Vertex buffer
    VK_CHECK_RESULT(base.createBuffer(
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        &vertexBuffer,
                        vBufferSize));


    // Copy from staging buffers
//    VkCommandBuffer copyCmd = device->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, true);
    VkCommandBuffer copyCmd = base.createAndBeginTransferCommand();

    VkBufferCopy copyRegion{};

    copyRegion.size = vertexBuffer.size;
    vkCmdCopyBuffer(copyCmd, vertexStaging.buffer, vertexBuffer.buffer, 1, &copyRegion);


//    device->flushCommandBuffer(copyCmd, copyQueue);
    base.endTransferWait(copyCmd);

    // Destroy staging resources
    vkDestroyBuffer(base.device, vertexStaging.buffer, nullptr);
    vkFreeMemory(base.device, vertexStaging.memory, nullptr);
}

void VulkanPointCloudAsset::destroy()
{
    vertexBuffer.destroy();
    vertexCount = 0;
}


}
}
