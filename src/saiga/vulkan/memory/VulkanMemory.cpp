//
// Created by Peter Eichinger on 2018-11-30.
//

#include "VulkanMemory.h"

#include "saiga/imgui/imgui.h"

#include <memory>
void VulkanMemory::init(vk::PhysicalDevice _pDevice, vk::Device _device)
{
    m_pDevice = _pDevice;
    m_device  = _device;
    strategy  = FirstFitStrategy();
    chunkAllocator.init(_pDevice, _device);

    auto vertIndexType = BufferType{vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                                        vk::BufferUsageFlagBits::eTransferDst,
                                    vk::MemoryPropertyFlagBits::eDeviceLocal};
    bufferAllocators.emplace(vertIndexType, std::make_shared<BufferChunkAllocator>(
                                                m_device, &chunkAllocator, vertIndexType.memoryFlags,
                                                vertIndexType.usageFlags, strategy, fallback_buffer_chunk_size, false));

    auto vertIndexHostType =
        BufferType{vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};
    bufferAllocators.emplace(vertIndexHostType,
                             std::make_shared<BufferChunkAllocator>(
                                 m_device, &chunkAllocator, vertIndexHostType.memoryFlags, vertIndexHostType.usageFlags,
                                 strategy, fallback_buffer_chunk_size, true));

    auto stagingType = BufferType{vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};
    bufferAllocators.emplace(stagingType,
                             std::make_shared<SimpleMemoryAllocator>(m_device, m_pDevice, stagingType.memoryFlags,
                                                                     stagingType.usageFlags, true));
}

VulkanMemory::BufferIter VulkanMemory::createNewBufferAllocator(VulkanMemory::BufferMap& map,
                                                                const VulkanMemory::BufferDefaultMap& defaultSizes,
                                                                const VulkanMemory::BufferType& type)
{
    auto found = find_default_size<BufferDefaultMap, BufferType>(default_buffer_chunk_sizes, type);

    bool mapped =
        (type.memoryFlags & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible;
    auto emplaced =
        map.emplace(type, std::make_shared<BufferChunkAllocator>(m_device, &chunkAllocator, type.memoryFlags,
                                                                 type.usageFlags, strategy, found->second, mapped));
    SAIGA_ASSERT(emplaced.second, "Allocator was already present.");
    return emplaced.first;
}

VulkanMemory::ImageIter VulkanMemory::createNewImageAllocator(VulkanMemory::ImageMap& map,
                                                              const VulkanMemory::ImageDefaultMap& defaultSizes,
                                                              const VulkanMemory::ImageType& type)
{
    auto found = find_default_size<ImageDefaultMap, ImageType>(default_image_chunk_sizes, type);
    bool mapped =
        (type.memoryFlags & vk::MemoryPropertyFlagBits::eHostVisible) == vk::MemoryPropertyFlagBits::eHostVisible;

    auto emplaced = map.emplace(
        type, ImageChunkAllocator(m_device, &chunkAllocator, type.memoryFlags, strategy, found->second, mapped));
    SAIGA_ASSERT(emplaced.second, "Allocator was already present.");
    return emplaced.first;
}

BaseMemoryAllocator& VulkanMemory::getAllocator(const vk::BufferUsageFlags& usage, const vk::MemoryPropertyFlags& flags)
{
    auto foundAllocator = findAllocator<BufferMap, vk::BufferUsageFlags>(bufferAllocators, {usage, flags});
    if (foundAllocator == bufferAllocators.end())
    {
        foundAllocator = createNewBufferAllocator(bufferAllocators, default_buffer_chunk_sizes, {usage, flags});
    }
    return *(foundAllocator->second);
}

BaseMemoryAllocator& VulkanMemory::getImageAllocator(const vk::MemoryPropertyFlags& flags,
                                                     const vk::ImageUsageFlags& usage)
{
    auto foundAllocator = findAllocator<ImageMap, vk::ImageUsageFlags>(imageAllocators, {usage, flags});

    if (foundAllocator == imageAllocators.end())
    {
        foundAllocator = createNewImageAllocator(imageAllocators, default_image_chunk_sizes, {usage, flags});
    }

    return foundAllocator->second;
}

void VulkanMemory::destroy()
{
    chunkAllocator.destroy();

    for (auto& allocator : bufferAllocators)
    {
        allocator.second->destroy();
    }

    for (auto& allocator : imageAllocators)
    {
        allocator.second.destroy();
    }
}

void VulkanMemory::renderGUI()
{
    if (!ImGui::CollapsingHeader("Memory Stats"))
    {
        return;
    }

    ImGui::Indent();

    for (auto& allocator : bufferAllocators)
    {
        allocator.second->renderInfoGUI();
    }
    for (auto& allocator : imageAllocators)
    {
        allocator.second.renderInfoGUI();
    }

    ImGui::Unindent();
}
