//
// Created by Peter Eichinger on 2018-11-30.
//

#include "VulkanMemory.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/tostring.h"

#include <memory>
namespace Saiga::Vulkan::Memory
{
void VulkanMemory::init(vk::PhysicalDevice _pDevice, vk::Device _device)
{
    m_pDevice = _pDevice;
    m_device  = _device;
    strategy  = std::make_unique<FirstFitStrategy>();
    chunkCreator.init(_pDevice, _device);

    auto props = _pDevice.getMemoryProperties();
    memoryTypes.resize(props.memoryTypeCount);
    for (auto i = 0U; i < props.memoryTypeCount; ++i)
    {
        memoryTypes[i] = props.memoryTypes[i];
    }

    auto vertIndexType = BufferType{vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                                        vk::BufferUsageFlagBits::eTransferDst,
                                    vk::MemoryPropertyFlagBits::eDeviceLocal};
    auto vertIndexHostType =
        BufferType{vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                       vk::BufferUsageFlagBits::eTransferDst,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};

    getAllocator(vertIndexType);
    getAllocator(vertIndexHostType);

    auto stagingType = BufferType{vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent};

    auto effectiveFlags   = getEffectiveFlags(stagingType.memoryFlags);
    auto effectiveStaging = BufferType{stagingType.usageFlags, effectiveFlags};
    // bufferAllocators.emplace(effectiveStaging,
    //                         std::make_unique<SimpleMemoryAllocator>(m_device, m_pDevice, effectiveStaging));
    getAllocator(effectiveStaging);

    fallbackAllocator = std::make_unique<FallbackAllocator>(_device, _pDevice);
}

VulkanMemory::BufferIter VulkanMemory::createNewBufferAllocator(VulkanMemory::BufferMap& map,
                                                                const VulkanMemory::BufferDefaultMap& defaultSizes,
                                                                const BufferType& type)
{
    auto effectiveFlags = getEffectiveFlags(type.memoryFlags);

    auto effectiveType = BufferType{type.usageFlags, effectiveFlags};

    auto found = find_default_size<BufferDefaultMap, BufferType>(default_buffer_chunk_sizes, effectiveType);


    auto emplaced = map.emplace(effectiveType, std::make_unique<BufferChunkAllocator>(
                                                   m_device, &chunkCreator, effectiveType, *strategy, found->second));
    SAIGA_ASSERT(emplaced.second, "Allocator was already present.");
    return emplaced.first;
}

VulkanMemory::ImageIter VulkanMemory::createNewImageAllocator(VulkanMemory::ImageMap& map,
                                                              const VulkanMemory::ImageDefaultMap& defaultSizes,
                                                              const ImageType& type)
{
    auto effectiveFlags = getEffectiveFlags(type.memoryFlags);

    auto effectiveType = ImageType{type.usageFlags, effectiveFlags};

    auto found = find_default_size<ImageDefaultMap, ImageType>(default_image_chunk_sizes, type);

    auto emplaced = map.emplace(effectiveType, std::make_unique<ImageChunkAllocator>(
                                                   m_device, &chunkCreator, effectiveType, *strategy, found->second));
    SAIGA_ASSERT(emplaced.second, "Allocator was already present.");
    return emplaced.first;
}

BufferChunkAllocator& VulkanMemory::getAllocator(const BufferType& type)
{
    auto foundAllocator = findAllocator<BufferMap, vk::BufferUsageFlags>(bufferAllocators, type);
    if (foundAllocator == bufferAllocators.end())
    {
        foundAllocator = createNewBufferAllocator(bufferAllocators, default_buffer_chunk_sizes, type);
    }
    return *(foundAllocator->second);
}

ImageChunkAllocator& VulkanMemory::getImageAllocator(const ImageType& type)
{
    auto foundAllocator = findAllocator<ImageMap, vk::ImageUsageFlags>(imageAllocators, type);

    if (foundAllocator == imageAllocators.end())
    {
        foundAllocator = createNewImageAllocator(imageAllocators, default_image_chunk_sizes, type);
    }

    return *(foundAllocator->second);
}

void VulkanMemory::destroy()
{
    chunkCreator.destroy();

    for (auto& allocator : bufferAllocators)
    {
        allocator.second->destroy();
    }

    for (auto& allocator : imageAllocators)
    {
        allocator.second->destroy();
    }
}


MemoryLocation* VulkanMemory::allocate(const BufferType& type, vk::DeviceSize size)
{
    auto& allocator = getAllocator(type);

    if (size > allocator.maxAllocationSize)
    {
        return fallbackAllocator->allocate(type, size);
    }
    return allocator.allocate(size);
}

MemoryLocation* VulkanMemory::allocate(const ImageType& type, const vk::Image& image)
{
    auto image_mem_reqs = m_device.getImageMemoryRequirements(image);

    auto& allocator = getImageAllocator(type);

    if (image_mem_reqs.size > allocator.maxAllocationSize)
    {
        return fallbackAllocator->allocate(type, image);
    }
    return allocator.allocate(image_mem_reqs.size, image);
}

void VulkanMemory::deallocateBuffer(const BufferType& type, MemoryLocation* location)
{
    auto& allocator = getAllocator(type);
    if (location->size > allocator.maxAllocationSize)
    {
        fallbackAllocator->deallocate(location);
    }
    else
    {
        allocator.deallocate(location);
    }
}

void VulkanMemory::deallocateImage(const ImageType& type, MemoryLocation* location)
{
    auto& allocator = getImageAllocator(type);
    if (location->size > allocator.maxAllocationSize)
    {
        fallbackAllocator->deallocate(location);
    }
    else
    {
        allocator.deallocate(location);
    }
}
}  // namespace Saiga::Vulkan::Memory