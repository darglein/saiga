//
// Created by Peter Eichinger on 2018-11-30.
//

#include "VulkanMemory.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/tostring.h"

#include <memory>
void VulkanMemory::init(vk::PhysicalDevice _pDevice, vk::Device _device)
{
    m_pDevice = _pDevice;
    m_device  = _device;
    strategy  = FirstFitStrategy();
    chunkCreator.init(_pDevice, _device);

    auto props = _pDevice.getMemoryProperties();
    memoryTypes.resize(props.memoryTypeCount);
    for (int i = 0; i < props.memoryTypeCount; ++i)
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
    bufferAllocators.emplace(effectiveStaging,
                             std::make_unique<SimpleMemoryAllocator>(m_device, m_pDevice, effectiveStaging));

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
                                                   m_device, &chunkCreator, effectiveType, strategy, found->second));
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
                                                   m_device, &chunkCreator, effectiveType, strategy, found->second));
    SAIGA_ASSERT(emplaced.second, "Allocator was already present.");
    return emplaced.first;
}

BaseMemoryAllocator& VulkanMemory::getAllocator(const BufferType& type)
{
    auto foundAllocator = findAllocator<BufferMap, vk::BufferUsageFlags>(bufferAllocators, type);
    if (foundAllocator == bufferAllocators.end())
    {
        foundAllocator = createNewBufferAllocator(bufferAllocators, default_buffer_chunk_sizes, type);
    }
    return *(foundAllocator->second);
}

BaseMemoryAllocator& VulkanMemory::getImageAllocator(const ImageType& type)
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

void VulkanMemory::renderGUI()
{
    if (!ImGui::CollapsingHeader("Memory Stats"))
    {
        return;
    }
    static std::unordered_map<vk::MemoryPropertyFlags, MemoryStats> memoryTypeStats;


    for (auto& entry : memoryTypeStats)
    {
        entry.second = MemoryStats();
    }


    for (auto& allocator : bufferAllocators)
    {
        if (memoryTypeStats.find(allocator.first.memoryFlags) == memoryTypeStats.end())
        {
            memoryTypeStats[allocator.first.memoryFlags] = MemoryStats();
        }
        memoryTypeStats[allocator.first.memoryFlags] += allocator.second->collectMemoryStats();
    }
    for (auto& allocator : imageAllocators)
    {
        if (memoryTypeStats.find(allocator.first.memoryFlags) == memoryTypeStats.end())
        {
            memoryTypeStats[allocator.first.memoryFlags] = MemoryStats();
        }
        memoryTypeStats[allocator.first.memoryFlags] += allocator.second->collectMemoryStats();
    }

    static std::vector<ImGui::ColoredBar> bars;

    bars.resize(memoryTypeStats.size(),
                ImGui::ColoredBar({0, 16}, {{0.1f, 0.1f, 0.1f, 1.0f}, {0.4f, 0.4f, 0.4f, 1.0f}}, true));
    int index = 0;
    for (auto& memStat : memoryTypeStats)
    {
        ImGui::Text("%s", vk::to_string(memStat.first).c_str());
        // ImGui::SameLine(150.0f);
        auto& bar = bars[index];

        ImGui::Indent();
        bar.renderBackground();
        bar.renderArea(0.0f, static_cast<float>(memStat.second.used) / memStat.second.allocated,
                       ImGui::ColoredBar::BarColor{{0.0f, 0.2f, 0.2f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}});

        ImGui::Text("%s / %s (%s fragmented free)", sizeToString(memStat.second.used).c_str(),
                    sizeToString(memStat.second.allocated).c_str(), sizeToString(memStat.second.fragmented).c_str());

        ImGui::Unindent();
        index++;
    }

    ImGui::Spacing();


    if (!ImGui::CollapsingHeader("Detailed Memory Stats"))
    {
        return;
    }

    ImGui::Indent();

    for (auto& allocator : bufferAllocators)
    {
        allocator.second->showDetailStats();
    }
    for (auto& allocator : imageAllocators)
    {
        allocator.second->showDetailStats();
    }

    ImGui::Unindent();
}

MemoryLocation VulkanMemory::allocate(const BufferType& type, vk::DeviceSize size)
{
    auto& allocator = getAllocator(type);

    if (size > allocator.maxAllocationSize)
    {
        return fallbackAllocator->allocate(type, size);
    }
    return allocator.allocate(size);
}

MemoryLocation VulkanMemory::allocate(const ImageType& type, const vk::Image& image)
{
    auto image_mem_reqs = m_device.getImageMemoryRequirements(image);

    auto& allocator = getImageAllocator(type);

    if (image_mem_reqs.size > allocator.maxAllocationSize)
    {
        return fallbackAllocator->allocate(type, image);
    }
    return allocator.allocate(image_mem_reqs.size);
}

void VulkanMemory::deallocateBuffer(const BufferType& type, MemoryLocation& location)
{
    auto& allocator = getAllocator(type);
    if (location.size > allocator.maxAllocationSize)
    {
        fallbackAllocator->deallocate(location);
    }
    else
    {
        allocator.deallocate(location);
    }
}

void VulkanMemory::deallocateImage(const ImageType& type, MemoryLocation& location)
{
    auto& allocator = getImageAllocator(type);
    if (location.size > allocator.maxAllocationSize)
    {
        fallbackAllocator->deallocate(location);
    }
    else
    {
        allocator.deallocate(location);
    }
}
