//
// Created by Peter Eichinger on 2018-12-18.
//

#include "FallbackAllocator.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"

#include <numeric>

namespace Saiga::Vulkan::Memory
{
void FallbackAllocator::deallocate(BufferMemoryLocation* location)
{
    std::scoped_lock lock(mutex);
    LOG(INFO) << "Fallback deallocate: " << location;

    auto foundAllocation = std::find_if(m_allocations.begin(), m_allocations.end(),
                                        [location](auto& alloc) { return alloc.get() == location; });
    if (foundAllocation == m_allocations.end())
    {
        LOG(FATAL) << "Allocation was not made with this allocator";
        return;
    }
    destroy(m_device, foundAllocation->get());
    m_allocations.erase(foundAllocation);
}

void FallbackAllocator::deallocate(ImageMemoryLocation* location)
{
    std::scoped_lock lock(mutex);
    LOG(INFO) << "Fallback deallocate: " << location;

    auto foundAllocation = std::find_if(m_image_allocations.begin(), m_image_allocations.end(),
                                        [location](auto& alloc) { return alloc.get() == location; });
    if (foundAllocation == m_image_allocations.end())
    {
        LOG(FATAL) << "Allocation was not made with this allocator";
        return;
    }
    destroy(m_device, foundAllocation->get());
    m_image_allocations.erase(foundAllocation);
}


void FallbackAllocator::destroy()
{
    for (auto& location : m_allocations)
    {
        destroy(m_device, location.get());
    }
    m_allocations.clear();

    for (auto& image_location : m_image_allocations)
    {
        destroy(m_device, image_location.get());
    }
    m_image_allocations.clear();
}

BufferMemoryLocation* FallbackAllocator::allocate(const BufferType& type, vk::DeviceSize size)
{
    vk::BufferCreateInfo bufferCreateInfo{vk::BufferCreateFlags(), size, type.usageFlags};
    auto buffer = m_device.createBuffer(bufferCreateInfo);

    auto memReqs = m_device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo info;
    info.allocationSize  = memReqs.size;
    info.memoryTypeIndex = findMemoryType(m_physicalDevice, memReqs.memoryTypeBits, type.memoryFlags);
    auto memory          = SafeAllocator::instance()->allocateMemory(m_device, info);

    void* mappedPtr = nullptr;
    if (type.is_mappable())
    {
        mappedPtr = m_device.mapMemory(memory, 0, memReqs.size);
    }
    m_device.bindBufferMemory(buffer, memory, 0);

    BufferMemoryLocation* retVal;
    {
        std::scoped_lock lock(mutex);
        m_allocations.emplace_back(std::make_unique<BufferMemoryLocation>(buffer, memory, 0, size, mappedPtr));
        retVal = m_allocations.back().get();
    }
    LOG(INFO) << "Fallback allocation: " << type << "->" << retVal;
    return retVal;
}

ImageMemoryLocation* FallbackAllocator::allocate(const ImageType& type, ImageData& image_data)
{
    image_data.create_image(m_device);

    vk::MemoryAllocateInfo info;
    info.allocationSize = image_data.image_requirements.size;
    info.memoryTypeIndex =
        findMemoryType(m_physicalDevice, image_data.image_requirements.memoryTypeBits, type.memoryFlags);
    auto memory = SafeAllocator::instance()->allocateMemory(m_device, info);


    ImageMemoryLocation* retVal;
    {
        std::scoped_lock lock(mutex);
        m_image_allocations.emplace_back(
            std::make_unique<ImageMemoryLocation>(image_data, memory, 0, image_data.image_requirements.size, nullptr));
        retVal = m_image_allocations.back().get();
    }
    SAIGA_ASSERT(retVal, "Invalid pointer returned");
    bind_image_data(m_device, retVal, std::move(image_data));

    retVal->data.create_view(m_device);
    retVal->data.create_sampler(m_device);

    LOG(INFO) << "Fallback image allocation: " << type << "->" << retVal;
    return retVal;
}

void FallbackAllocator::showDetailStats()
{
    if (ImGui::CollapsingHeader(gui_identifier.c_str()))
    {
        ImGui::Indent();

        ImGui::LabelText("Number of allocations", "%lu", m_allocations.size());
        const auto totalSize = std::accumulate(m_allocations.begin(), m_allocations.end(), 0UL,
                                               [](const auto& a, const auto& b) { return a + b->size; });
        ImGui::LabelText("Size of allocations", "%s", sizeToString(totalSize).c_str());

        ImGui::Unindent();
    }
}

MemoryStats FallbackAllocator::collectMemoryStats()
{
    const auto totalSize = std::accumulate(m_allocations.begin(), m_allocations.end(), 0UL,
                                           [](const auto& a, const auto& b) { return a + b->size; });

    return MemoryStats{totalSize, totalSize, 0};
}
void FallbackAllocator::destroy(const vk::Device& device, BufferMemoryLocation* memory_location)
{
    SAIGA_ASSERT(memory_location->memory, "Already destroyed");
    memory_location->destroy_owned_data(device);
    device.destroy(memory_location->data.buffer);
    if (memory_location->memory)
    {
        device.free(memory_location->memory);
        memory_location->memory = nullptr;
    }
    memory_location->mappedPointer = nullptr;
}
void FallbackAllocator::destroy(const vk::Device& device, ImageMemoryLocation* memory_location)
{
    SAIGA_ASSERT(memory_location->memory, "Already destroyed");
    memory_location->destroy_owned_data(device);
    if (memory_location->memory)
    {
        device.free(memory_location->memory);
        memory_location->memory = nullptr;
    }
    memory_location->mappedPointer = nullptr;
}

}  // namespace Saiga::Vulkan::Memory