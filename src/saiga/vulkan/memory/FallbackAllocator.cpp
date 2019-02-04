//
// Created by Peter Eichinger on 2018-12-18.
//

#include "FallbackAllocator.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/tostring.h"

#include <numeric>

namespace Saiga::Vulkan::Memory
{
void FallbackAllocator::deallocate(MemoryLocation* location)
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
    (*foundAllocation)->destroy(m_device);
    m_allocations.erase(foundAllocation);
}

void FallbackAllocator::destroy()
{
    for (auto& location : m_allocations)
    {
        location->destroy(m_device);
    }
    m_allocations.clear();
}

MemoryLocation* FallbackAllocator::allocate(const BufferType& type, vk::DeviceSize size)
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

    MemoryLocation* retVal;
    {
        std::scoped_lock lock(mutex);
        m_allocations.emplace_back(std::make_unique<MemoryLocation>(buffer, memory, 0, size, mappedPtr));
        retVal = m_allocations.back().get();
    }
    LOG(INFO) << "Fallback allocation: " << type << "->" << retVal;
    return retVal;
}

MemoryLocation* FallbackAllocator::allocate(const ImageType& type, const vk::Image& image)
{
    auto memReqs = m_device.getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo info;
    info.allocationSize  = memReqs.size;
    info.memoryTypeIndex = findMemoryType(m_physicalDevice, memReqs.memoryTypeBits, type.memoryFlags);
    auto memory          = SafeAllocator::instance()->allocateMemory(m_device, info);

    mutex.lock();
    m_allocations.emplace_back(std::make_unique<MemoryLocation>(vk::Buffer(), memory, 0, memReqs.size, nullptr));
    auto retVal = m_allocations.back().get();
    mutex.unlock();

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

MemoryLocation* FallbackAllocator::allocate(vk::DeviceSize size)
{
    SAIGA_ASSERT(false, "Fallback allocator must specify a buffer/image type for allocations");
    return nullptr;
}
}  // namespace Saiga::Vulkan::Memory