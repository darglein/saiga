//
// Created by Peter Eichinger on 24.11.18.
//

#include "SimpleMemoryAllocator.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"

#include <numeric>

namespace Saiga::Vulkan::Memory
{
void SimpleMemoryAllocator::deallocate(MemoryLocation* location)
{
    std::scoped_lock lock(mutex);
    LOG(INFO) << "Simple deallocate " << type << ":" << location;

    auto foundAllocation = std::find_if(m_allocations.begin(), m_allocations.end(),
                                        [location](auto& alloc) { return alloc.get() == location; });
    if (foundAllocation == m_allocations.end())
    {
        LOG(ERROR) << "Allocation was not made with this allocator";
        return;
    }
    (*foundAllocation)->destroy(m_device);
    m_allocations.erase(foundAllocation);
}

void SimpleMemoryAllocator::destroy()
{
    for (auto& location : m_allocations)
    {
        if (location->buffer != static_cast<vk::Buffer>(nullptr))
        {
            location->destroy(m_device);
        }
    }
    m_allocations.clear();
}

MemoryLocation* SimpleMemoryAllocator::allocate(vk::DeviceSize size)
{
    m_bufferCreateInfo.size = size;
    auto buffer             = m_device.createBuffer(m_bufferCreateInfo);

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

    mutex.lock();
    m_allocations.push_back(std::make_unique<MemoryLocation>(buffer, memory, 0, size, mappedPtr));
    auto retVal = m_allocations.back().get();
    mutex.unlock();

    LOG(INFO) << "Simple allocate   " << type << " " << retVal;
    return retVal;
}

void SimpleMemoryAllocator::showDetailStats()
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

MemoryStats SimpleMemoryAllocator::collectMemoryStats()
{
    const auto totalSize = std::accumulate(m_allocations.begin(), m_allocations.end(), 0UL,
                                           [](const auto& a, const auto& b) { return a + b->size; });

    return MemoryStats{totalSize, totalSize, 0};
}
}  // namespace Saiga::Vulkan::Memory