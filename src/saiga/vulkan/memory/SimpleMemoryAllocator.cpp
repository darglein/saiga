//
// Created by Peter Eichinger on 24.11.18.
//

#include "SimpleMemoryAllocator.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/tostring.h"

#include <numeric>
void SimpleMemoryAllocator::deallocate(MemoryLocation& location)
{
    mutex.lock();
    LOG(INFO) << "Simple deallocate " << vk::to_string(usageFlags) << " " << vk::to_string(flags) << " " << location;

    auto foundAllocation = std::find(m_allocations.begin(), m_allocations.end(), location);
    if (foundAllocation == m_allocations.end())
    {
        LOG(ERROR) << "Allocation was not made with this allocator";
        return;
    }
    foundAllocation->destroy(m_device);
    m_allocations.erase(foundAllocation);
    mutex.unlock();
}

void SimpleMemoryAllocator::destroy()
{
    for (auto& location : m_allocations)
    {
        if (location.buffer != static_cast<vk::Buffer>(nullptr))
        {
            location.destroy(m_device);
        }
    }
    m_allocations.clear();
}

MemoryLocation SimpleMemoryAllocator::allocate(vk::DeviceSize size)
{
    m_bufferCreateInfo.size = size;
    auto buffer             = m_device.createBuffer(m_bufferCreateInfo);

    auto memReqs = m_device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo info;
    info.allocationSize  = memReqs.size;
    info.memoryTypeIndex = findMemoryType(m_physicalDevice, memReqs.memoryTypeBits, flags);
    auto memory          = SafeAllocator::instance()->allocateMemory(m_device, info);

    void* mappedPtr = nullptr;
    if (mapped)
    {
        mappedPtr = m_device.mapMemory(memory, 0, memReqs.size);
    }
    m_device.bindBufferMemory(buffer, memory, 0);

    mutex.lock();
    m_allocations.emplace_back(buffer, memory, 0, size, mappedPtr);
    auto retVal = m_allocations.back();
    mutex.unlock();

    LOG(INFO) << "Simple allocate   " << vk::to_string(usageFlags) << " " << vk::to_string(flags) << " " << retVal;
    return retVal;
}

void SimpleMemoryAllocator::renderInfoGUI()
{
    if (ImGui::CollapsingHeader(gui_identifier.c_str()))
    {
        ImGui::Indent();

        ImGui::LabelText("Number of allocations", "%lu", m_allocations.size());
        const auto totalSize = std::accumulate(m_allocations.begin(), m_allocations.end(), 0UL,
                                               [](const auto& a, const auto& b) { return a + b.size; });
        ImGui::LabelText("Size of allocations", "%s", sizeToString(totalSize).c_str());

        ImGui::Unindent();
    }
}
