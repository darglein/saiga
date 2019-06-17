//
// Created by Peter Eichinger on 10.10.18.
//

#include "BufferChunkAllocator.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/easylogging++.h"

#include "FindMemoryType.h"
#include "SafeAllocator.h"

#include <functional>
#include <sstream>
#include <string>

namespace Saiga::Vulkan::Memory
{
BufferMemoryLocation* BufferChunkAllocator::allocate(vk::DeviceSize size)
{
    SAIGA_ASSERT(size > 0, "Allocations must have a size > 0");
    std::scoped_lock lock(allocationMutex);
    auto alignedSize = iAlignUp(size, m_alignment);
    SAIGA_ASSERT(alignedSize <= m_chunkSize, "Can't allocate sizes bigger than chunk size");
    auto location = ChunkAllocator::base_allocate(alignedSize);
    VLOG(3) << "Allocate buffer " << type << ":" << *location;
    return location;
}

ChunkIterator<BufferMemoryLocation> BufferChunkAllocator::createNewChunk()
{
    auto newBuffer = m_device.createBuffer(m_bufferCreateInfo);
    auto memReqs   = m_device.getBufferMemoryRequirements(newBuffer);

    vk::MemoryAllocateInfo info;
    info.allocationSize  = memReqs.size;
    info.memoryTypeIndex = findMemoryType(m_pDevice, memReqs.memoryTypeBits, type.memoryFlags);
    auto newChunk        = SafeAllocator::instance()->allocateMemory(m_device, info);
    auto memRequirements = m_device.getBufferMemoryRequirements(newBuffer);
    VLOG(3) << "New chunk: " << chunks.size() << " Mem " << newChunk << ", Buffer " << newBuffer;
    if (m_allocateSize != memRequirements.size)
    {
        LOG(ERROR) << "New buffer has differing memory requirements size";
    }
    m_device.bindBufferMemory(newBuffer, newChunk, 0);
    void* mappedPointer = nullptr;
    if (type.is_mappable())
    {
        mappedPointer = m_device.mapMemory(newChunk, 0, m_chunkSize);
        VLOG(3) << "Mapped pointer = " << mappedPointer;
    }
    chunks.emplace_back(newChunk, newBuffer, m_chunkSize, mappedPointer);

    return --chunks.end();
}

void BufferChunkAllocator::deallocate(BufferMemoryLocation* location)
{
    VLOG(3) << "Trying to deallocate buffer " << type << ":" << *location;
    ChunkAllocator::deallocate(location);
}

void BufferChunkAllocator::headerInfo()
{
    ImGui::LabelText("Buffer Usage", "%s", vk::to_string(type.usageFlags).c_str());
    ImGui::LabelText("Memory Type", "%s", vk::to_string(type.memoryFlags).c_str());
}

std::unique_ptr<BufferMemoryLocation> BufferChunkAllocator::create_location(
    ChunkIterator<Saiga::Vulkan::Memory::BaseMemoryLocation<Saiga::Vulkan::Memory::BufferData>>& chunk_alloc,
    vk::DeviceSize start, vk::DeviceSize size)
{
    return std::make_unique<BufferMemoryLocation>(chunk_alloc->buffer, chunk_alloc->memory, start, size,
                                                  chunk_alloc->mappedPointer);
}
}  // namespace Saiga::Vulkan::Memory
