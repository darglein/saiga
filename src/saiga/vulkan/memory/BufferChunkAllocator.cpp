//
// Created by Peter Eichinger on 10.10.18.
//

#include "BufferChunkAllocator.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/assert.h"
#include "saiga/util/easylogging++.h"

#include "BaseChunkAllocator.h"

#include <functional>
#include <sstream>
#include <string>

MemoryLocation* BufferChunkAllocator::allocate(vk::DeviceSize size)
{
    auto alignedSize = iAlignUp(size, m_alignment);
    // LOG(INFO) << "Requested " << size << " (~" << alignedSize << ") bytes";
    SAIGA_ASSERT(alignedSize <= m_chunkSize, "Can't allocate sizes bigger than chunk size");
    auto location = BaseChunkAllocator::allocate(alignedSize);
    LOG(INFO) << "Allocate buffer " << type << ":" << *location;
    return location;
}

ChunkIterator BufferChunkAllocator::createNewChunk()
{
    auto newChunk        = m_chunkAllocator->allocate(type.memoryFlags, m_allocateSize);
    auto newBuffer       = m_device.createBuffer(m_bufferCreateInfo);
    auto memRequirements = m_device.getBufferMemoryRequirements(newBuffer);
    LOG(INFO) << "New chunk: " << m_chunkAllocations.size() << " Mem " << newChunk->memory << ", Buffer " << newBuffer;
    if (m_allocateSize != memRequirements.size)
    {
        LOG(ERROR) << "New buffer has differing memory requirements size";
    }
    m_device.bindBufferMemory(newBuffer, newChunk->memory, 0);
    void* mappedPointer = nullptr;
    if (type.is_mappable())
    {
        mappedPointer = m_device.mapMemory(newChunk->memory, 0, m_chunkSize);
        LOG(INFO) << "Mapped pointer = " << mappedPointer;
    }
    m_chunkAllocations.emplace_back(newChunk, newBuffer, m_chunkSize, mappedPointer);

    return --m_chunkAllocations.end();
}

void BufferChunkAllocator::deallocate(MemoryLocation* location)
{
    LOG(INFO) << "Trying to deallocate buffer " << type << ":" << *location;
    BaseChunkAllocator::deallocate(location);
}

void BufferChunkAllocator::headerInfo()
{
    ImGui::LabelText("Buffer Usage", "%s", vk::to_string(type.usageFlags).c_str());
    ImGui::LabelText("Memory Type", "%s", vk::to_string(type.memoryFlags).c_str());
}
