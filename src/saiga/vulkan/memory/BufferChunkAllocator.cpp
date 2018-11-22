//
// Created by Peter Eichinger on 10.10.18.
//

#include "BufferChunkAllocator.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/assert.h"
#include <string>
#include <functional>
#include "saiga/util/assert.h"
#include "BaseChunkAllocator.h"


MemoryLocation BufferChunkAllocator::allocate(vk::DeviceSize size) {
    SAIGA_ASSERT(size < m_chunkSize, "Can't allocate sizes bigger than chunk size");

    auto alignedSize = iAlignUp(size, m_alignment);
    LOG(INFO) << "Requested " << size <<" (~"<< alignedSize<< ") bytes" ;
    return BaseChunkAllocator::allocate(alignedSize);
}

ChunkIterator BufferChunkAllocator::createNewChunk() {
    LOG(INFO) << "Creating new chunk: " << m_chunkAllocations.size();
    auto newChunk = m_chunkAllocator->allocate(flags, m_allocateSize);
    auto newBuffer = m_device.createBuffer(m_bufferCreateInfo);
    auto memRequirements = m_device.getBufferMemoryRequirements(newBuffer);
    if (m_allocateSize != memRequirements.size) {
        LOG(INFO) << "New buffer has differing memory requirements size";
    }
    m_device.bindBufferMemory(newBuffer, newChunk->memory, 0);
    void *mappedPointer = nullptr;
    if (mapped) {
        mappedPointer = m_device.mapMemory(newChunk->memory, 0, m_chunkSize);
    }
    m_chunkAllocations.emplace_back(newChunk, newBuffer, m_chunkSize, mappedPointer);

    return --m_chunkAllocations.end();
}

void BufferChunkAllocator::destroy() {
    for(auto& alloc : m_chunkAllocations) {
        m_device.destroy(alloc.buffer);
    }
}

