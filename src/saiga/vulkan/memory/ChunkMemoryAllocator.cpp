//
// Created by Peter Eichinger on 10.10.18.
//

#include "ChunkMemoryAllocator.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/assert.h"
#include <string>
void
ChunkMemoryAllocator::init(vk::Device _device, ChunkAllocator *chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                           const vk::BufferUsageFlags &usage, vk::DeviceSize chunkSize, const std::string& name) {
    m_logger = name;
    el::Loggers::getLogger(m_logger);
    m_device = _device;
    m_chunkAllocator = chunkAllocator;
    m_chunkSize= chunkSize;
    flags = _flags;
    usageFlags = usage;
    m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
    m_bufferCreateInfo.usage = usage;
    m_bufferCreateInfo.size = m_chunkSize;

    auto buffer = m_device.createBuffer(m_bufferCreateInfo);
    auto requirements = m_device.getBufferMemoryRequirements(buffer);
    m_allocateSize = requirements.size;
    m_alignment = requirements.alignment;

    m_device.destroy(buffer);

    CLOG(INFO, m_logger.c_str()) << "Created new allocator, alignment " << m_alignment;
}

MemoryLocation ChunkMemoryAllocator::allocate(vk::DeviceSize size) {
    SAIGA_ASSERT(size < m_chunkSize, "Can't allocate sizes bigger than chunk size");

    auto alignedSize = iAlignUp(size, m_alignment);
    CLOG(INFO, m_logger.c_str()) << "Allocating " << size << " bytes, aligned: " << alignedSize;
    auto range = m_strategy->findRange(m_chunkAllocations, alignedSize);

    if (!range.first) {
        range = createNewChunk();
    }
    auto memoryStart = range.second->start;
    range.second->start += alignedSize;
    range.second->range -= alignedSize;

    return MemoryLocation{range.first->buffer, range.first->chunk->memory,memoryStart,size};
}

std::pair<ChunkAllocation*, MemoryRange*> ChunkMemoryAllocator::createNewChunk() {

    CLOG(INFO, m_logger.c_str()) << "Creating new chunk";
    auto newChunk = m_chunkAllocator->allocate(flags, m_allocateSize);
    auto newBuffer = m_device.createBuffer(m_bufferCreateInfo);
    auto memRequirements = m_device.getBufferMemoryRequirements(newBuffer);
    if (m_allocateSize != memRequirements.size) {
        CLOG(WARNING, m_logger.c_str()) << "New buffer has differing memory requirements size";
    }
    m_device.bindBufferMemory(newBuffer, newChunk->memory,0);
    m_chunkAllocations.emplace_back(newChunk,newBuffer,m_chunkSize);
    return std::make_pair(&*m_chunkAllocations.rbegin(),&*m_chunkAllocations.rbegin()->freeList.begin());
}

void ChunkMemoryAllocator::destroy() {
    for(auto& alloc : m_chunkAllocations) {
        m_device.destroy(alloc.buffer);
    }
}
