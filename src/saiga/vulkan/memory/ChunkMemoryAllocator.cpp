//
// Created by Peter Eichinger on 10.10.18.
//

#include "ChunkMemoryAllocator.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/assert.h"
#include <string>
MemoryLocation ChunkMemoryAllocator::allocate(vk::DeviceSize size) {
    SAIGA_ASSERT(size < m_chunkSize, "Can't allocate sizes bigger than chunk size");

    if (m_currentChunk == nullptr) {
        createNewBuffer();
    }

    auto alignedSize = iAlignUp(size, m_alignment);
    CLOG(INFO, m_logger.c_str()) << "Allocating " << size << " bytes, aligned: " << alignedSize;

    if (m_currentOffset + alignedSize > m_chunkSize) {
        createNewBuffer();
    }

    MemoryLocation targetLocation = {m_currentBuffer, m_currentChunk->memory, m_currentOffset,size};
    m_currentOffset += alignedSize;
    return targetLocation;
}

void ChunkMemoryAllocator::createNewBuffer() {
//    el::Loggers::getLogger("chunkMemAllocator");
    CLOG(INFO, m_logger.c_str()) << "Creating new chunk";
    m_currentChunk = m_chunkAllocator->allocate(flags, m_allocateSize);
    m_currentBuffer = m_device.createBuffer(m_bufferCreateInfo);
    m_currentOffset = 0;
    m_device.getBufferMemoryRequirements(m_currentBuffer);
    m_device.bindBufferMemory(m_currentBuffer, m_currentChunk->memory, 0);
    m_buffers.push_back(m_currentBuffer);
}

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

    if (m_allocateSize != m_chunkSize) {
        CLOG(FATAL, m_logger.c_str()) << vk::to_string(usage) <<  " buffer usage: Allocation / Chunk size is different!!!: " << m_allocateSize << "/" << m_chunkSize;
    }
    m_device.destroy(buffer);
}
