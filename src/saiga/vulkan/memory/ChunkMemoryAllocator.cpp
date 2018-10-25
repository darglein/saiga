//
// Created by Peter Eichinger on 10.10.18.
//

#include "ChunkMemoryAllocator.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/assert.h"
#include <string>
#include <functional>
#include "saiga/util/assert.h"
void
ChunkMemoryAllocator::init(vk::Device _device, ChunkAllocator *chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                           const vk::BufferUsageFlags &usage,std::shared_ptr<FitStrategy> strategy,
                           vk::DeviceSize chunkSize, const std::string& name) {
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
    m_strategy = strategy;

    CLOG(INFO, m_logger.c_str()) << "Created new allocator, alignment " << m_alignment;
}

MemoryLocation& ChunkMemoryAllocator::allocate(vk::DeviceSize size) {
    SAIGA_ASSERT(size < m_chunkSize, "Can't allocate sizes bigger than chunk size");

    auto alignedSize = iAlignUp(size, m_alignment);
    CLOG(INFO, m_logger.c_str()) << "Requested " << size <<" (~"<< alignedSize<< ") bytes" ;
    ChunkIterator chunk;
    LocationIterator freeSpace;
    std::tie(chunk, freeSpace) = m_strategy->findRange(m_chunkAllocations, alignedSize);

    if (chunk == m_chunkAllocations.end()) {
        chunk = createNewChunk();
        freeSpace = chunk->freeList.begin();
    }

    auto memoryStart = freeSpace->offset;

    freeSpace->offset += alignedSize;
    freeSpace->size -= alignedSize;

    if (&*freeSpace == chunk->maxFreeRange) {
        chunk->maxFreeSize = freeSpace->size;
    }

    CLOG(INFO, m_logger.c_str()) <<
            "Allocating in chunk/offset [" << std::distance(m_chunkAllocations.begin(), chunk) << ", " <<
            memoryStart << "]";


    MemoryLocation targetLocation {chunk->buffer, chunk->chunk->memory,memoryStart, alignedSize};
    auto memoryEnd = memoryStart + alignedSize;
    auto insertionPoint = std::find_if (chunk->allocations.begin(), chunk->allocations.end(),
            [=](MemoryLocation& loc){return loc.offset > memoryEnd;});

    return *chunk->allocations.insert(insertionPoint,targetLocation);
}

ChunkIterator ChunkMemoryAllocator::createNewChunk() {
    CLOG(INFO, m_logger.c_str()) << "Creating new chunk: " << m_chunkAllocations.size();
    auto newChunk = m_chunkAllocator->allocate(flags, m_allocateSize);
    auto newBuffer = m_device.createBuffer(m_bufferCreateInfo);
    auto memRequirements = m_device.getBufferMemoryRequirements(newBuffer);
    if (m_allocateSize != memRequirements.size) {
        CLOG(WARNING, m_logger.c_str()) << "New buffer has differing memory requirements size";
    }
    m_device.bindBufferMemory(newBuffer, newChunk->memory,0);
    m_chunkAllocations.emplace_back(newChunk,newBuffer,m_chunkSize);
    return --m_chunkAllocations.end();
}

void ChunkMemoryAllocator::destroy() {
    for(auto& alloc : m_chunkAllocations) {
        m_device.destroy(alloc.buffer);
    }
    // TODO: Should destroy chunks as well
}

void ChunkMemoryAllocator::deallocate(MemoryLocation &location) {

    CLOG(INFO, m_logger.c_str()) << "Deallocating " << location.size << " bytes";
    auto fChunk = std::find_if(m_chunkAllocations.begin(), m_chunkAllocations.end(),
            [&](ChunkAllocation const & alloc){return alloc.chunk->memory == location.memory;});

    SAIGA_ASSERT(fChunk != m_chunkAllocations.end(), "Allocation was not done with this allocator!");
    auto& chunkAllocs = fChunk->allocations;
    auto& chunkFree = fChunk->freeList;
    auto fLoc = std::find(chunkAllocs.begin(), chunkAllocs.end(), location);
    SAIGA_ASSERT(fLoc != chunkAllocs.end(), "Allocation is not part of the chunk");
    CLOG(INFO, m_logger.c_str()) << "Found chunk/allocation [" << std::distance(m_chunkAllocations.begin(), fChunk)<<
                                 "/" << std::distance(chunkAllocs.begin(), fLoc) << "]";

    LocationIterator freePrev, freeNext, freeInsert;
    bool foundInsert = false;
    freePrev = freeNext = chunkFree.end();
    freeInsert = chunkFree.end();
    for(auto freeIt = chunkFree.begin(); freeIt != chunkFree.end(); ++freeIt) {
        if (freeIt->offset + freeIt->size == location.offset) {
            freePrev = freeIt;
        }
        if (freeIt->offset == location.offset+ location.size) {
            freeNext = freeIt;
            break;
        }
        if (freeIt->offset + freeIt->size < location.offset) {
            freeInsert = freeIt;
            foundInsert = true;
        }
    }


//    SAIGA_ASSERT(freeInsert != chunkFree.end(), "No point to insert in free list found");

    if (freePrev != chunkFree.end() && freeNext != chunkFree.end()) {
        freePrev->size += location.size + freeNext->size;
        chunkFree.erase(freeNext);
    } else if (freePrev!= chunkFree.end()) {
        freePrev->size += location.size;
    } else if (freeNext != chunkFree.end()) {
        freeNext->offset = location.offset;
        freeNext->size += location.size;
    } else {
        if (foundInsert) {
            chunkFree.insert(freeInsert, location);
        } else {
            chunkFree.push_front(location);
        }
    }

    chunkAllocs.erase(fLoc);
}

std::pair<ChunkIterator, LocationIterator>
FirstFitStrategy::findRange(std::vector<ChunkAllocation> &_allocations, vk::DeviceSize size) {

    auto foundChunk =std::find_if(_allocations.begin(), _allocations.end(),
            [&](ChunkAllocation& alloc){ return alloc.maxFreeSize > size;});

    if (foundChunk == _allocations.end()) {
        return std::make_pair(_allocations.end(), LocationIterator());
    }

    auto& chunk = *foundChunk;

    auto foundRange = std::find_if(chunk.freeList.begin(), chunk.freeList.end(),
            [&](MemoryLocation& loc) {return loc.size >= size;});

    SAIGA_ASSERT(foundRange != chunk.freeList.end(), "free size is invalid.");

    return std::make_pair(foundChunk, foundRange);
}
