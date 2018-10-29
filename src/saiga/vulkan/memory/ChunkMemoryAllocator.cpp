//
// Created by Peter Eichinger on 10.10.18.
//

#include "ChunkMemoryAllocator.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/assert.h"
#include <string>
#include <functional>
#include "saiga/util/assert.h"
#include <algorithm>

void
ChunkMemoryAllocator::init(vk::Device _device, ChunkAllocator *chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                           const vk::BufferUsageFlags &usage,std::shared_ptr<FitStrategy> strategy,
                           vk::DeviceSize chunkSize, const std::string& name, bool _mapped) {
    m_logger = name;
    el::Loggers::getLogger(m_logger);

    mapped = _mapped;
    m_device = _device;
    m_chunkAllocator = chunkAllocator;
    m_chunkSize= chunkSize;
    flags = _flags;
    usageFlags = usage;
    m_bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive;
    m_bufferCreateInfo.usage = usageFlags;
    m_bufferCreateInfo.size = m_chunkSize;

    auto buffer = m_device.createBuffer(m_bufferCreateInfo);
    auto requirements = m_device.getBufferMemoryRequirements(buffer);
    m_allocateSize = requirements.size;
    m_alignment = requirements.alignment;

    m_device.destroy(buffer);
    m_strategy = strategy;

    CLOG(INFO, m_logger.c_str()) << "Created new allocator, alignment " << m_alignment;
}

MemoryLocation ChunkMemoryAllocator::allocate(vk::DeviceSize size) {
    SAIGA_ASSERT(size < m_chunkSize, "Can't allocate sizes bigger than chunk size");

    auto alignedSize = iAlignUp(size, m_alignment);
    CLOG(INFO, m_logger.c_str()) << "Requested " << size <<" (~"<< alignedSize<< ") bytes" ;
    ChunkIterator chunkAlloc;
    LocationIterator freeSpace;
    std::tie(chunkAlloc, freeSpace) = m_strategy->findRange(m_chunkAllocations, alignedSize);

    if (chunkAlloc == m_chunkAllocations.end()) {
        chunkAlloc = createNewChunk();
        freeSpace = chunkAlloc->freeList.begin();
    }

    auto memoryStart = freeSpace->offset;

    freeSpace->offset += alignedSize;
    freeSpace->size -= alignedSize;

    CLOG(INFO, m_logger.c_str()) <<
            "Allocating in chunk/offset [" << std::distance(m_chunkAllocations.begin(), chunkAlloc) << "/" <<
            memoryStart << "]";



    bool searchNewMax = false;


    if (chunkAlloc->maxFreeRange == freeSpace) {
        searchNewMax = true;
    }

    if (freeSpace->size == 0) {
        chunkAlloc->freeList.erase(freeSpace);
    }

    if (searchNewMax) {
        findNewMax(chunkAlloc);
    }

    MemoryLocation targetLocation = createMemoryLocation(chunkAlloc, memoryStart, alignedSize);
    auto memoryEnd = memoryStart + alignedSize;
    auto insertionPoint = std::find_if (chunkAlloc->allocations.begin(), chunkAlloc->allocations.end(),
            [=](MemoryLocation& loc){return loc.offset > memoryEnd;});

    return *chunkAlloc->allocations.insert(insertionPoint,targetLocation);
}

void ChunkMemoryAllocator::findNewMax(ChunkIterator &chunkAlloc) const {
    auto& freeList = chunkAlloc->freeList;
    chunkAlloc->maxFreeRange  = max_element(freeList.begin(), freeList.end(),
                                            [](MemoryLocation &first, MemoryLocation &second) { return first.size < second.size; });
}

MemoryLocation ChunkMemoryAllocator::createMemoryLocation(ChunkIterator iter, vk::DeviceSize offset,
                                                          vk::DeviceSize size) {
    return MemoryLocation{iter->buffer, iter->chunk->memory, offset,size, iter->mappedPointer};
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
    void* mappedPointer=nullptr;
    if (mapped) {
        mappedPointer = m_device.mapMemory(newChunk->memory,0, m_chunkSize);
    }
    m_chunkAllocations.emplace_back(newChunk,newBuffer,m_chunkSize,mappedPointer);

    return --m_chunkAllocations.end();
}

void ChunkMemoryAllocator::destroy() {
    for(auto& alloc : m_chunkAllocations) {
        m_device.destroy(alloc.buffer);
    }
    // TODO: Should destroy chunks as well
}

void ChunkMemoryAllocator::deallocate(MemoryLocation &location) {

    auto fChunk = std::find_if(m_chunkAllocations.begin(), m_chunkAllocations.end(),
            [&](ChunkAllocation const & alloc){return alloc.chunk->memory == location.memory;});

    SAIGA_ASSERT(fChunk != m_chunkAllocations.end(), "Allocation was not done with this allocator!");
    auto& chunkAllocs = fChunk->allocations;
    auto& chunkFree = fChunk->freeList;
    auto fLoc = std::find(chunkAllocs.begin(), chunkAllocs.end(), location);
    SAIGA_ASSERT(fLoc != chunkAllocs.end(), "Allocation is not part of the chunk");
    CLOG(INFO, m_logger.c_str()) << "Deallocating " << location.size << " bytes in chunk/offset [" << std::distance(m_chunkAllocations.begin(), fChunk)<<
                                 "/" << fLoc->offset << "]";

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


    bool shouldFindNewMax = (freePrev == fChunk->maxFreeRange || freeNext == fChunk->maxFreeRange || freeInsert == fChunk->maxFreeRange) ;


    if (freePrev != chunkFree.end() && freeNext != chunkFree.end()) {
        // Free space before and after newly freed space -> merge
        freePrev->size += location.size + freeNext->size;
        chunkFree.erase(freeNext);
    } else if (freePrev!= chunkFree.end()) {
        // Free only before -> increase size
        freePrev->size += location.size;
    } else if (freeNext != chunkFree.end()) {
        // Free only after newly freed -> move and increase size
        freeNext->offset = location.offset;
        freeNext->size += location.size;
    } else {
        if (foundInsert) {
            chunkFree.insert(freeInsert, location);
        } else {
            chunkFree.push_front(location);
        }
    }

    if (shouldFindNewMax) {
        findNewMax(fChunk);
    }

    chunkAllocs.erase(fLoc);
}

std::pair<ChunkIterator, LocationIterator>
FirstFitStrategy::findRange(std::vector<ChunkAllocation> &_allocations, vk::DeviceSize size) {

    auto foundChunk =std::find_if(_allocations.begin(), _allocations.end(),
            [&](ChunkAllocation& alloc){ return alloc.maxFreeRange->size > size;});

    if (foundChunk == _allocations.end()) {
        return std::make_pair(_allocations.end(), LocationIterator());
    }

    auto& chunk = *foundChunk;

    auto foundRange = std::find_if(chunk.freeList.begin(), chunk.freeList.end(),
            [&](MemoryLocation& loc) {return loc.size >= size;});

    SAIGA_ASSERT(foundRange != chunk.freeList.end(), "free size is invalid.");

    return std::make_pair(foundChunk, foundRange);
}
