//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once

#include "saiga/vulkan/memory/MemoryLocation.h"
#include <list>

namespace Saiga{
namespace Vulkan{
namespace Memory {


typedef std::list<MemoryLocation>::iterator LocationIterator;

struct SAIGA_GLOBAL ChunkAllocation{
        std::shared_ptr < Chunk > chunk;
        vk::Buffer buffer;
        std::list<MemoryLocation> allocations;
        std::list<MemoryLocation> freeList;
        LocationIterator maxFreeRange;
        void* mappedPointer;

        ChunkAllocation(std::shared_ptr<Chunk> _chunk, vk::Buffer _buffer, vk::DeviceSize size, void* _mappedPointer) :
        chunk(_chunk), buffer(_buffer), freeList(), allocations(), mappedPointer(_mappedPointer){
            freeList.emplace_back(_buffer, _chunk->memory, 0, size);
            maxFreeRange = freeList.begin();
        }
};

typedef std::vector<ChunkAllocation>::iterator ChunkIterator;

}
}
}