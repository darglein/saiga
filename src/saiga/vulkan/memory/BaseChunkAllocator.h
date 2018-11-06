//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once

#include "BaseMemoryAllocator.h"
#include "ChunkCreator.h"
#include "FitStrategy.h"
#include "saiga/vulkan/memory/BaseMemoryAllocator.h"
#include <mutex>

namespace Saiga{
namespace Vulkan{
namespace Memory {
class SAIGA_LOCAL BaseChunkAllocator : public BaseMemoryAllocator{
private:
    std::mutex allocationMutex;

    void findNewMax(ChunkIterator &chunkAlloc) const;

    MemoryLocation createMemoryLocation(ChunkIterator iter, vk::DeviceSize start, vk::DeviceSize size);

public:

    BaseChunkAllocator() : BaseMemoryAllocator(false) {

    }
    BaseChunkAllocator(vk::Device _device, ChunkCreator* chunkAllocator, const vk::MemoryPropertyFlags &_flags,
                        FitStrategy& strategy, vk::DeviceSize chunkSize = 64* 1024* 1024,
                        bool _mapped = false) : m_device(_device), m_chunkAllocator(chunkAllocator), flags(_flags),
                                                m_strategy(&strategy), m_chunkSize(chunkSize), BaseMemoryAllocator(_mapped), m_allocateSize(m_chunkSize){

    }

    MemoryLocation allocate(vk::DeviceSize size) override;

    void deallocate(MemoryLocation &location) override;

protected:
    vk::Device m_device;
    ChunkCreator* m_chunkAllocator{};
    vk::DeviceSize m_chunkSize{};
    vk::DeviceSize m_allocateSize{};

    FitStrategy* m_strategy{};
    vk::MemoryPropertyFlags flags;
    std::vector<ChunkAllocation> m_chunkAllocations;


    virtual ChunkIterator createNewChunk() = 0;
};


}
}
}