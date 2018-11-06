//
// Created by Peter Eichinger on 30.10.18.
//

#include "ImageChunkAllocator.h"
#include "saiga/util/easylogging++.h"

Saiga::Vulkan::Memory::ChunkIterator Saiga::Vulkan::Memory::ImageChunkAllocator::createNewChunk() {
    auto newChunk = m_chunkAllocator->allocate(flags, m_allocateSize);

    m_chunkAllocations.emplace_back(newChunk, vk::Buffer(), m_chunkSize, nullptr);

    return --m_chunkAllocations.end();
}

Saiga::Vulkan::Memory::ImageChunkAllocator::ImageChunkAllocator(const vk::Device &_device,
                                                                Saiga::Vulkan::Memory::ChunkCreator *chunkAllocator,
                                                                const vk::MemoryPropertyFlags &_flags,
                                                                Saiga::Vulkan::Memory::FitStrategy &strategy,
                                                                vk::DeviceSize chunkSize,bool _mapped)
        : BaseChunkAllocator(_device, chunkAllocator, _flags, strategy, chunkSize, _mapped) {

    LOG(INFO) << "Created new image allocator for flags " << vk::to_string(_flags);
}

void Saiga::Vulkan::Memory::ImageChunkAllocator::destroy() {

}
