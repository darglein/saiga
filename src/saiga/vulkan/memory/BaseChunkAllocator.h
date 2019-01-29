//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once

#include "saiga/imgui/imgui.h"
#include "saiga/vulkan/Queue.h"
#include "saiga/vulkan/memory/BaseMemoryAllocator.h"

#include "BaseMemoryAllocator.h"
#include "ChunkCreator.h"
#include "FitStrategy.h"

#include <mutex>


namespace Saiga::Vulkan::Memory
{
class SAIGA_GLOBAL BaseChunkAllocator : public BaseMemoryAllocator
{
   private:
    std::mutex allocationMutex;
    void findNewMax(ChunkIterator& chunkAlloc) const;

   protected:
    vk::Device m_device;
    ChunkCreator* m_chunkAllocator{};

   public:
    FitStrategy* strategy{};
    Queue* queue;

    vk::DeviceSize m_chunkSize{};
    vk::DeviceSize m_allocateSize{};
    ChunkContainer chunks;

   protected:
    std::string gui_identifier;

    virtual ChunkIterator createNewChunk() = 0;

    virtual void headerInfo() {}

   public:
    BaseChunkAllocator(vk::Device _device, ChunkCreator* chunkAllocator, FitStrategy& strategy, Queue* _queue,
                       vk::DeviceSize chunkSize = 64 * 1024 * 1024)
        : BaseMemoryAllocator(chunkSize),
          m_device(_device),
          m_chunkAllocator(chunkAllocator),
          strategy(&strategy),
          queue(_queue),
          m_chunkSize(chunkSize),
          m_allocateSize(chunkSize),
          gui_identifier("")
    {
    }

    BaseChunkAllocator(BaseChunkAllocator&& other) noexcept
        : BaseMemoryAllocator(std::move(other)),
          m_device(other.m_device),
          m_chunkAllocator(other.m_chunkAllocator),
          strategy(other.strategy),
          queue(other.queue),
          m_chunkSize(other.m_chunkSize),
          m_allocateSize(other.m_allocateSize),
          chunks(std::move(other.chunks)),
          gui_identifier(std::move(other.gui_identifier))
    {
    }

    BaseChunkAllocator& operator=(BaseChunkAllocator&& other) noexcept
    {
        BaseMemoryAllocator::operator=(std::move(static_cast<BaseMemoryAllocator&&>(other)));
        m_device                     = other.m_device;
        m_chunkAllocator             = other.m_chunkAllocator;
        strategy                     = other.strategy;
        queue                        = other.queue;
        m_chunkSize                  = other.m_chunkSize;
        m_allocateSize               = other.m_allocateSize;
        chunks                       = std::move(other.chunks);
        gui_identifier               = std::move(other.gui_identifier);
        return *this;
    }

    ~BaseChunkAllocator() override = default;

    MemoryLocation* allocate(vk::DeviceSize size) override;

    void deallocate(MemoryLocation* location) override;

    void destroy() override;

    MemoryStats collectMemoryStats() override;

    void showDetailStats() override;
};


}  // namespace Saiga::Vulkan::Memory