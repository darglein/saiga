//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once

#include "saiga/imgui/imgui.h"
#include "saiga/vulkan/memory/BaseMemoryAllocator.h"

#include "BaseMemoryAllocator.h"
#include "ChunkCreator.h"
#include "Defragger.h"
#include "FitStrategy.h"

#include <mutex>


namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
class SAIGA_GLOBAL BaseChunkAllocator : public BaseMemoryAllocator
{
   private:
    std::mutex allocationMutex;
    std::unique_ptr<Defragger> defragger;
    void findNewMax(ChunkIterator& chunkAlloc) const;

    MemoryLocation createMemoryLocation(ChunkIterator iter, vk::DeviceSize start, vk::DeviceSize size);

   protected:
    vk::Device m_device;
    ChunkCreator* m_chunkAllocator{};
    FitStrategy* m_strategy{};

    vk::DeviceSize m_chunkSize{};
    vk::DeviceSize m_allocateSize{};
    ChunkContainer m_chunkAllocations;


    std::string gui_identifier;

    virtual ChunkIterator createNewChunk() = 0;

    virtual void headerInfo() {}

   public:
    BaseChunkAllocator(vk::Device _device, ChunkCreator* chunkAllocator, FitStrategy& strategy,
                       vk::DeviceSize chunkSize = 64 * 1024 * 1024)
        : BaseMemoryAllocator(chunkSize),
          defragger(std::make_unique<Defragger>(&m_chunkAllocations, &strategy)),
          m_device(_device),
          m_chunkAllocator(chunkAllocator),
          m_strategy(&strategy),
          m_chunkSize(chunkSize),
          m_allocateSize(chunkSize),
          gui_identifier("")
    {
        defragger->start();
    }

    BaseChunkAllocator(BaseChunkAllocator&& other) noexcept
        : BaseMemoryAllocator(std::move(other)),
          m_device(other.m_device),
          m_chunkAllocator(other.m_chunkAllocator),
          m_strategy(other.m_strategy),
          m_chunkSize(other.m_chunkSize),
          m_allocateSize(other.m_allocateSize),
          m_chunkAllocations(std::move(other.m_chunkAllocations)),
          gui_identifier(std::move(other.gui_identifier))
    {
    }

    BaseChunkAllocator& operator=(BaseChunkAllocator&& other) noexcept
    {
        BaseMemoryAllocator::operator=(std::move(static_cast<BaseMemoryAllocator&&>(other)));
        m_device                     = other.m_device;
        m_chunkAllocator             = other.m_chunkAllocator;
        m_strategy                   = other.m_strategy;
        m_chunkSize                  = other.m_chunkSize;
        m_allocateSize               = other.m_allocateSize;
        m_chunkAllocations           = std::move(other.m_chunkAllocations);
        gui_identifier               = std::move(other.gui_identifier);
        return *this;
    }

    ~BaseChunkAllocator() override = default;

    MemoryLocation allocate(vk::DeviceSize size) override;

    void deallocate(MemoryLocation& location) override;

    void destroy() override;

    MemoryStats collectMemoryStats() override;

    void showDetailStats() override;

    void enable_defragger(bool enable) { defragger->setEnabled(enable); }
};


}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga