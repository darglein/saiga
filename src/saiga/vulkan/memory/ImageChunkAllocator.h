//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include "BaseChunkAllocator.h"

namespace Saiga
{
namespace Vulkan
{
namespace Memory
{
class SAIGA_GLOBAL ImageChunkAllocator : public BaseChunkAllocator
{
   public:
    ~ImageChunkAllocator() override = default;

    ImageChunkAllocator(const vk::Device& _device,
                                                                    Saiga::Vulkan::Memory::ChunkCreator* chunkAllocator,
                                                                    const vk::MemoryPropertyFlags& _flags,
                                                                    Saiga::Vulkan::Memory::FitStrategy& strategy,
                                                                    vk::DeviceSize chunkSize, bool _mapped)
            : BaseChunkAllocator(_device, chunkAllocator, _flags, strategy, chunkSize, _mapped)
    {
        LOG(INFO) << "Created new image allocator for flags " << vk::to_string(_flags);
    }

    ImageChunkAllocator(ImageChunkAllocator&& other) noexcept : BaseChunkAllocator(std::move(other)) {}

   protected:
    ChunkIterator createNewChunk() override;
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
