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
    ImageChunkAllocator(const vk::Device& _device, Saiga::Vulkan::Memory::ChunkCreator* chunkAllocator,
                        const vk::MemoryPropertyFlags& _flags, Saiga::Vulkan::Memory::FitStrategy& strategy,
                        vk::DeviceSize chunkSize, bool _mapped)
        : BaseChunkAllocator(_device, chunkAllocator, _flags, strategy, chunkSize, _mapped)
    {
        LOG(INFO) << "Created new image allocator for flags " << vk::to_string(_flags);
        std::stringstream identifier_stream;
        identifier_stream << "Image Chunk " << vk::to_string(flags);
        gui_identifier = identifier_stream.str();
    }

    ImageChunkAllocator(const ImageChunkAllocator& other)     = delete;
    ImageChunkAllocator(ImageChunkAllocator&& other) noexcept = default;

    ImageChunkAllocator& operator=(const ImageChunkAllocator& other) = delete;
    ImageChunkAllocator& operator=(ImageChunkAllocator&& other) = default;

    ~ImageChunkAllocator() override = default;

   protected:
    ChunkIterator createNewChunk() override;

    void headerInfo() override;
};

}  // namespace Memory
}  // namespace Vulkan
}  // namespace Saiga
