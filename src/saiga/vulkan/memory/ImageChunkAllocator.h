//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include "BaseChunkAllocator.h"
#include "MemoryType.h"
namespace Saiga::Vulkan::Memory
{
class SAIGA_GLOBAL ImageChunkAllocator final : public BaseChunkAllocator
{
   public:
    ImageType type;
    ImageChunkAllocator(const vk::Device& _device, ChunkCreator* chunkAllocator, ImageType _type, FitStrategy& strategy,
                        Queue* _queue, vk::DeviceSize chunkSize)
        : BaseChunkAllocator(_device, chunkAllocator, strategy, _queue, chunkSize), type(std::move(_type))
    {
        LOG(INFO) << "Created new image allocator for flags " << type;
        std::stringstream identifier_stream;
        identifier_stream << "Image Chunk " << type;
        gui_identifier = identifier_stream.str();
    }

    ImageChunkAllocator(const ImageChunkAllocator& other) = delete;
    ImageChunkAllocator(ImageChunkAllocator&& other)      = default;

    ImageChunkAllocator& operator=(const ImageChunkAllocator& other) = delete;
    ImageChunkAllocator& operator=(ImageChunkAllocator&& other) = default;

    ~ImageChunkAllocator() override = default;

    MemoryLocation* allocate(vk::DeviceSize size, const vk::Image& image);

   protected:
    ChunkIterator createNewChunk() override;

    void headerInfo() override;

   private:
    using BaseChunkAllocator::allocate;
};

}  // namespace Saiga::Vulkan::Memory
