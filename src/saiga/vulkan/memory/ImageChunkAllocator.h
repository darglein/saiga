//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include "BaseChunkAllocator.h"
#include "ImageMemoryLocation.h"
#include "MemoryType.h"
namespace Saiga::Vulkan::Memory
{
class SAIGA_VULKAN_API ImageChunkAllocator final : public ChunkAllocator<ImageMemoryLocation>
{
   protected:
    ChunkIterator<ImageMemoryLocation> createNewChunk() override;

    void headerInfo() override;

    std::unique_ptr<ImageMemoryLocation> create_location(ChunkIterator<ImageMemoryLocation>& chunk_alloc,
                                                         vk::DeviceSize start, vk::DeviceSize size) override;

   public:
    ImageType type;
    ImageChunkAllocator(const vk::Device& _device, ChunkCreator* chunkAllocator, ImageType _type,
                        FitStrategy<ImageMemoryLocation>& strategy, Queue* _queue, vk::DeviceSize chunkSize)
        : ChunkAllocator(_device, chunkAllocator, strategy, _queue, chunkSize), type(std::move(_type))
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

    ImageMemoryLocation* allocate(ImageData& image);

    void deallocate(ImageMemoryLocation* location) override;
};

}  // namespace Saiga::Vulkan::Memory
