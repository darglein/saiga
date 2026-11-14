//
// Created by Peter Eichinger on 30.10.18.
//

#pragma once
#include "ChunkAllocator.h"
#include "FindMemoryType.h"
#include "ImageMemoryLocation.h"
#include "MemoryType.h"

namespace Saiga::Vulkan::Memory
{
class SAIGA_VULKAN_API ImageChunkAllocator final : public ChunkAllocator<ImageMemoryLocation>
{
   private:
    bool hasInfo;
    vk::MemoryAllocateInfo allocateInfo;

   protected:
    ChunkIterator<ImageMemoryLocation> createNewChunk() override;

    void headerInfo() override;

    std::unique_ptr<ImageMemoryLocation> create_location(ChunkIterator<ImageMemoryLocation>& chunk_alloc,
                                                         vk::DeviceSize start, vk::DeviceSize size) override;

   public:
    ImageType type;
    ImageChunkAllocator(vk::PhysicalDevice _pDevice, const vk::Device& _device, ImageType _type,
                        FitStrategy<ImageMemoryLocation>& strategy, Queue* _queue, vk::DeviceSize chunkSize);

    ImageChunkAllocator(const ImageChunkAllocator& other) = delete;
    ImageChunkAllocator(ImageChunkAllocator&& other)      = default;

    ImageChunkAllocator& operator=(const ImageChunkAllocator& other) = delete;
    ImageChunkAllocator& operator=(ImageChunkAllocator&& other)      = default;

    ~ImageChunkAllocator() override = default;

    /**
     * Allocates memory for the (created) image and binds it. Returns nullptr if the image can not be allocated by
     * this allocator (memory type, alignment or size), the image is not changed then.
     */
    ImageMemoryLocation* allocate(ImageData& image);

    void deallocate(ImageMemoryLocation* location) override;
};

}  // namespace Saiga::Vulkan::Memory
