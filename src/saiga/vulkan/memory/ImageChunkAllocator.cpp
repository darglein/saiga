//
// Created by Peter Eichinger on 30.10.18.
//

#include "ImageChunkAllocator.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/imath.h"

namespace Saiga::Vulkan::Memory
{
ChunkIterator<ImageMemoryLocation> ImageChunkAllocator::createNewChunk()
{
    auto newChunk = m_chunkAllocator->allocate(type.memoryFlags, m_allocateSize);

    chunks.emplace_back(newChunk, vk::Buffer(), m_chunkSize, nullptr);

    return --chunks.end();
}

void Saiga::Vulkan::Memory::ImageChunkAllocator::headerInfo()
{
    std::stringstream ss;
    ss << type;
    ImGui::LabelText("Memory Type", "%s", ss.str().c_str());
}

ImageMemoryLocation* ImageChunkAllocator::allocate(ImageData& image_data)
{
    auto aligned_size = Saiga::iAlignUp(image_data.image_requirements.size, image_data.image_requirements.alignment);
    auto location     = BaseChunkAllocator::base_allocate(aligned_size);

    bind_image_data(m_device, location, std::move(image_data));

    location->data.create_view(m_device);
    location->data.create_sampler(m_device);

    VLOG(1) << "Allocated image" << *location;
    return location;
}

std::unique_ptr<ImageMemoryLocation> ImageChunkAllocator::create_location(
    ChunkIterator<ImageMemoryLocation>& chunk_alloc, vk::DeviceSize start, vk::DeviceSize size)
{
    return std::make_unique<ImageMemoryLocation>(nullptr, chunk_alloc->chunk->memory, start, size,
                                                 chunk_alloc->mappedPointer);
}

void ImageChunkAllocator::deallocate(ImageMemoryLocation* location)
{
    VLOG(1) << "Trying to deallocate image" << *location;
    BaseChunkAllocator::deallocate(location);
}

}  // namespace Saiga::Vulkan::Memory
