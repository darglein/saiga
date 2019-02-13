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
    image_data.create_image(m_device);
    auto aligned_size = Saiga::iAlignUp(image_data.image_requirements.size, image_data.image_requirements.alignment);
    auto location     = BaseChunkAllocator::base_allocate(aligned_size);

    location->data = std::move(image_data);

    bind_image_data(m_device, location, image_data);

    location->data.create_view(m_device);
    // m_device.bindImageMemory(location->data.image, location->memory, location->offset);

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
    location->destroy_data(m_device);

    BaseChunkAllocator::base_deallocate(location);
}

}  // namespace Saiga::Vulkan::Memory
