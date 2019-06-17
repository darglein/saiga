//
// Created by Peter Eichinger on 30.10.18.
//

#include "ImageChunkAllocator.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/math/imath.h"

#include "SafeAllocator.h"

namespace Saiga::Vulkan::Memory
{
ChunkIterator<ImageMemoryLocation> ImageChunkAllocator::createNewChunk()
{
    auto newMem = SafeAllocator::instance()->allocateMemory(m_device, allocateInfo);

    chunks.emplace_back(newMem, vk::Buffer(), m_chunkSize, nullptr);

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
    if (!hasInfo)
    {
        auto memRequirements = image_data.image_requirements;

        vk::MemoryAllocateInfo info;
        info.allocationSize  = m_chunkSize;
        info.memoryTypeIndex = findMemoryType(m_pDevice, memRequirements.memoryTypeBits, type.memoryFlags);
        allocateInfo         = info;
        hasInfo              = true;
    }
    auto aligned_size = Saiga::iAlignUp(image_data.image_requirements.size, image_data.image_requirements.alignment);
    auto location     = ChunkAllocator::base_allocate(aligned_size);

    bind_image_data(m_device, location, std::move(image_data));

    location->data.create_view(m_device);
    location->data.create_sampler(m_device);

    VLOG(3) << "Allocated image" << *location;
    return location;
}

std::unique_ptr<ImageMemoryLocation> ImageChunkAllocator::create_location(
    ChunkIterator<ImageMemoryLocation>& chunk_alloc, vk::DeviceSize start, vk::DeviceSize size)
{
    return std::make_unique<ImageMemoryLocation>(nullptr, chunk_alloc->memory, start, size, chunk_alloc->mappedPointer);
}

ImageChunkAllocator::ImageChunkAllocator(vk::PhysicalDevice _pDevice, const vk::Device& _device, ImageType _type,
                                         FitStrategy<ImageMemoryLocation>& strategy, Queue* _queue,
                                         vk::DeviceSize chunkSize)
    : ChunkAllocator(_pDevice, _device, strategy, _queue, chunkSize), hasInfo(false), type(std::move(_type))
{
    VLOG(3) << "Created new image allocator for flags " << type;
    std::stringstream identifier_stream;
    identifier_stream << "Image Chunk " << type;
    gui_identifier = identifier_stream.str();
}

void ImageChunkAllocator::deallocate(ImageMemoryLocation* location)
{
    VLOG(3) << "Trying to deallocate image" << *location;
    ChunkAllocator::deallocate(location);
}

}  // namespace Saiga::Vulkan::Memory
