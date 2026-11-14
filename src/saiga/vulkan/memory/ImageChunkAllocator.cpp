//
// Created by Peter Eichinger on 30.10.18.
//

#include "ImageChunkAllocator.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/easylogging++.h"

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
    std::scoped_lock lock(allocationMutex);
    const auto& requirements = image_data.image_requirements;

    // The chunk allocator places an allocation at the start of a free range, and all offsets are sums of allocation
    // sizes. Rounding every size up to a common alignment (a multiple of every image's alignment) therefore keeps
    // all offsets aligned, also for images with a larger alignment than the ones before.
    constexpr vk::DeviceSize common_alignment = 64 * 1024;
    if (requirements.alignment == 0 || common_alignment % requirements.alignment != 0)
    {
        VLOG(3) << "Unsupported image alignment " << requirements.alignment;
        return nullptr;
    }
    auto aligned_size = Saiga::iAlignUp(requirements.size, common_alignment);
    if (aligned_size > m_chunkSize)
    {
        return nullptr;
    }

    if (!hasInfo)
    {
        vk::MemoryAllocateInfo info;
        info.allocationSize  = m_chunkSize;
        info.memoryTypeIndex = findMemoryType(m_pDevice, requirements.memoryTypeBits, type.memoryFlags);
        allocateInfo         = info;
        hasInfo              = true;
    }
    if (((1u << allocateInfo.memoryTypeIndex) & requirements.memoryTypeBits) == 0)
    {
        // The memory type of this allocator can not be used for this image.
        VLOG(3) << "Image memory type bits " << requirements.memoryTypeBits << " do not contain the memory type "
                << allocateInfo.memoryTypeIndex << " of the allocator";
        return nullptr;
    }

    auto location = ChunkAllocator::base_allocate(aligned_size);

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
