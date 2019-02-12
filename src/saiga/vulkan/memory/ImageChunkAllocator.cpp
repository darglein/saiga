//
// Created by Peter Eichinger on 30.10.18.
//

#include "ImageChunkAllocator.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/imath.h"

namespace Saiga::Vulkan::Memory
{
ChunkIterator<MemoryLocation> Saiga::Vulkan::Memory::ImageChunkAllocator::createNewChunk()
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

Saiga::Vulkan::Memory::MemoryLocation* Saiga::Vulkan::Memory::ImageChunkAllocator::allocate(vk::DeviceSize size,
                                                                                            const vk::Image& image)
{
    auto image_mem_reqs = m_device.getImageMemoryRequirements(image);
    auto aligned_size   = Saiga::iAlignUp(size, image_mem_reqs.alignment);
    return BaseChunkAllocator::allocate(aligned_size);
}

}  // namespace Saiga::Vulkan::Memory
