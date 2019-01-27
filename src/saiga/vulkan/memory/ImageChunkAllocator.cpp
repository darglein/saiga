//
// Created by Peter Eichinger on 30.10.18.
//

#include "ImageChunkAllocator.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/easylogging++.h"
#include "saiga/util/imath.h"
Saiga::Vulkan::Memory::ChunkIterator Saiga::Vulkan::Memory::ImageChunkAllocator::createNewChunk()
{
    auto newChunk = m_chunkAllocator->allocate(type.memoryFlags, m_allocateSize);

    m_chunkAllocations.emplace_back(newChunk, vk::Buffer(), m_chunkSize, nullptr);

    return --m_chunkAllocations.end();
}

void Saiga::Vulkan::Memory::ImageChunkAllocator::headerInfo()
{
    ImGui::LabelText("Memory Type", "%s", type.to_string().c_str());
}

Saiga::Vulkan::Memory::MemoryLocation Saiga::Vulkan::Memory::ImageChunkAllocator::allocate(vk::DeviceSize size, const vk::Image &image)
{
    auto image_mem_reqs = m_device.getImageMemoryRequirements(image);
    auto aligned_size = Saiga::iAlignUp(size,image_mem_reqs.alignment);
    return BaseChunkAllocator::allocate(aligned_size);
}
