//
// Created by Peter Eichinger on 30.10.18.
//

#include "ImageChunkAllocator.h"

#include "saiga/imgui/imgui.h"
#include "saiga/util/easylogging++.h"

Saiga::Vulkan::Memory::ChunkIterator Saiga::Vulkan::Memory::ImageChunkAllocator::createNewChunk()
{
    auto newChunk = m_chunkAllocator->allocate(flags, m_allocateSize);

    m_chunkAllocations.emplace_back(newChunk, vk::Buffer(), m_chunkSize, nullptr);

    return --m_chunkAllocations.end();
}

void Saiga::Vulkan::Memory::ImageChunkAllocator::headerInfo()
{
    ImGui::LabelText("Memory Type", "%s", vk::to_string(flags).c_str());
}
