//
// Created by Peter Eichinger on 2019-01-21.
//

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/easylogging++.h"
#include "saiga/core/util/tostring.h"

#include "BaseChunkAllocator.h"
namespace Saiga::Vulkan::Memory
{
void BaseChunkAllocator::showDetailStats()
{
    using BarColor = ImGui::ColoredBar::BarColor;
    static const BarColor alloc_color_static{{1.00f, 0.447f, 0.133f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}};
    static const BarColor alloc_color_dynamic{{1.00f, 0.812f, 0.133f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}};

    static std::vector<ImGui::ColoredBar> allocation_bars;

    if (ImGui::CollapsingHeader(gui_identifier.c_str()))
    {
        std::scoped_lock lock(allocationMutex);
        ImGui::Indent();

        headerInfo();

        allocation_bars.resize(
            chunks.size(), ImGui::ColoredBar({0, 40}, {{0.1f, 0.1f, 0.1f, 1.0f}, {0.4f, 0.4f, 0.4f, 1.0f}}, true, 1));

        int numAllocs           = 0;
        uint64_t usedSpace      = 0;
        uint64_t innerFreeSpace = 0;
        uint64_t totalFreeSpace = 0;
        for (auto i = 0U; i < allocation_bars.size(); ++i)
        {
            auto& bar   = allocation_bars[i];
            auto& chunk = chunks[i];

            std::stringstream ss;
            ss << "Mem " << std::hex << chunk.chunk->memory << " Buffer " << chunk.buffer;

            ImGui::Text("Chunk %d (%s, %s) %s", i + 1, sizeToString(chunk.getFree()).c_str(),
                        sizeToString(chunk.allocated).c_str(), ss.str().c_str());
            ImGui::Indent();
            bar.renderBackground();
            int j = 0;
            ConstAllocationIterator allocIter;
            ConstFreeIterator freeIter;
            for (allocIter = chunk.allocations.cbegin(), j = 0; allocIter != chunk.allocations.cend(); ++allocIter, ++j)
            {
                auto& color = (*allocIter)->is_static() ? alloc_color_static : alloc_color_dynamic;
                bar.renderArea(static_cast<float>((*allocIter)->offset) / m_chunkSize,
                               static_cast<float>((*allocIter)->offset + (*allocIter)->size) / m_chunkSize, color,
                               false);
                usedSpace += (*allocIter)->size;
            }
            numAllocs += j;

            if (!chunk.freeList.empty())
            {
                auto freeEnd = --chunk.freeList.cend();
                for (freeIter = chunk.freeList.cbegin(); freeIter != freeEnd; freeIter++)
                {
                    innerFreeSpace += freeIter->size;
                    totalFreeSpace += freeIter->size;
                }

                totalFreeSpace += chunk.freeList.back().size;
            }
            ImGui::Unindent();
        }
        ImGui::LabelText("Number of allocations", "%d", numAllocs);
        auto totalSpace = m_chunkSize * chunks.size();


        ImGui::LabelText("Usage", "%s / %s (%.2f%%)", sizeToString(usedSpace).c_str(), sizeToString(totalSpace).c_str(),
                         100 * static_cast<float>(usedSpace) / totalSpace);
        ImGui::LabelText("Free Space (total / fragmented)", "%s / %s", sizeToString(totalFreeSpace).c_str(),
                         sizeToString(innerFreeSpace).c_str());


        ImGui::Unindent();
    }
}

MemoryStats BaseChunkAllocator::collectMemoryStats()
{
    std::scoped_lock lock(allocationMutex);
    int numAllocs                = 0;
    uint64_t usedSpace           = 0;
    uint64_t fragmentedFreeSpace = 0;
    uint64_t totalFreeSpace      = 0;
    for (auto& chunk : chunks)
    {
        int j = 0;
        ConstAllocationIterator allocIter;
        ConstFreeIterator freeIter;
        for (allocIter = chunk.allocations.cbegin(), j = 0; allocIter != chunk.allocations.cend(); ++allocIter, ++j)
        {
            usedSpace += (*allocIter)->size;
        }
        numAllocs += j;

        if (!chunk.freeList.empty())
        {
            auto freeEnd = --chunk.freeList.cend();

            for (freeIter = chunk.freeList.cbegin(); freeIter != freeEnd; freeIter++)
            {
                fragmentedFreeSpace += freeIter->size;
                totalFreeSpace += freeIter->size;
            }

            totalFreeSpace += chunk.freeList.back().size;
        }
    }
    auto totalSpace = m_chunkSize * chunks.size();
    //
    return MemoryStats{totalSpace, usedSpace, fragmentedFreeSpace};
}
}  // namespace Saiga::Vulkan::Memory
