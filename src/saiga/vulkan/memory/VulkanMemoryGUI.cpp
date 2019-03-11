//
// Created by Peter Eichinger on 2019-01-21.
//


#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Base.h"

#include "VulkanMemory.h"

namespace Saiga::Vulkan::Memory
{
void VulkanMemory::renderGUI()
{
    static bool mem_stats_open = base->m_parameters.expand_memory_stats;

    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_None;
    if (mem_stats_open)
    {
        node_flags = ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (!ImGui::CollapsingHeader("Memory Stats", node_flags))
    {
        return;
    }

    ImGui::Indent();

    static std::unordered_map<vk::MemoryPropertyFlags, MemoryStats> memoryTypeStats;


    for (auto& entry : memoryTypeStats)
    {
        entry.second = MemoryStats();
    }


    for (auto& allocator : bufferAllocators)
    {
        if (memoryTypeStats.find(allocator.first.memoryFlags) == memoryTypeStats.end())
        {
            memoryTypeStats[allocator.first.memoryFlags] = MemoryStats();
        }
        memoryTypeStats[allocator.first.memoryFlags] += allocator.second.allocator->collectMemoryStats();
    }
    for (auto& allocator : imageAllocators)
    {
        if (memoryTypeStats.find(allocator.first.memoryFlags) == memoryTypeStats.end())
        {
            memoryTypeStats[allocator.first.memoryFlags] = MemoryStats();
        }
        memoryTypeStats[allocator.first.memoryFlags] += allocator.second.allocator->collectMemoryStats();
    }

    static std::vector<ImGui::ColoredBar> bars;

    bars.resize(memoryTypeStats.size(),
                ImGui::ColoredBar({0, 16}, {{0.1f, 0.1f, 0.1f, 1.0f}, {0.4f, 0.4f, 0.4f, 1.0f}}, true));
    int index = 0;
    for (auto& memStat : memoryTypeStats)
    {
        ImGui::Text("%s", vk::to_string(memStat.first).c_str());
        // ImGui::SameLine(150.0f);
        auto& bar = bars[index];

        bar.renderBackground();
        bar.renderArea(0.0f, static_cast<float>(memStat.second.used) / memStat.second.allocated,
                       ImGui::ColoredBar::BarColor{{0.0f, 0.2f, 0.2f, 1.0f}, {0.133f, 0.40f, 0.40f, 1.0f}});

        ImGui::Text("%s / %s (%s fragmented free)", sizeToString(memStat.second.used).c_str(),
                    sizeToString(memStat.second.allocated).c_str(), sizeToString(memStat.second.fragmented).c_str());

        index++;
    }


    ImGui::Unindent();

    ImGui::Spacing();


    ImGui::Indent();
    if (!ImGui::CollapsingHeader("Detailed Memory Stats", node_flags))
    {
        return;
    }

    ImGui::Indent();

    for (auto& allocator : bufferAllocators)
    {
        allocator.second.allocator->showDetailStats(mem_stats_open);
    }
    for (auto& allocator : imageAllocators)
    {
        allocator.second.allocator->showDetailStats(mem_stats_open);
    }

    ImGui::Unindent();
    ImGui::Unindent();
}
}  // namespace Saiga::Vulkan::Memory
