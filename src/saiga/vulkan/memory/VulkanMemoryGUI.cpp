/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"
#include "saiga/vulkan/Base.h"

#include "VulkanMemory.h"

#include <vk_mem_alloc.h>

namespace Saiga::Vulkan::Memory
{
void VulkanMemory::renderGUI()
{
    if (!allocator) return;

    ImGuiTreeNodeFlags node_flags = ImGuiTreeNodeFlags_None;
    if (base->m_parameters.expand_memory_stats)
    {
        node_flags = ImGuiTreeNodeFlags_DefaultOpen;
    }
    if (!ImGui::CollapsingHeader("Memory Stats", node_flags))
    {
        return;
    }
    ImGui::Indent();

    // Allocations of saiga (blocks: the device memory allocated by VMA).
    const VkPhysicalDeviceMemoryProperties* properties = nullptr;
    vmaGetMemoryProperties(allocator, &properties);
    std::vector<VmaBudget> budgets(properties->memoryHeapCount);
    vmaGetHeapBudgets(allocator, budgets.data());

    for (uint32_t heap = 0; heap < properties->memoryHeapCount; ++heap)
    {
        const auto& budget = budgets[heap];
        const bool local   = properties->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
        ImGui::Text("Heap %u (%s, %s)", heap, local ? "device local" : "host",
                    sizeToString(properties->memoryHeaps[heap].size).c_str());
        float fraction = budget.statistics.blockBytes > 0
                             ? static_cast<float>(budget.statistics.allocationBytes) / budget.statistics.blockBytes
                             : 0.0f;
        ImGui::ProgressBar(fraction, ImVec2(-1, 0));
        ImGui::Text("%u allocations: %s used of %s in %u blocks", budget.statistics.allocationCount,
                    sizeToString(budget.statistics.allocationBytes).c_str(),
                    sizeToString(budget.statistics.blockBytes).c_str(), budget.statistics.blockCount);
    }

    ImGui::Separator();
    ImGui::Text("Defragmentation (dynamic allocations)");
    auto [dynamicBlocks, dynamicUsed] = dynamicMemoryUsage();
    ImGui::Text("Dynamic: %s used of %s", sizeToString(dynamicUsed).c_str(), sizeToString(dynamicBlocks).c_str());

    {
        std::scoped_lock lock(mutex);
        ImGui::Checkbox("Incremental defragmentation", &enableDefragmentation);
        ImGui::Text("Moved %u allocations (%s), freed %u blocks (%s)", defragStats.allocationsMoved,
                    sizeToString(defragStats.bytesMoved).c_str(), defragStats.deviceMemoryBlocksFreed,
                    sizeToString(defragStats.bytesFreed).c_str());
    }
    if (ImGui::Button("Full defragmentation"))
    {
        // The GUI is recorded into the current frame. Its command buffer may already reference the old handles, so
        // the moved resources are only swapped by the next update().
        std::scoped_lock lock(mutex);
        fullDefragRequested = true;
    }

    ImGui::Unindent();
}
}  // namespace Saiga::Vulkan::Memory
