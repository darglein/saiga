//
// Created by Peter Eichinger on 2018-12-21.
//


#include "saiga/imgui/imgui.h"
#include "saiga/util/tostring.h"

#include "Base.h"
using namespace Saiga::Vulkan;
void VulkanBase::renderGUI()
{
    if (!ImGui::CollapsingHeader("Device Stats"))
    {
        return;
    }

    ImGui::Indent();
    ImGui::LabelText("Device Name", "%s", physicalDevice.getProperties().deviceName);


    ImGui::Text("Heaps");
    auto memProps = physicalDevice.getMemoryProperties();

    ImGui::Indent();

    ImGui::Columns(3, "HEAPINFO");
    ImGui::SetColumnWidth(0, ImGui::GetFontSize() * 3);
    ImGui::SetColumnWidth(1, ImGui::GetFontSize() * 6);

    ImGui::Text("ID");
    ImGui::NextColumn();
    ImGui::Text("Size");
    ImGui::NextColumn();
    ImGui::Text("Heap Flags");
    ImGui::NextColumn();
    for (uint32_t heapIdx = 0U; heapIdx < memProps.memoryHeapCount; ++heapIdx)
    {
        ImGui::Separator();

        auto& heap = memProps.memoryHeaps[heapIdx];

        ImGui::Text("%d", heapIdx);
        ImGui::NextColumn();
        ImGui::Text("%s", sizeToString(heap.size).c_str());
        ImGui::NextColumn();
        ImGui::TextWrapped("%s", vk::to_string(heap.flags).c_str());
        ImGui::NextColumn();
    }

    ImGui::Columns(1);
    ImGui::Unindent();


    ImGui::Spacing();
    ImGui::Text("Memory Types");
    ImGui::Indent();

    ImGui::Columns(3, "MEMINFO");
    ImGui::SetColumnWidth(0, ImGui::GetFontSize() * 3);
    ImGui::SetColumnWidth(1, ImGui::GetFontSize() * 6);


    ImGui::Text("ID");
    ImGui::NextColumn();
    ImGui::Text("Heap Idx");
    ImGui::NextColumn();
    ImGui::TextWrapped("Property Flags");
    ImGui::NextColumn();
    for (auto typeIdx = 0U; typeIdx < memProps.memoryTypeCount; ++typeIdx)
    {
        ImGui::Separator();

        auto& type = memProps.memoryTypes[typeIdx];

        ImGui::Text("%d", typeIdx);
        ImGui::NextColumn();
        ImGui::Text("%u", type.heapIndex);
        ImGui::NextColumn();
        ImGui::TextWrapped("%s", vk::to_string(type.propertyFlags).c_str());
        ImGui::NextColumn();
    }
    ImGui::Unindent();

    ImGui::Columns(1);

    ImGui::Spacing();

    ImGui::Text("Available Queues");
    ImGui::Indent();

    auto queueFamilyProps = physicalDevice.getQueueFamilyProperties();

    ImGui::Columns(3, "QUEUEINFO");
    ImGui::SetColumnWidth(0, ImGui::GetFontSize() * 3);
    ImGui::SetColumnWidth(1, ImGui::GetFontSize() * 6);

    ImGui::Text("ID");
    ImGui::NextColumn();
    ImGui::Text("Count");
    ImGui::NextColumn();
    ImGui::Text("Type");
    ImGui::NextColumn();

    size_t index = 0;
    for (auto& prop : queueFamilyProps)
    {
        ImGui::Separator();
        ImGui::Text("%lu", index);
        ImGui::NextColumn();
        ImGui::Text("%u", prop.queueCount);
        ImGui::NextColumn();
        ImGui::TextWrapped("%s", vk::to_string(prop.queueFlags).c_str());
        ImGui::NextColumn();
        ++index;
    }
    ImGui::Columns(1);
    ImGui::Spacing();

    ImGui::Unindent();


    ImGui::Spacing();

    ImGui::Indent();

    ImGui::Text("Used queues");

    ImGui::Text("Main: %d, %d", main_queue_info.first, main_queue_info.second);
    ImGui::Text("Transfer: %d, %d", transfer_info.first, transfer_info.second);
    ImGui::Text("Compute: %d, %d", compute_info.first, compute_info.second);

    ImGui::Unindent();

    ImGui::Unindent();
}
