/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/app/DebugUiStyle.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace Saiga
{
namespace Engine
{
namespace DebugUiStyle
{
namespace
{
int pushedColors = 0;

bool startsWith(const std::string& s, const char* prefix)
{
    size_t i = 0;
    for (; prefix[i]; ++i)
        if (i >= s.size() || std::tolower((unsigned char)s[i]) != prefix[i]) return false;
    return true;
}
}  // namespace

ImU32 categoryColor(const std::string& category, const std::string& message)
{
    if (startsWith(message, "error") || startsWith(message, "could not") || startsWith(message, "failed"))
        return kError;
    if (startsWith(message, "warning")) return kWarning;
    // A failed tool call
    if (category == "mcp") return message.find(" -> error") != std::string::npos ? kError : kMcp;
    if (category == "api") return kApi;
    return ImGui::GetColorU32(ImGuiCol_Text);
}

void push()
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 4));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 3.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
    // A neutral palette of its own (the app's ImGui theme may be colorful); the quantities keep their colors
    const std::pair<ImGuiCol, ImU32> colors[] = {
        {ImGuiCol_Text, IM_COL32(225, 228, 232, 255)},
        {ImGuiCol_TextDisabled, IM_COL32(138, 144, 152, 255)},
        {ImGuiCol_WindowBg, IM_COL32(26, 28, 33, 255)},
        {ImGuiCol_ChildBg, IM_COL32(0, 0, 0, 0)},
        {ImGuiCol_PopupBg, IM_COL32(30, 33, 39, 245)},
        {ImGuiCol_Border, IM_COL32(70, 76, 88, 255)},
        {ImGuiCol_FrameBg, IM_COL32(40, 44, 52, 255)},
        {ImGuiCol_FrameBgHovered, IM_COL32(52, 58, 68, 255)},
        {ImGuiCol_FrameBgActive, IM_COL32(62, 70, 84, 255)},
        {ImGuiCol_Button, IM_COL32(56, 62, 74, 255)},
        {ImGuiCol_ButtonHovered, IM_COL32(74, 84, 100, 255)},
        {ImGuiCol_ButtonActive, IM_COL32(66, 112, 186, 255)},
        {ImGuiCol_Header, IM_COL32(48, 54, 66, 255)},
        {ImGuiCol_HeaderHovered, IM_COL32(60, 68, 82, 255)},
        {ImGuiCol_HeaderActive, IM_COL32(70, 80, 96, 255)},
        {ImGuiCol_CheckMark, IM_COL32(110, 180, 255, 255)},
        {ImGuiCol_SliderGrab, IM_COL32(100, 150, 215, 255)},
        {ImGuiCol_SliderGrabActive, IM_COL32(130, 185, 255, 255)},
        {ImGuiCol_Separator, IM_COL32(70, 76, 88, 255)},
        {ImGuiCol_SeparatorHovered, IM_COL32(110, 160, 230, 255)},
        {ImGuiCol_SeparatorActive, IM_COL32(130, 185, 255, 255)},
        {ImGuiCol_TableHeaderBg, IM_COL32(44, 48, 57, 255)},
        {ImGuiCol_TableRowBgAlt, IM_COL32(255, 255, 255, 10)},
        {ImGuiCol_TableBorderLight, IM_COL32(60, 65, 75, 255)},
        {ImGuiCol_TableBorderStrong, IM_COL32(70, 76, 88, 255)},
        {ImGuiCol_ScrollbarBg, IM_COL32(0, 0, 0, 0)},
        {ImGuiCol_ScrollbarGrab, IM_COL32(70, 76, 88, 255)},
        {ImGuiCol_PlotLines, IM_COL32(120, 230, 120, 255)},
    };
    for (const auto& [c, v] : colors) ImGui::PushStyleColor(c, v);
    pushedColors = int(sizeof(colors) / sizeof(colors[0]));
}

void pop()
{
    ImGui::PopStyleColor(pushedColors);
    ImGui::PopStyleVar(7);
}

bool section(const char* title, bool& open)
{
    ImGui::Spacing();
    ImGui::SetNextItemOpen(open, ImGuiCond_Always);
    open = ImGui::CollapsingHeader(title);
    return open;
}

bool beginProperties(const char* id)
{
    if (!ImGui::BeginTable(id, 2, ImGuiTableFlags_None)) return false;
    // The labels in a fixed column (short labels: the values start close to them), the values take the rest
    ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed, 8.5f * ImGui::GetFontSize());
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
    return true;
}

void property(const char* label)
{
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("%s", label);
    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(-1);
}

void endProperties()
{
    ImGui::EndTable();
}

void coloredText(ImU32 color, const char* text)
{
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void cellRight(const char* text)
{
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetColumnWidth() - w));
    ImGui::TextUnformatted(text);
}

void headerRow(std::initializer_list<const char*> names)
{
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    bool first = true;
    for (const char* n : names)
    {
        if (first)
        {
            ImGui::TableNextColumn();
            ImGui::TableHeader(n);
            first = false;
        }
        else
            cellRight(n);
    }
}

void emptyState(const char* text)
{
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const ImVec2 size  = ImGui::CalcTextSize(text);
    ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + std::max(0.0f, (avail.x - size.x) / 2),
                               ImGui::GetCursorPosY() + std::max(0.0f, (avail.y - size.y) / 2)));
    ImGui::TextDisabled("%s", text);
}

}  // namespace DebugUiStyle
}  // namespace Engine
}  // namespace Saiga
