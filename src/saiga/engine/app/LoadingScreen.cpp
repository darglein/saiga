/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/app/LoadingScreen.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/app/EngineApp.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace Saiga
{
namespace Engine
{
bool loadEnds(const SpaceLoading& loading, double r, size_t uploadsPending, uint64_t maxFrames)
{
    return (r >= 1 && uploadsPending == 0) || loading.frames >= maxFrames;
}

void drawLoadingScreen(EngineApp& app, const SpaceLoading& loading)
{
    ImDrawList* dl          = ImGui::GetBackgroundDrawList();
    const ImVec2 size       = ImGui::GetIO().DisplaySize;
    const ImU32 background  = IM_COL32(12, 13, 16, 255);
    const ImU32 barEmpty    = IM_COL32(48, 50, 58, 255);
    const ImU32 barFull     = IM_COL32(200, 170, 90, 255);
    const ImU32 textColor   = IM_COL32(230, 230, 235, 255);
    const World& w          = app.world();
    const std::string name  = loading.space < w.spaceCount() ? w.spaceTemplate(loading.space).name : "?";
    const std::string title = "Loading " + name;
    dl->AddRectFilled(ImVec2(0, 0), size, background);
    const float barWidth  = std::min(480.0f, size.x * 0.6f);
    const float barHeight = 10;
    const ImVec2 center(size.x * 0.5f, size.y * 0.5f);
    const ImVec2 textSize = ImGui::CalcTextSize(title.c_str());
    dl->AddText(ImVec2(center.x - textSize.x * 0.5f, center.y - textSize.y - 14), textColor, title.c_str());
    const ImVec2 a(center.x - barWidth * 0.5f, center.y), b(center.x + barWidth * 0.5f, center.y + barHeight);
    dl->AddRectFilled(a, b, barEmpty, 3);
    const float p = float(std::clamp(loading.progress, 0.0, 1.0));
    if (p > 0) dl->AddRectFilled(a, ImVec2(a.x + barWidth * p, b.y), barFull, 3);
    char percent[16];
    std::snprintf(percent, sizeof(percent), "%d %%", int(p * 100));
    const ImVec2 ps = ImGui::CalcTextSize(percent);
    dl->AddText(ImVec2(center.x - ps.x * 0.5f, b.y + 8), textColor, percent);
}

}  // namespace Engine
}  // namespace Saiga
