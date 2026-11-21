/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>

namespace Saiga
{
namespace Vulkan
{
namespace
{
// "1.23 M", "45.6 k" or "789"
std::string compact(double v)
{
    char buf[32];
    if (v >= 1e6)
        std::snprintf(buf, sizeof(buf), "%.2f M", v / 1e6);
    else if (v >= 1e4)
        std::snprintf(buf, sizeof(buf), "%.1f k", v / 1e3);
    else
        std::snprintf(buf, sizeof(buf), "%.0f", v);
    return buf;
}

std::string milliseconds(double ms)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3f", ms);
    return buf;
}

void cell(const char* fmt, ...) IM_FMTARGS(1);
void cell(const char* fmt, ...)
{
    ImGui::TableNextColumn();
    va_list args;
    va_start(args, fmt);
    ImGui::TextV(fmt, args);
    va_end(args);
}

void cellRight(const std::string& text)
{
    ImGui::TableNextColumn();
    float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetColumnWidth() - w));
    ImGui::TextUnformatted(text.c_str());
}

bool beginTable(const char* id, int columns)
{
    return ImGui::BeginTable(id, columns,
                             ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuterH |
                                 ImGuiTableFlags_SizingStretchProp);
}

// A bar of a share with its label at a fixed place (ImGui's ProgressBar moves the label with the fill)
void shareBar(float share, const char* label, ImU32 color)
{
    const ImVec2 p0   = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const float h     = ImGui::GetFrameHeight();
    ImGui::Dummy(ImVec2(width, h));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p0, ImVec2(p0.x + width, p0.y + h), ImGui::GetColorU32(ImGuiCol_FrameBg));
    dl->AddRectFilled(p0, ImVec2(p0.x + width * std::clamp(share, 0.0f, 1.0f), p0.y + h), color);
    const ImVec2 at(p0.x + 6, p0.y + (h - ImGui::GetTextLineHeight()) / 2);
    dl->AddText(ImVec2(at.x + 1, at.y + 1), IM_COL32(0, 0, 0, 200), label);
    dl->AddText(at, IM_COL32(255, 255, 255, 255), label);
}

uint32_t treesVisible(const PrototypeForest::Stats& s)
{
    return s.visible[0] + s.visible[1] + s.visible[2];
}

}  // namespace

namespace
{
// A sub-section of the world statistics: a tree node without indent (the debug UI's section header is the parent)
bool subSection(const char* label, bool defaultOpen)
{
    return ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                        (defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None));
}
}  // namespace

void WorldRenderer::imguiStatistics()
{
    // Widgets only (no window): a section of the debug UI's Performance tab (S19c)
    if (!renderScene)
    {
        ImGui::TextDisabled("not attached to a RenderScene");
        return;
    }
    const Shown& shown              = shownStatistics();
    const Stats& s                  = statistics;
    const PrototypeWorld* pw        = renderScene->prototypeWorld();
    const PrototypeWorld::Stats ps  = pw ? pw->stats() : PrototypeWorld::Stats();
    const PrototypeForest::Stats fs = trees.stats();

    // ---------------------------------------------------------------------------------------- Overview
    // The frame times, CPU and GPU busy are in the debug UI's Performance tab (one place for them); this section breaks
    // the world's part down. The times are means over the last kShownSeconds, refreshed every kRefreshSeconds (numbers
    // that change every frame cannot be read)
    ImGui::TextDisabled("%.1f km world, %d trees%s, %d draw calls", heightField().worldSize() / 1000.0f,
                        trees.treeCount(), settings.depthPrePass ? ", depth pre-pass" : "", s.drawCalls);

    // ---------------------------------------------------------------------------------------- GPU sections
    char header[64];
    std::snprintf(header, sizeof(header), "GPU sections (mean of %.0f s)###gpu sections", kShownSeconds);
    if (subSection(header, true))
    {
        if (shown.gpuSectionMs.empty())
        {
            ImGui::TextDisabled("No GPU times (no timestamps, or the app does not take them).");
        }
        else if (beginTable("gpu sections", 3))
        {
            ImGui::TableSetupColumn("section", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_WidthStretch, 1.5f);
            ImGui::TableHeadersRow();
            const auto& names  = RenderScene::gpuSectionNames();
            const double total = std::max(shown.gpuMs, 1e-6);
            // A pass's share of the frame; a sub-section's (dotted name) share of its pass, indented and dimmer. A
            // pass below 5 us (off, e.g. the depth pre-pass) has no meaningful shares
            constexpr double kOff = 0.005;
            std::string offNames;
            double passMs = total;
            for (size_t i = 0; i < shown.gpuSectionMs.size() && i < names.size(); ++i)
            {
                const double ms = std::max(shown.gpuSectionMs[i], 0.0);
                const bool sub  = std::string(names[i]).find('.') != std::string::npos;
                if (!sub) passMs = ms;
                const bool off     = (sub ? std::min(passMs, ms) : ms) < kOff;
                const double share = sub ? ms / std::max(passMs, 1e-6) : ms / total;
                // A pass that is off (or a sub-section below 5 us) is listed below the table, not as a row of zeros
                if (off)
                {
                    if (!offNames.empty()) offNames += ", ";
                    offNames += names[i];
                    continue;
                }
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                if (sub) ImGui::Indent(12);
                ImGui::TextUnformatted(names[i]);
                if (sub) ImGui::Unindent(12);
                cellRight(milliseconds(ms));
                ImGui::TableNextColumn();
                char label[32];
                // A sub-section's bar indented and dimmer: its share of the pass, not of the frame
                std::snprintf(label, sizeof(label), sub ? "%.0f%% of the pass" : "%.0f%% of the GPU time",
                              100.0 * share);
                if (sub) ImGui::Indent(16);
                shareBar(float(share), label, sub ? IM_COL32(130, 95, 45, 255) : IM_COL32(220, 140, 40, 255));
                if (sub) ImGui::Unindent(16);
            }
            cell("frame");
            cellRight(milliseconds(std::max(shown.gpuMs, 0.0)));
            cell(" ");
            ImGui::EndTable();
            if (!offNames.empty()) ImGui::TextDisabled("off or below 5 us: %s", offNames.c_str());
        }
    }

    // ---------------------------------------------------------------------------------------- CPU
    std::snprintf(header, sizeof(header), "CPU per frame (mean of %.0f s)###cpu", kShownSeconds);
    if (subSection(header, false))
    {
        if (beginTable("cpu", 2))
        {
            ImGui::TableSetupColumn("work", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableHeadersRow();
            auto row = [&](const char* name, double ms)
            {
                cell("%s", name);
                cellRight(milliseconds(ms));
            };
            row("terrain node selection", shown.terrainSelectMs);
            row("grass tile culling", shown.grassCullMs);
            row("recording and submit (RenderScene)", shown.recordingMs);
            ImGui::EndTable();
        }
    }

    // ---------------------------------------------------------------------------------------- Geometry
    if (subSection("Geometry", false))
    {
        int terrainTriangles = 0;
        for (int l = 0; l <= ps.terrainMaxLevel && l < 16; ++l) terrainTriangles += ps.trianglesPerLevel[size_t(l)];
        uint32_t treesVisible = fs.visible[0] + fs.visible[1] + fs.visible[2];

        if (beginTable("geometry", 3))
        {
            ImGui::TableSetupColumn("pass", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("instances", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("triangles", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableHeadersRow();

            auto row = [&](const char* name, const std::string& instances, double tris, bool enabled)
            {
                cell("%s%s", name, enabled ? "" : " (off)");
                cellRight(instances);
                cellRight(compact(tris));
            };
            row("terrain", std::to_string(ps.terrainNodes + ps.quarterNodes) + " nodes", terrainTriangles,
                settings.terrain);
            row("water", std::to_string(ps.waterNodes), ps.waterTriangles, settings.water);
            row("trees", compact(treesVisible), double(fs.triangles), settings.trees);
            row("grass", compact(ps.grassClumps), ps.grassTriangles, settings.grass);
            double total = (settings.terrain ? terrainTriangles : 0) + (settings.water ? ps.waterTriangles : 0) +
                           (settings.trees ? double(fs.triangles) : 0) + (settings.grass ? ps.grassTriangles : 0);
            cell("total");
            cellRight("");
            cellRight(compact(total));
            ImGui::EndTable();
        }
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetColorU32(ImGuiCol_TextDisabled));
        ImGui::TextWrapped("grass: clumps of the visible tiles, before the per clump rejection in the shader");
        ImGui::PopStyleColor();
    }

    // ---------------------------------------------------------------------------------------- Terrain
    if (subSection("Terrain levels of detail", false) && beginTable("terrain levels", 3))
    {
        ImGui::TableSetupColumn("level");
        ImGui::TableSetupColumn("nodes");
        ImGui::TableSetupColumn("triangles");
        ImGui::TableHeadersRow();
        for (int l = 0; l <= ps.terrainMaxLevel && l < 16; ++l)
        {
            cell("%d", l);
            cellRight(std::to_string(ps.nodesPerLevel[size_t(l)]));
            cellRight(compact(ps.trianglesPerLevel[size_t(l)]));
        }
        ImGui::EndTable();
    }

    // ---------------------------------------------------------------------------------------- Trees
    if (subSection("Trees", true))
    {
        const auto ends = prototypeTreeLodEnds(settings);
        ImGui::Text("LOD end distances %.0f / %.0f / %.0f m, %d sets", ends[0], ends[1], ends[2], trees.setCount());
        auto& species = trees.treeSpecies();
        if (beginTable("trees", 4))
        {
            ImGui::TableSetupColumn("species");
            ImGui::TableSetupColumn("lod 0");
            ImGui::TableSetupColumn("lod 1");
            ImGui::TableSetupColumn("lod 2");
            ImGui::TableHeadersRow();
            for (size_t sp = 0; sp < species.size() && sp < fs.perSpecies.size(); ++sp)
            {
                cell("%s", species[sp].name);
                for (int l = 0; l < 3; ++l) cellRight(compact(fs.perSpecies[sp][size_t(l)]));
            }
            cell("drawn");
            for (int l = 0; l < 3; ++l) cellRight(compact(fs.visible[size_t(l)]));
            ImGui::EndTable();
        }
        ImGui::TextDisabled("%s of %s trees drawn (frame %llu)", compact(treesVisible(fs)).c_str(),
                            compact(trees.treeCount()).c_str(), (unsigned long long)fs.frame);
    }

    // ---------------------------------------------------------------------------------------- Memory
    if (subSection("GPU memory", false))
    {
        if (beginTable("memory", 2))
        {
            ImGui::TableSetupColumn("resource", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("MB", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableHeadersRow();
            size_t total = 0;
            for (auto& m : memoryUsage())
            {
                cell("%s", m.name);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.2f", m.bytes / (1024.0 * 1024.0));
                cellRight(buf);
                total += m.bytes;
            }
            cell("total");
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f", total / (1024.0 * 1024.0));
            cellRight(buf);
            ImGui::EndTable();
        }
        ImGui::TextDisabled("without the render targets and RenderScene's shared buffers");
    }
}

}  // namespace Vulkan
}  // namespace Saiga
