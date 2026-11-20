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

// Plot of a ring buffer with avg / max in the overlay
void historyPlot(const char* label, const std::vector<float>& values, int offset)
{
    float maxValue = 0, sum = 0;
    int count = 0;
    for (float v : values)
    {
        if (v <= 0) continue;
        maxValue = std::max(maxValue, v);
        sum += v;
        count++;
    }
    float avg = count > 0 ? sum / count : 0;
    ImGui::Text("%s", label);
    ImGui::SameLine();
    ImGui::TextDisabled("avg %.2f ms, max %.2f ms (last %d frames)", avg, maxValue, int(values.size()));
    ImGui::PlotLines("##history", values.data(), int(values.size()), offset, nullptr, 0.0f,
                     std::max(maxValue * 1.15f, 1.0f), ImVec2(-1, 45));
}

uint32_t treesVisible(const PrototypeForest::Stats& s)
{
    return s.visible[0] + s.visible[1] + s.visible[2];
}

}  // namespace

void WorldRenderer::imguiStatistics()
{
    // Right edge of the screen, next to the settings window
    ImVec2 display = ImGui::GetIO().DisplaySize;
    ImGui::SetNextWindowPos(ImVec2(std::max(display.x - 490, 0.0f), 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(480, std::max(display.y - 20, 300.0f)), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("World Statistics", &showStatistics))
    {
        ImGui::End();
        return;
    }
    if (!renderScene)
    {
        ImGui::TextDisabled("not attached to a RenderScene");
        ImGui::End();
        return;
    }
    auto& s                         = statistics;
    const PrototypeWorld* pw        = renderScene->prototypeWorld();
    const PrototypeWorld::Stats ps  = pw ? pw->stats() : PrototypeWorld::Stats();
    const PrototypeForest::Stats fs = trees.stats();

    // ---------------------------------------------------------------------------------------- Overview
    ImGui::Text("%.1f fps   frame %.2f ms   GPU %.2f ms   %d draw calls", s.fps, s.cpuFrameMs, s.gpuMs, s.drawCalls);
    ImGui::TextDisabled("%dx MSAA, reversed depth, %.1f km world, %d trees%s", int(renderScene->formats().samples),
                        heightField().worldSize() / 1000.0f, trees.treeCount(),
                        settings.depthPrePass ? ", depth pre-pass" : "");
    ImGui::PushID("cpu");
    historyPlot("frame (CPU)", cpuHistory, historyIndex);
    ImGui::PopID();
    ImGui::PushID("gpu");
    historyPlot("frame (GPU)", gpuHistory, historyIndex);
    ImGui::PopID();

    // ---------------------------------------------------------------------------------------- GPU sections
    if (ImGui::CollapsingHeader("GPU sections", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (s.gpuSectionMs.empty())
        {
            ImGui::TextDisabled("No GPU times (no timestamps, or the app does not take them).");
        }
        else if (beginTable("gpu sections", 3))
        {
            ImGui::TableSetupColumn("section", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_WidthStretch, 1.5f);
            ImGui::TableHeadersRow();
            const auto& names = RenderScene::gpuSectionNames();
            double total      = std::max(s.gpuMs, 1e-6);
            for (size_t i = 0; i < s.gpuSectionMs.size() && i < names.size(); ++i)
            {
                double ms = std::max(s.gpuSectionMs[i], 0.0);
                // The sub-sections (dotted names) indented below their pass
                const bool sub = std::string(names[i]).find('.') != std::string::npos;
                cell("%s%s", sub ? "  " : "", names[i]);
                cellRight(milliseconds(ms));
                ImGui::TableNextColumn();
                char label[16];
                std::snprintf(label, sizeof(label), "%.0f%%", 100.0 * ms / total);
                ImGui::ProgressBar(float(ms / total), ImVec2(-1, 0), label);
            }
            cell("frame");
            cellRight(milliseconds(std::max(s.gpuMs, 0.0)));
            cell(" ");
            ImGui::EndTable();
        }
    }

    // ---------------------------------------------------------------------------------------- CPU
    if (ImGui::CollapsingHeader("CPU per frame", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (beginTable("cpu", 2))
        {
            ImGui::TableSetupColumn("work", ImGuiTableColumnFlags_WidthStretch, 2.0f);
            ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableHeadersRow();
            auto row = [&](const char* name, double ms)
            {
                cell("%s", name);
                char buf[32];
                std::snprintf(buf, sizeof(buf), "%.3f", ms);
                cellRight(buf);
            };
            row("terrain node selection", s.cpuTerrainSelectMs);
            row("grass tile culling", s.cpuGrassCullMs);
            row("recording and submit (RenderScene)", renderScene->lastRecordingMs());
            row("frame (everything)", s.cpuFrameMs);
            ImGui::EndTable();
        }
    }

    // ---------------------------------------------------------------------------------------- Geometry
    if (ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_DefaultOpen))
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
        ImGui::TextDisabled("grass: clumps of the visible tiles, before the per clump rejection in the shader");
    }

    // ---------------------------------------------------------------------------------------- Terrain
    if (ImGui::CollapsingHeader("Terrain levels of detail") && beginTable("terrain levels", 3))
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
    if (ImGui::CollapsingHeader("Trees", ImGuiTreeNodeFlags_DefaultOpen))
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
    if (ImGui::CollapsingHeader("GPU memory"))
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

    ImGui::End();
}

}  // namespace Vulkan
}  // namespace Saiga
