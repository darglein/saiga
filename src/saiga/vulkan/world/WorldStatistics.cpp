/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
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
    auto& s  = statistics;
    auto& ts = terrainRenderer.stats();
    auto& fs = forestRenderer.stats();
    auto& gs = grassRenderer.stats();

    // ---------------------------------------------------------------------------------------- Overview
    ImGui::Text("%.1f fps   frame %.2f ms   GPU (world) %.2f ms   %d draw calls", s.fps, s.cpuFrameMs, s.gpuMs,
                s.drawCalls);
    ImGui::TextDisabled("%dx MSAA, reversed depth, %.1f km world, %d trees%s", int(formats.samples),
                        hf->worldSize() / 1000.0f, forestRenderer.treeCount(),
                        settings.freezeCulling ? ", CULLING FROZEN" : "");
    ImGui::PushID("cpu");
    historyPlot("frame (CPU)", cpuHistory, historyIndex);
    ImGui::PopID();
    ImGui::PushID("gpu");
    historyPlot("world (GPU)", gpuHistory, historyIndex);
    ImGui::PopID();

    // ---------------------------------------------------------------------------------------- GPU passes
    if (ImGui::CollapsingHeader("GPU passes", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (!timestampsSupported)
        {
            ImGui::TextDisabled("Timestamps are not supported on this device.");
        }
        else if (beginTable("gpu passes", 3))
        {
            ImGui::TableSetupColumn("pass", ImGuiTableColumnFlags_WidthStretch, 1.2f);
            ImGui::TableSetupColumn("ms", ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("share", ImGuiTableColumnFlags_WidthStretch, 1.5f);
            ImGui::TableHeadersRow();
            double total = std::max(s.gpuMs, 1e-6);
            for (int p = 0; p < PASS_COUNT; ++p)
            {
                double ms = std::max(s.gpuPassMs[p], 0.0);
                cell("%s", gpuPassName(p));
                cellRight(milliseconds(ms));
                ImGui::TableNextColumn();
                char label[16];
                std::snprintf(label, sizeof(label), "%.0f%%", 100.0 * ms / total);
                ImGui::ProgressBar(float(ms / total), ImVec2(-1, 0), label);
            }
            cell("total");
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
            row("transfer() total (uploads, culling)", s.cpuTransferMs);
            row("render() command recording", s.cpuRenderMs);
            row("frame (everything)", s.cpuFrameMs);
            ImGui::EndTable();
        }
    }

    // ---------------------------------------------------------------------------------------- Geometry
    if (ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_DefaultOpen))
    {
        int terrainTriangles = 0;
        for (int l = 0; l <= terrainRenderer.maxLevel(); ++l) terrainTriangles += ts.trianglesPerLevel[l];
        int waterTriangles = ts.waterNodes * TerrainRenderer::gridResolution * TerrainRenderer::gridResolution / 2;
        int treesVisible   = fs.visible[0] + fs.visible[1] + fs.visible[2];

        if (beginTable("geometry", 4))
        {
            ImGui::TableSetupColumn("pass", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableSetupColumn("draws", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("instances", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableSetupColumn("triangles", ImGuiTableColumnFlags_WidthStretch, 0.9f);
            ImGui::TableHeadersRow();

            auto row =
                [&](const char* name, const std::string& draws, const std::string& instances, double tris, bool enabled)
            {
                cell("%s%s", name, enabled ? "" : " (off)");
                cellRight(draws);
                cellRight(instances);
                cellRight(compact(tris));
            };
            row("terrain", std::to_string(int(ts.nodes > 0) + int(ts.quarterNodes > 0)),
                std::to_string(ts.nodes + ts.quarterNodes) + " nodes", terrainTriangles, settings.terrain);
            row("water", std::to_string(ts.waterNodes), std::to_string(ts.waterNodes), waterTriangles, settings.water);
            row("trees", "1 MDI / " + std::to_string(forestRenderer.drawCommands()), compact(treesVisible),
                fs.triangles, settings.trees);
            row("grass", std::to_string(gs.tiles) + " tiles", compact(gs.clumps), gs.triangles, settings.grass);
            row("sky", settings.sky ? "1" : "0", "-", settings.sky ? 1 : 0, settings.sky);

            double total = (settings.terrain ? terrainTriangles : 0) + (settings.water ? waterTriangles : 0) +
                           (settings.trees ? fs.triangles : 0) + (settings.grass ? gs.triangles : 0);
            cell("total");
            cellRight(std::to_string(s.drawCalls));
            cellRight("");
            cellRight(compact(total));
            ImGui::EndTable();
        }
        ImGui::TextDisabled("grass: clumps of the visible tiles, before the per clump rejection in the shader");
    }

    // ---------------------------------------------------------------------------------------- Terrain
    if (ImGui::CollapsingHeader("Terrain levels of detail"))
    {
        ImGui::Text("level 0 range %.0f m, morph from %.0f%% of each range", terrainRenderer.range0(),
                    settings.morphStart * 100);
        if (beginTable("terrain levels", 5))
        {
            ImGui::TableSetupColumn("level");
            ImGui::TableSetupColumn("spacing");
            ImGui::TableSetupColumn("range");
            ImGui::TableSetupColumn("nodes");
            ImGui::TableSetupColumn("triangles");
            ImGui::TableHeadersRow();
            for (int l = 0; l <= terrainRenderer.maxLevel(); ++l)
            {
                float spacing = terrainRenderer.nodeSize(l) / TerrainRenderer::gridResolution;
                cell("%d", l);
                cellRight(std::to_string(int(spacing)) + " m");
                cellRight(l == terrainRenderer.maxLevel() ? "inf"
                                                          : compact(terrainRenderer.range0() * float(1 << l)) + " m");
                cellRight(std::to_string(ts.nodesPerLevel[l]));
                cellRight(compact(ts.trianglesPerLevel[l]));
            }
            ImGui::EndTable();
        }
    }

    // ---------------------------------------------------------------------------------------- Trees
    if (ImGui::CollapsingHeader("Trees", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Text("lod 0 < %.0f m, lod 1 < %.0f m, lod 2 < %.0f m", settings.treeLod1, settings.treeLod2,
                    settings.treeMaxDistance);
        auto& species  = forestRenderer.treeSpecies();
        auto& meshTris = forestRenderer.bucketTriangleCounts();
        if (beginTable("trees", 1 + ForestRenderer::lodCount + 1))
        {
            ImGui::TableSetupColumn("species");
            for (int l = 0; l < ForestRenderer::lodCount; ++l)
            {
                static const char* names[] = {"lod 0", "lod 1", "lod 2"};
                ImGui::TableSetupColumn(names[l]);
            }
            ImGui::TableSetupColumn("triangles");
            ImGui::TableHeadersRow();

            std::array<double, ForestRenderer::lodCount> lodTris = {};
            for (size_t sp = 0; sp < species.size(); ++sp)
            {
                cell("%s", species[sp].name);
                double tris = 0;
                for (int l = 0; l < ForestRenderer::lodCount; ++l)
                {
                    size_t b    = sp * ForestRenderer::lodCount + l;
                    int visible = b < fs.bucketVisible.size() ? fs.bucketVisible[b] : 0;
                    cellRight(compact(visible));
                    tris += double(visible) * meshTris[b];
                    lodTris[l] += double(visible) * meshTris[b];
                }
                cellRight(compact(tris));
            }
            cell("visible");
            for (int l = 0; l < ForestRenderer::lodCount; ++l) cellRight(compact(fs.visible[l]));
            cellRight(compact(fs.triangles));
            cell("triangles");
            for (int l = 0; l < ForestRenderer::lodCount; ++l) cellRight(compact(lodTris[l]));
            cellRight("");
            ImGui::EndTable();
        }
        if (ImGui::TreeNode("triangles per tree mesh"))
        {
            if (beginTable("tree meshes", 1 + ForestRenderer::lodCount))
            {
                ImGui::TableSetupColumn("species");
                ImGui::TableSetupColumn("lod 0");
                ImGui::TableSetupColumn("lod 1");
                ImGui::TableSetupColumn("lod 2");
                ImGui::TableHeadersRow();
                for (size_t sp = 0; sp < species.size(); ++sp)
                {
                    cell("%s", species[sp].name);
                    for (int l = 0; l < ForestRenderer::lodCount; ++l)
                        cellRight(std::to_string(meshTris[sp * ForestRenderer::lodCount + l]));
                }
                ImGui::EndTable();
            }
            ImGui::TreePop();
        }
        int total = fs.visible[0] + fs.visible[1] + fs.visible[2];
        ImGui::TextDisabled("%s of %s trees pass the frustum and distance test (%.1f%%)", compact(total).c_str(),
                            compact(forestRenderer.treeCount()).c_str(),
                            100.0 * total / std::max(forestRenderer.treeCount(), 1));
    }

    // ---------------------------------------------------------------------------------------- Grass
    if (ImGui::CollapsingHeader("Grass"))
    {
        int tiles = GrassRenderer::tiles * GrassRenderer::tiles;
        ImGui::Text("radius %.0f m, cell %.2f m, density %.2f", settings.grassRadius, settings.grassCellSize,
                    settings.grassDensity);
        ImGui::Text("%d / %d tiles visible, %d x %d cells per tile", gs.tiles, tiles, gs.tileCells, gs.tileCells);
        ImGui::Text("%s clumps submitted, %s triangles", compact(gs.clumps).c_str(), compact(gs.triangles).c_str());
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
        ImGui::TextDisabled("without the render targets of the renderer");
    }

    ImGui::End();
}

}  // namespace Vulkan
}  // namespace Saiga
