/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/world/WorldRenderer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/scene/PrototypeWorld.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace Saiga
{
namespace Vulkan
{
void WorldRenderer::init(const WorldParameters& parameters)
{
    destroy();
    params         = parameters;
    auto generated = std::make_shared<HeightField>();
    auto t0        = std::chrono::steady_clock::now();
    generated->generate(params.terrain);
    hf = std::move(generated);
    std::cout << "World: terrain " << hf->size() << "^2 (" << hf->worldSize() / 1000.0f << " km) generated in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << std::endl;
}

void WorldRenderer::init(const WorldParameters& parameters, std::shared_ptr<const HeightField> heightField)
{
    if (!heightField) SAIGA_EXIT_ERROR("WorldRenderer::init: no height field");
    destroy();
    params         = parameters;
    hf             = std::move(heightField);
    params.terrain = hf->parameters();
}

void WorldRenderer::attach(RenderScene& scene)
{
    SAIGA_ASSERT(hf, "WorldRenderer::attach: init first");
    SAIGA_ASSERT(!renderScene, "WorldRenderer::attach: once");
    auto t0     = std::chrono::steady_clock::now();
    renderScene = &scene;
    settings.clamp();
    scene.setPrototypeTerrain(hf, settings.prototype(), params.seed);
    trees.init(scene, *hf, params.treeCount, params.seed, settings);
    std::cout << "World: " << trees.treeCount() << " trees in " << trees.setCount() << " sets, GPU resources in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << " s" << std::endl;
}

void WorldRenderer::destroy()
{
    if (renderScene) trees.clear();
    trees        = PrototypeForest();
    renderScene  = nullptr;
    hasLastFrame = false;
    statistics   = Stats();
    hf.reset();
}

const HeightField& WorldRenderer::heightField() const
{
    static const HeightField empty;
    return hf ? *hf : empty;
}

mat4 WorldRenderer::projection(float aspect) const
{
    return reversedZInfinitePerspective(radians(settings.fovy), aspect, 0.1f);
}

void WorldRenderer::smooth(double& average, double value, int frames)
{
    double a = frames < 10 ? 0.5 : 0.05;
    average  = average < 0 ? value : average * (1 - a) + value * a;
}

void WorldRenderer::update()
{
    if (!renderScene) return;
    settings.clamp();
    renderScene->setPrototypeSettings(settings.prototype());
    trees.apply(settings);

    // Frame timing (between two updates)
    auto now     = std::chrono::steady_clock::now();
    double cpuMs = 0;
    if (hasLastFrame)
    {
        cpuMs                 = std::chrono::duration<double, std::milli>(now - lastFrame).count();
        float a               = statistics.frames < 10 ? 0.5f : 0.05f;
        statistics.cpuFrameMs = statistics.cpuFrameMs * (1 - a) + cpuMs * a;
        statistics.fps        = statistics.cpuFrameMs > 0 ? 1000.0 / statistics.cpuFrameMs : 0;
    }
    lastFrame    = now;
    hasLastFrame = true;
    statistics.frames++;

    // RenderScene's newest GPU frame (taken by the app)
    const FrameTimes& gpu = renderScene->lastGpuFrameTimes();
    if (gpu.frame > lastGpuFrame)
    {
        lastGpuFrame = gpu.frame;
        smooth(statistics.gpuMs, gpu.gpuMs, statistics.frames);
        statistics.gpuSectionMs.resize(gpu.gpuSectionMs.size(), -1);
        for (size_t i = 0; i < gpu.gpuSectionMs.size(); ++i)
            smooth(statistics.gpuSectionMs[i], gpu.gpuSectionMs[i], statistics.frames);
    }
    if (cpuMs > 0)
    {
        cpuHistory[size_t(historyIndex)] = float(cpuMs);
        gpuHistory[size_t(historyIndex)] = float(std::max(statistics.gpuMs, 0.0));
        historyIndex                     = (historyIndex + 1) % historySize;
    }

    // The prototype's CPU statistics of the last recorded frame, the trees from the newest completed frame
    if (const PrototypeWorld* w = renderScene->prototypeWorld())
    {
        const auto& ps = w->stats();
        smooth(statistics.cpuTerrainSelectMs, ps.cpuTerrainSelectMs, statistics.frames);
        smooth(statistics.cpuGrassCullMs, ps.cpuGrassCullMs, statistics.frames);
        statistics.terrainNodes     = ps.terrainNodes + ps.quarterNodes;
        statistics.terrainTriangles = ps.terrainTriangles;
        statistics.grassClumps      = ps.grassClumps;
        statistics.grassTriangles   = ps.grassTriangles;
        statistics.drawCalls        = ps.drawCalls + int(renderScene->statistics().drawCommands);
    }
    const auto ts            = trees.stats();
    statistics.treesVisible  = ts.visible;
    statistics.treeTriangles = ts.triangles;
}

std::vector<WorldRenderer::MemoryEntry> WorldRenderer::memoryUsage() const
{
    const PrototypeWorld* w = renderScene ? renderScene->prototypeWorld() : nullptr;
    return {{"terrain, grass, foliage texture, uniforms", w ? w->memoryBytes() : 0},
            {"trees (instances, meshes)", trees.memoryBytes()}};
}

json11::Json WorldRenderer::statsJson() const
{
    auto& s = statistics;
    json11::Json::object gpuSections;
    const auto& names = RenderScene::gpuSectionNames();
    for (size_t i = 0; i < s.gpuSectionMs.size() && i < names.size(); ++i) gpuSections[names[i]] = s.gpuSectionMs[i];
    size_t memory = 0;
    for (auto& m : memoryUsage()) memory += m.bytes;
    const int samples = renderScene ? int(renderScene->formats().samples) : 0;
    return json11::Json::object{
        {"fps", s.fps},
        {"cpu_frame_ms", s.cpuFrameMs},
        {"gpu_frame_ms", s.gpuMs},
        {"terrain_nodes", s.terrainNodes},
        {"terrain_triangles", s.terrainTriangles},
        {"trees_total", trees.treeCount()},
        {"trees_visible_lod0", int(s.treesVisible[0])},
        {"trees_visible_lod1", int(s.treesVisible[1])},
        {"trees_visible_lod2", int(s.treesVisible[2])},
        {"tree_triangles", double(s.treeTriangles)},
        {"grass_clumps_submitted", s.grassClumps},
        {"grass_triangles_submitted", s.grassTriangles},
        {"msaa_samples", samples},
        {"draw_calls", s.drawCalls},
        {"gpu_section_ms", gpuSections},
        {"cpu_ms", json11::Json::object{{"terrain_select", s.cpuTerrainSelectMs}, {"grass_cull", s.cpuGrassCullMs}}},
        {"gpu_memory_mb", double(memory) / (1024.0 * 1024.0)},
    };
}

void WorldRenderer::imgui()
{
    if (!ImGui::Begin("World"))
    {
        ImGui::End();
        return;
    }
    auto& s = statistics;
    ImGui::Text("%.1f fps  cpu %.2f ms  gpu %.2f ms", s.fps, s.cpuFrameMs, s.gpuMs);
    ImGui::Text("terrain: %d nodes, %.2f M tris", s.terrainNodes, s.terrainTriangles / 1e6);
    ImGui::Text("trees: %u / %u / %u of %d, %.2f M tris", s.treesVisible[0], s.treesVisible[1], s.treesVisible[2],
                trees.treeCount(), double(s.treeTriangles) / 1e6);
    ImGui::Text("grass: %d clumps, %.2f M tris (submitted)", s.grassClumps, s.grassTriangles / 1e6);

    ImGui::Checkbox("statistics window", &showStatistics);

    auto& st = settings;
    if (ImGui::CollapsingHeader("Passes", ImGuiTreeNodeFlags_DefaultOpen))
    {
        ImGui::Checkbox("terrain", &st.terrain);
        ImGui::SameLine();
        ImGui::Checkbox("trees", &st.trees);
        ImGui::SameLine();
        ImGui::Checkbox("grass", &st.grass);
        ImGui::Checkbox("water", &st.water);
        ImGui::SameLine();
        ImGui::Checkbox("sky", &st.sky);
        ImGui::SameLine();
        ImGui::Checkbox("fog", &st.fog);
        ImGui::Checkbox("flat shading", &st.flatShading);
        ImGui::SameLine();
        ImGui::Checkbox("depth pre-pass", &st.depthPrePass);
        ImGui::Combo("debug", &st.debugMode, "none\0lod tint\0\0");
    }
    if (ImGui::CollapsingHeader("Level of detail"))
    {
        ImGui::SliderFloat("terrain lod range", &st.terrainLodRange, 100, 2000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("morph start", &st.morphStart, 0.3f, 0.95f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree lod1", &st.treeLod1, 10, 500, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree lod2", &st.treeLod2, 50, 3000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("tree max distance", &st.treeMaxDistance, 500, 50000, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass radius", &st.grassRadius, 0, 200, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass cell size", &st.grassCellSize, 0.1f, 2.0f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("grass density", &st.grassDensity, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("alpha cutoff", &st.alphaCutoff, 0.05f, 0.95f, "%.3f", ImGuiSliderFlags_AlwaysClamp);
    }
    if (ImGui::CollapsingHeader("Lighting"))
    {
        ImGui::SliderFloat("fovy", &st.fovy, 20, 110, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun azimuth", &st.sunAzimuth, -180, 180, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun elevation", &st.sunElevation, -10, 90, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("sun intensity", &st.sunIntensity, 0, 2, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("ambient", &st.ambient, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("albedo", &st.albedo, 0, 1, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("wind", &st.wind, 0, 4, "%.3f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("fog density", &st.fogDensity, 0, 0.002f, "%.5f", ImGuiSliderFlags_AlwaysClamp);
        ImGui::SliderFloat("fog falloff", &st.fogFalloff, 0.0001f, 0.01f, "%.5f", ImGuiSliderFlags_AlwaysClamp);
    }
    ImGui::End();

    if (showStatistics) imguiStatistics();
}

}  // namespace Vulkan
}  // namespace Saiga
