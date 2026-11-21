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
    // The GPU times of the frames recorded from now on (a scene that rendered before, or another one after a destroy:
    // neither its earlier frames nor the previous scene's frame numbers count); the plots start empty
    lastGpuFrame = scene.frames();
    buckets.clear();
    current        = Bucket();
    sectionCount   = 0;
    shown          = Shown();
    bucketGpuFrame = scene.frames();
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
    auto now      = std::chrono::steady_clock::now();
    double cpuMs  = 0;
    double waitMs = 0;
    if (hasLastFrame)
    {
        cpuMs                 = std::chrono::duration<double, std::milli>(now - lastFrame).count();
        float a               = statistics.frames < 10 ? 0.5f : 0.05f;
        statistics.cpuFrameMs = statistics.cpuFrameMs * (1 - a) + cpuMs * a;
        statistics.fps        = statistics.cpuFrameMs > 0 ? 1000.0 / statistics.cpuFrameMs : 0;
        // The last render (inside this interval) blocked for this long: not work
        waitMs               = std::min(renderScene->lastWaitMs(), cpuMs);
        statistics.cpuWaitMs = statistics.cpuWaitMs * (1 - a) + waitMs * a;
        statistics.cpuWorkMs = statistics.cpuWorkMs * (1 - a) + (cpuMs - waitMs) * a;
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
    // The shown means: the raw values of this bucket (each GPU frame once)
    if (gpu.frame > bucketGpuFrame)
    {
        bucketGpuFrame = gpu.frame;
        // The sections by position (RenderScene::gpuSectionNames, a fixed list); a different layout starts anew
        if (gpu.gpuSectionMs.size() != sectionCount)
        {
            sectionCount = gpu.gpuSectionMs.size();
            current      = Bucket();
            buckets.clear();
        }
        current.gpuSectionSum.resize(sectionCount, 0.0);
        current.gpuSum += gpu.gpuMs;
        for (size_t i = 0; i < gpu.gpuSectionMs.size(); ++i) current.gpuSectionSum[i] += gpu.gpuSectionMs[i];
        current.gpuFrames++;
    }
    if (const PrototypeWorld* w = renderScene->prototypeWorld())
    {
        current.terrainSelectSum += w->stats().cpuTerrainSelectMs;
        current.grassCullSum += w->stats().cpuGrassCullMs;
    }
    current.recordingSum += renderScene->lastRecordingMs();
    current.cpuFrames++;
    rotateBuckets();

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

void WorldRenderer::rotateBuckets()
{
    // Every frame (also while the window is hidden, so the window never covers more than kShownSeconds)
    const auto now = std::chrono::steady_clock::now();
    if (!bucketStarted)
    {
        bucketStarted = true;
        bucketStart   = now;
    }
    // A full bucket enters the window (the buckets of the last kShownSeconds)
    if (now - bucketStart < std::chrono::duration<double>(kRefreshSeconds)) return;
    current.end = now;
    if (current.cpuFrames > 0 || current.gpuFrames > 0) buckets.push_back(current);
    current     = Bucket();
    bucketStart = now;
    // Expired by time (after a pause of the updates the old buckets are older than kShownSeconds)
    while (!buckets.empty() && now - buckets.front().end >= std::chrono::duration<double>(kShownSeconds))
        buckets.pop_front();
    Bucket sum;
    for (const Bucket& b : buckets)
    {
        sum.gpuSum += b.gpuSum;
        sum.gpuSectionSum.resize(std::max(sum.gpuSectionSum.size(), b.gpuSectionSum.size()), 0.0);
        for (size_t i = 0; i < b.gpuSectionSum.size(); ++i) sum.gpuSectionSum[i] += b.gpuSectionSum[i];
        sum.gpuFrames += b.gpuFrames;
        sum.terrainSelectSum += b.terrainSelectSum;
        sum.grassCullSum += b.grassCullSum;
        sum.recordingSum += b.recordingSum;
        sum.cpuFrames += b.cpuFrames;
    }
    shown.valid = true;
    shown.gpuSectionMs.clear();
    if (sum.gpuFrames > 0)
    {
        shown.gpuMs = sum.gpuSum / sum.gpuFrames;
        for (double v : sum.gpuSectionSum) shown.gpuSectionMs.push_back(v / sum.gpuFrames);
    }
    if (sum.cpuFrames > 0)
    {
        shown.terrainSelectMs = sum.terrainSelectSum / sum.cpuFrames;
        shown.grassCullMs     = sum.grassCullSum / sum.cpuFrames;
        shown.recordingMs     = sum.recordingSum / sum.cpuFrames;
    }
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
        {"cpu_wait_ms", s.cpuWaitMs},
        {"cpu_work_ms", s.cpuWorkMs},
        {"vsync", renderScene && renderScene->vsync()},
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

namespace
{
// A sub-section of the settings: a tree node without indent (the debug UI's section header is the parent)
bool subSection(const char* label, bool defaultOpen)
{
    return ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_NoTreePushOnOpen |
                                        (defaultOpen ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None));
}
}  // namespace

void WorldRenderer::imgui()
{
    // Widgets only (no window): the engine's debug UI shows them as a section of its Engine tab (S19c). The frame
    // times are in its Performance tab, not here: one place for them
    auto& st = settings;
    if (subSection("Passes", true))
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
    if (subSection("Level of detail", false))
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
    if (subSection("Lighting", false))
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
}

}  // namespace Vulkan
}  // namespace Saiga
